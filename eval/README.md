# eval/ — the harness every change is judged by

Spec: `docs/EVAL.md`. Owner: T09. Deps: `pip install -r eval/requirements.txt`
(pyarrow for the canonical parquet; scipy/sklearn/motmetrics only for cross-check tests).

```bash
python -m eval.run --list
python -m eval.run --suite public_pose --models behavior=eval.models.dummy:WristMotionBehavior
scripts/gpu shared -- python -m eval.run --suite meva_interaction --suite meva_fa \
    --pipeline scs.app.pipeline:build --policy configs/policy.yaml --workers 8 --compare main
python -m eval.converters meva            # raw (T14) -> data/meva/converted/
python -m eval.make_splits meva           # regenerate frozen split IDs (policy lives in code)
```

Outputs: `runs/eval/<git-sha>/<suite>.json` + `summary.md` (headline metrics with 95% CI
and sample counts, labels such as "proxy, not retail", git sha, models + lineage, dataset
and split digests, policy hash, GPU/driver). `--compare main` adds a delta table using
`runs/eval/<sha of main>/`; run the baseline at `main` first (a worktree works).

## Layout

| path | what |
|---|---|
| `metrics/` | pure-numpy metrics returning `MetricValue` (value, CI, n, or "unavailable") |
| `canonical.py` | on-disk format (tracks parquet, labels JSON, frame labels) |
| `datasets.py` | **loader API** (below) |
| `splits.py`, `splits/*.json`, `make_splits.py` | group-aware splits; frozen IDs; policy |
| `converters/` | PoseLift, RetailS (refuses while `pending`), MEVA, SmartSpaces, take log, sim stub |
| `suites/` | one module per suite; auto-discovered |
| `pose_bench.py` | pose-window scoring + default `public_pose` |
| `e2e.py` | streaming driver for end-to-end suites |
| `models/` | dummy behavior models, harness-validation pipelines (not product models) |

## Loader API (T06 and anyone training/evaluating)

```python
from eval.datasets import load_dataset

ds = load_dataset("poselift")            # license gate + frozen split + converted data
for clip in ds.clips("train"):           # never read raw files: splits/groups live here
    t = clip.tracks()                    # TrackTable: frame_idx, ts, track_id, boxes (N,4), kps (N,17,3), score
    y = clip.frame_labels()              # per-frame 0/1 or None
    clip.labels                          # ClipLabels: events, fps, camera_id, group_id, supports, ...
    for seq in clip.pose_sequences(min_len=24):
        seq.kps, seq.frame_idx           # one track, frame order
for w in ds.windows("train", window=24, stride=6):
    w.kps, w.label, w.group_id           # causal window; label = frame label at its last frame
ds.version()                             # what reports record
```

Rules the API enforces: only `train` for training, `val` for calibration/thresholds,
`test` untouched; derivatives you create (crops, augmentations, emulated variants) must
carry the source clip's `group_id` and resolve via `ds.splits.split_for(clip_id, group_id)`.

## Adding a suite

1. Create `eval/suites/<name>.py`:

```python
from eval.datasets import load_dataset
from eval.metrics.frame import frame_auc
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite

@register_suite
class MySuite(Suite):
    name = "my_suite"
    description = "one line for --list"
    datasets = ("poselift",)
    labels = ("synthetic",)        # printed in every report header; be honest
    continuous = False             # True only for uncut footage: enables false alerts/hour
    end_to_end = False             # True: must drive the streaming pipeline (no saved predictions)
    gated = ("auc_roc",)           # headline keys the 2-point regression rule applies to

    def run(self, ctx: RunContext) -> SuiteResult:
        try:
            ds = load_dataset("poselift", ctx.data_root, ctx.split_dir)
        except FileNotFoundError as e:
            return self.unavailable(str(e))          # never a fake number
        ...
        return self.result("ok", {"auc_roc": auc["auc_roc"]}, datasets=[ds.version()])
```

2. Headline values must be `MetricValue`s (use `eval.metrics`; they carry CI + counts).
   Report "unavailable" with a reason whenever the labels can't support a metric.
3. Thresholds: fit on `val`, apply to `test` (`threshold_at_budget`). Label oracle numbers.
4. End-to-end suites call `eval.suites._common.run_e2e(ctx, ds, clip_ids)`; never read scores
   from disk. Continuous-footage FA only (`continuous = True`).
5. Add a test in `tests/eval/` with a tiny synthetic dataset and a hand-known answer
   (see `test_e2e_suites.py`: GT replay → recall 1.0; periodic alerts → exact FA/h).

A file with the name of a framework default (e.g. `public_pose.py`, owned by T06)
replaces the default automatically. `soak*`/`perf*` belong to T12: a soak suite is
`meva_fa`-style FA/h plus `run_e2e(..., realtime=True)` over 10 streams with crash count
(`crash_summary`) and RSS sampling.

## Streaming contract for end-to-end suites

```python
def build(camera_id: str, fps: float, policy: dict | None, models: dict[str, ModelSpec], clip_id: str, **_):
    return pipeline   # .process(ref: FrameRef, image) -> list[Event | Alert | Track]; .flush(now) -> list[...]
```

- Frames arrive once, in order (causal). Emit items as soon as the pipeline would.
- Timestamps you emit (`Event.ts`, `Alert.ts_open`, optional `Event.data["t_start"/"t_end"]`)
  must be `FrameRef.ts` values of frames already seen: replay runs faster than real time,
  so only real frame timestamps map back to media time.
- Interaction detections: `Event` of type `item_pickup`/`shelf_interaction`, or any Event
  with `data["interaction"]` = a canonical label; score = `Event.confidence`.
- `eval.e2e.ProtocolPipeline` composes Detector → Tracker → PoseEstimator →
  BehaviorModel → JourneyEngine (ARCHITECTURE §4) and is a valid `build` body.
- A crash in one clip is caught, reported, and marks the result `partial`.

## Models / lineage

`--models role=module:attr[?{"kw": 1}]`. Give model objects `model_id` and
`lineage = {"train": [{"dataset": "meva", "split": "train"}, ...], "license": "..."}`:
`meva_fa` uses it to decide which footage is held out; `sim_transfer` refuses variants
trained on its test split. No lineage = treated as "may have trained on anything".
