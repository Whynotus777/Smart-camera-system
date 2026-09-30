# T09 — eval harness & dataset converters: report

Branch stack `agent/T09-eval-harness` → `agent/T09-data` → `agent/T09-suites`, from `wave0-baseline`.
Date: 2026-09-29/30. Box: RTX 5090 workstation (shared by 8 agents during these runs).

## What exists now

| piece | where | status |
|---|---|---|
| Metrics: event recall @ FA budget + full curve, per-subtype recall, frame AUC-ROC/PR, IDF1 / ID switches / MOTA / det AP, journey F1, latency percentiles; bootstrap 95% CI + counts on each | `eval/metrics/` | done; hand-computed toy tests incl. overlapping + duplicate alerts; cross-checked vs sklearn, scipy, motmetrics |
| Canonical format + loader API | `eval/canonical.py`, `eval/datasets.py` | done |
| Split generator (actor / clip / camera / site / scene, derivative `group_id`s, view-linked leak exclusion) | `eval/splits.py`, `eval/make_splits.py` | done; CI test fails if any group spans splits |
| Frozen split IDs | `eval/splits/meva.json`, `smartspaces.json` | committed; `poselift.json` waits for pose files (see Blockers) |
| Converters | `eval/converters/` | MEVA indoor, SmartSpaces, PoseLift: done + run on real data (PoseLift: only its test masks are downloaded yet). RetailS: written to the documented format, **refuses to run** (`pending`). Take log (`quick_capture` / `lab_mock_aisle`): done, fixture-tested. Sim: stub. |
| Suites | `eval/suites/` + `eval/pose_bench.py` | `meva_interaction`, `meva_fa`, `smartspaces_track`, `lab_e2e`, `sim_transfer`, default `public_pose` |
| Runner, reports, `--compare` | `eval/run.py`, `eval/report.py` | done |
| Streaming e2e driver | `eval/e2e.py` | done; validated on real MEVA video with reference pipelines |

Tests: `pytest -m "not gpu and not data and not slow"` → 150 passed, 12 skipped (whole repo);
`tests/eval/` alone 87 passed; `pytest -m data tests/eval/test_real_data.py` → 2 passed, 1 skipped
(PoseLift pose files not downloaded).

## Acceptance (tasks/T09 + addenda)

- [x] Unit tests per metric on hand-computed toy cases incl. overlapping and duplicate alerts
  (`tests/eval/test_events.py`, `test_frame_latency.py`, `test_tracking_journey.py`).
- [x] `public_pose` < 5 min on the 5090 with a dummy model — see "PoseLift runtime". Measured on
  PoseLift-shaped data because PoseLift's pose files aren't downloaded yet (T14: Drive throttling,
  44/329 files); re-run on real files is one command once they land.
- [x] Reports include git sha (+dirty), model ids + lineage, dataset versions (converter version,
  T14 manifest digest, split version + digest), policy hash, GPU + driver (`nvidia-smi`, read-only).
- [x] Derivative groups; test fails if a group spans splits (`test_splits.py`, incl. committed files).
- [x] `lab_e2e` / `soak` drive the streaming pipeline from video: `lab_e2e` implemented via `eval.e2e`;
  runner refuses `--predictions` for any end-to-end suite. `soak*` is T12-owned (tasks/README):
  framework support provided (realtime replay, crash summary), suite file not written by T09.
- [x] FA/h only from continuous footage: enforced in `SuiteResult` (raises otherwise).
- [x] Bootstrap 95% CIs over clips/sequences + sample counts on every headline metric; LOW-N flag < 30.
- [x] Label honesty: `LabelSupport` per clip; unsupported metrics → "unavailable" with reason;
  positives = `visible: observed` for that camera.
- [x] Thresholds fit on val, frozen for test; oracle numbers labeled "oracle".
- [x] RetailS converter written against the documented format, not downloaded, refuses while `pending`.
- [x] `sim_transfer` suite (paired bootstrap vs `real`, leak guard on lineage).
- [x] MEVA converter (activities → interaction labels), held-out split by camera AND site;
  SmartSpaces converter; suites `meva_interaction`, `meva_fa`, `smartspaces_track`; "proxy, not retail"
  in every MEVA report header.
- [x] `eval/README.md`: how to add a suite.
- [ ] `meva_fa` ≥ 100 held-out camera-hours: **not reachable yet** (see Blockers).

## PoseLift runtime (dummy model)

`scripts/gpu shared -- python -m eval.run --suite public_pose --models behavior=eval.models.dummy:WristMotionBehavior`
(CPU work; run under the lock anyway). Data = synthetic poses laid out exactly like PoseLift's STG-NF
release, 3 people per frame.

| data | clips | frames | windows scored | wall (convert) | wall (suite, B=1000) |
|---|---|---|---|---|---|
| PoseLift **real test masks** (44 clips, real names/lengths) + synthetic poses | 44 | 3,603 | 7,812 | 1.8 s | **3.4 s** |
| full-dataset scale (155 clips, 57,240 frames ≈ 1.06 h) | 155 | 57,240 | 161,025 | 7.3 s | **13.0 s** |

The first version took 205 s at full scale; 190 s of it was the clip bootstrap re-sorting all frames
per replicate. It now uses exact multiplicity weights over one sort (`_WeightedBlocks`), verified equal
to naive concatenation in `test_fast_weighted_bootstrap_equals_naive_concatenation`.

## Real-data validation of the harness itself

Reference pipelines with hand-known answers (`eval/models/reference_pipelines.py`), real video:

- `meva_interaction` + `GTReplayPipeline` (emits the labels causally through the real decode path):
  all 271 converted val+test clips (2,439,799 frames decoded, 0 crashes): **recall 1.000 [1.000, 1.000]
  on 1,507 test positives / 204 clips / 17.0 camera-hours, 0 false, 0 duplicates, latency p95 0.0 s**,
  threshold fit on val (school.G423) and frozen; per-type recall 1.0 for pickup / put-down / transfer.
  The first full run returned recall 0.0, which exposed a real matching bug: a prediction on an
  ignore region (MEVA `not_good`) used the 2-s tolerance to grab the next positive, cascading into
  duplicates that pushed val over budget. Fixed (true intersection beats tolerance-only overlap)
  and pinned by `test_alert_on_ignore_region_does_not_steal_nearby_positive`; re-scored from the
  cached pipeline outputs at `7d22d56`.
- Budget resolution: val is 5.6 camera-hours, so at 0.1/h the fit tolerates zero val false
  detections. The suite notes this; see decision 4.
- `smartspaces_track` with GT as predictions (1,500 frames × 16 test cameras): IDF1 1.000, AP@0.5
  1.000, 0 ID switches (sanity only; correctness of IDF1/switches is from the motmetrics cross-check).
- Synthetic e2e tests: periodic alerts every 5 s → exactly 540 false alerts per camera-hour on
  20-s clips; null pipeline → recall 0; pipeline crash → reported, clip excluded, status `partial`.

## Data as converted (shared `data/<id>/converted/`)

| dataset | clips | notes |
|---|---|---|
| meva | 670 (55.9 h), all annotated so far | split: train 399 (33.3 h), val 67 (5.6 h), test 204 (17.0 h, 1,584 events). Indexed: 1,396 indoor clips; split file lists all of them. |
| smartspaces | 78 camera-videos, scenes 071-075; GT for 071-074 | 6.03 M GT boxes; scene_071/camera_0649 skipped (corrupt per README) |
| poselift | test masks only | 44 test clips, 3,603 frames, 1,344 positive frames |

## Decisions made (reviewers: push back here)

1. **Duplicates count toward the FA budget** (review burden) and are always reported separately.
2. **One alert credits at most one event** (greedy by score, best temporal IoU), so a long alert
   can't claim every event it touches.
3. MEVA labels are mapped honestly: `puts_down` → `item_put_down` (not `item_returned`),
   `transfers` → `item_transfer`, `steals` → `item_pickup` with `source_label` kept. New vocabulary
   documented in docs/EVAL.md; docs/DATA.md needs the same (handoff).
4. `meva_interaction` budget = 0.1 false *interaction detections* per camera-hour, fit on val
   (`school.G423`); recall also reported at 1 and 10 per camera-hour (oracle).
5. MEVA split: test = site `bus` + camera `school.G421`, val = `school.G423`. Admin cameras hold
   almost no interaction events, so holding out site `admin` would test nothing. MEVA camera sets
   (shared FOV, from the clip table) define `view_group`, finer than T14's slot+site: cameras
   that don't share a view don't see the same event.
6. `meva_fa` pool depends on declared lineage: no lineage → held-out split only (conservative).
7. `public_pose` fill for frames without a full window = clip minimum (STG-NF convention). It
   visibly favors models when clips start negative (random model: 0.56 on the synthetic
   full-scale set); T06 must match STG-NF's protocol before comparing to 67.5.
8. Streaming contract (factory → `process`/`flush`) proposed in `eval/README.md`.

## Blockers (paused parts; everything else continues)

1. **`meva_fa` ≥ 100 camera-hours (correctness decision).** The held-out cameras have 33.3 indexed
   hours (17.0 h downloaded). 100 h is only reachable if (a) evaluated models declare lineage without
   MEVA (then all indoor footage counts), (b) T14 adds more indoor cameras that never enter training,
   or (c) the owner accepts FA measured on train-camera footage for MEVA-trained models (optimistic,
   so I have not done it). The suite reports hours and "NOT MET" meanwhile. Needs: Abdul/T06 decision.
2. **Streaming pipeline interface (interface decision, T13/T12).** No pipeline exists at
   `wave0-baseline`. The e2e suites take `--pipeline module:factory` with the contract in
   `eval/README.md`. T13 should confirm or amend it; until then e2e numbers come only from
   reference pipelines.
3. **PoseLift official split (data).** Only the 44 test masks are downloaded; `eval/splits/poselift.json`
   is generated from the converted data once the pose files land (`make_splits poselift` refuses to
   guess). The pickle layout parser is written against the README; `iter_stgnf_json` matches the
   STG-NF layout the masks confirm. Verify with `pytest -m data` when T14's retry succeeds.
4. **RetailS license (`pending`)**: converter present, never run.

## Handoffs

- **T00 (pyproject/CI):** add an `eval` extra (`pyarrow`, Apache-2.0) and install it in CI;
  otherwise the converter/loader/e2e tests (`importorskip("pyarrow")`) are skipped in CI. Consider
  `ruff check eval` and `mypy eval` in CI (both clean now).
- **docs/DATA.md owner:** add `item_put_down`, `item_transfer`, `shoplifting` to the label vocabulary
  and the optional `ClipLabels` fields + `frame_labels/` (listed in docs/EVAL.md).
- **T14:** (1) `index/clips.jsonl` shows 0 events for 37 annotated clips (e.g.
  `2018-03-07.16-55-00.17-00-00.school.G421`, 52 events): KPF files sit under the *end* hour's
  directory; index by filename instead. (2) Not all MEVA frames are 1920×1072: `bus.G331` is 1920×1080
  (the converter probes each clip). (3) For `meva_fa`, more held-out indoor cameras would help (Blocker 1).
  (4) PoseLift pose files.
- **T06:** loader API in `eval/README.md`; `public_pose.py` can subclass/replace
  `eval.pose_bench.PublicPose`; give models `score_windows()` for speed and `lineage` for `meva_fa` /
  `sim_transfer`; SSL pretraining on MEVA must use split `train` only.
- **T07:** emulated variants must keep the source clip's `group_id`; `Splits.split_for(clip, group_id)`.
- **T08:** sim converter waits for your output format (expected fields in `eval/converters/own.py:convert_sim`).
- **T12:** `soak*` suite: `run_e2e(..., realtime=True)` over 10 streams + `crash_summary` + RSS sampling.
- **T13:** confirm the streaming factory contract; `eval.e2e.ProtocolPipeline` is a reference build.
- **Humans (H1/H1a):** take-log CSV columns for `quick_capture` / `lab_mock_aisle` are in
  `eval/converters/own.py`; please log `visible` per camera.

## Licenses

Code deps: pyarrow (Apache-2.0), optional test-only scipy (BSD-3), scikit-learn (BSD-3),
motmetrics (MIT). Data: MEVA CC-BY-4.0 (attribution in every converted INFO.json via T14's manifest),
SmartSpaces CC-BY-4.0, PoseLift R&D (data coverage of Apache-2.0 unconfirmed), RetailS pending (unused).
Nothing downloaded by T09; no data, weights or video committed.
