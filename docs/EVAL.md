# Evaluation spec

The eval harness (T09) is the product's source of truth. Anything that changes
detection behavior must report these numbers. Run: `python -m eval.run --suite <name>`
(`--list` shows suites; GPU pipelines go through `scripts/gpu shared -- ...`).
Output: `runs/eval/<git-sha>/<suite>.json` + `summary.md`; `--compare main` adds a delta
table. How to add a suite, the loader API and the streaming contract: `eval/README.md`.

## Suites

| Suite | Data | Purpose |
|---|---|---|
| `public_pose` | PoseLift official test (RetailS only once licensed) | Pose-sequence model comparability with published baselines. Not a system-level validation. **Theft recall on real footage (pose only)** comes from here. Framework default in `eval/pose_bench.py`; T06's `eval/suites/public_pose.py` replaces it. |
| `sim_matrix` | `sim_store_v*` × all camera profiles × mount heights | Camera selection and robustness |
| `emu_matrix` | `lab_mock_aisle` re-rendered through T07 per profile | Real-footage robustness to camera/codec/lighting |
| `lab_e2e` | `lab_mock_aisle` held-out actors, full pipeline from video file | End-to-end system metric (the one that gates releases) |
| `soak` | 24 h replay of normal-only footage at real time, 10 streams | Stability, false alerts per camera-hour, memory growth |
| `sim_transfer` | Train on {real/mock only, sim only, real+sim}; test on the same untouched real test set | Decides whether sim data earns more investment. Variants = `--models real=… sim=… real_sim=…`; paired bootstrap deltas vs `real`; fails if a variant's lineage includes the test split. |
| `perf` | Synthetic 10-stream load | Throughput, latency, GPU/NVDEC utilization |
| `meva_interaction` | MEVA indoor, held-out site `bus` + camera `school.G421` (val: `school.G423`): pick-up / put-down / transfer events | **Free real-footage proxy** for *interaction* recall at an FA budget, through the full streaming pipeline. **Not theft recall**: MEVA has only 5 scripted steals (counted as pickups). |
| `meva_fa` | ≥ 100 camera-hours of continuous MEVA indoor video that no evaluated model trained on | False **alerts** per camera-hour on real continuous footage (proxy until store shadow). Every alert is false except on the 5 steals (ignored). |
| `smartspaces_track` | SmartSpaces retail scenes 071-080, held-out scene 073 (val 072); falls back to labeled non-retail scenes **flagged** if no retail GT | Detection AP@0.5, IDF1, ID switches on overhead views. Synthetic. |

Every report header carries the suite's labels: "proxy, not retail" (MEVA), "synthetic"
(SmartSpaces, sim, T14 generated clips), "pose only" (PoseLift). Synthetic suites never
count as validation on their own.

## Metrics

**Primary (release gate):**
- **Event recall @ false-alert budget:** fraction of labeled theft events that produce
  an `Alert` whose window overlaps the event, while false alerts ≤ **1 per camera per
  10 open-hours**. That's about 10/day for a 10-cam store before review. Report the
  full recall-vs-FA curve too.
- **Precision after verifier** (T11 on): fraction of alerts reaching review that are true.

**Secondary:**
- Frame-level AUC-ROC / AUC-PR on `public_pose` (compare to STG-NF 67.5 PoseLift, 63.2 RetailS-real).
- Per-subtype recall (pocket, bag, waistband, jacket, grab_run).
- Tracking: IDF1, ID switches per person-minute (the legacy demo shows ~2 per 10 s).
- Journey accuracy: checkout-visit and store-exit event F1 (these feed the strongest rule).
- Robustness: worst-profile recall on `emu_matrix` / `sim_matrix` ÷ best-profile recall.
- Perf: p95 event→alert latency, sustained fps per stream, VRAM, NVDEC %.
- Soak: crashes (must be 0), RSS growth < 5%/24 h, reconnect success after forced RTSP drop.

## Gates

| Gate | When | Must show |
|---|---|---|
| **G1 — free data** | End of Wave 2 | One frozen pipeline config (code sha + policy hash), all reported "proxy, not retail" / "synthetic" as applicable: (a) `meva_fa` false alerts per camera-hour + CI on ≥ 100 held-out camera-hours; (b) `meva_interaction` pick-up/put-down/transfer recall + CI at the interaction threshold fit on val at 0.1 false detections per camera-hour (this is **interaction detection, not theft**); (c) **theft recall** from `public_pose` (real, pose only: behavior model ≥ STG-NF frame AUC) plus synthetic theft suites (sim v0, T14 generated clips), each flagged synthetic; (d) `smartspaces_track` IDF1 reported; (e) `sim_transfer` result for sim v0; (f) 10 MEVA replay streams for 24 h with 0 crashes. |
| **G2 — lab** | After `lab_mock_aisle` recorded | `lab_e2e` recall ≥ 0.6 at the FA budget on held-out actors; soak 24 h clean; camera recommendation written from `emu_matrix` + `sim_matrix`. |
| **G3 — store shadow** | In store, 2+ weeks | Alerts reviewed but NOT sent. Reviewers also check a random sample of **non-alerted** footage to estimate misses. Measured FA/day, recall on known incidents, review minutes/day. Acceptable review burden agreed with the operator **before** enabling live notifications. |

Numbers in G2 are starting targets. Adjust by ADR with evidence, not by vibe.

## Metric definitions (exact; hand-worked toy tests in `tests/eval/`)

- **Event matching**: per camera clip; overlap if `alert.t_start <= event.t_end + tol` and
  `alert.t_end >= event.t_start - tol` (tol default 2 s); one-to-one, greedy by score, each
  alert takes the overlapping unmatched positive with the highest temporal IoU. Outcomes:
  true / **duplicate** (overlaps only already-matched positives) / ignored (only ignore
  regions: partially or not observed, flagged annotations) / false. Duplicates count
  toward the FA budget by default (review burden) and are always reported separately.
- **Alert window** from the pipeline: earliest referenced `Event` → `Alert.ts_open`.
- **FA/h** = (false + duplicate) alerts / camera-hours of continuous footage. Clip-curated
  suites report "unavailable" (enforced by `SuiteResult`).
- **Threshold at budget** = lowest threshold whose FA/h ≤ budget on **val**, frozen for
  test (−∞ if every threshold meets it). Oracle test-fit numbers are labeled "oracle".
- **Frame AUC-ROC / AUC-PR** = sklearn `roc_auc_score` / `average_precision_score` over
  all frames of all clips; window score assigned to its last frame (causal); frame = max
  over tracks; frames without a scored person = clip minimum. Gaussian smoothing only via
  `--opt sigma=` and reported as NON-CAUSAL.
- **IDF1 / ID switches / MOTA** = motmetrics conventions at IoU ≥ 0.5 (cross-checked);
  switches per person-minute = switches / (GT boxes / fps / 60). **Det AP@0.5** = all-point
  interpolated.
- **Journey F1** per type (checkout_visit, store_exit): one-to-one within ±3 s, same person
  when both sides have ids.
- **Latency** = alert emit (media time of the frame being processed) − event end;
  numpy linear percentiles; CI on p95.
- **CIs**: percentile bootstrap (B = 1000, fixed seed) over clips (tracking: sequences);
  `LOW-N` when < 30 positives. Sample counts on every headline number.

## Splits (frozen IDs in `eval/splits/`, policy in `eval/make_splits.py`)

- Units = union-find over `group_id` (derivatives) + actor ids + `view_group`
  (simultaneous overlapping views). A unit never spans splits; clips linked to a held-out
  clip only by view/actor are **excluded**, never trained on. A test fails CI if any
  group spans splits. Unlisted derivatives resolve through their `group_id`.
- `meva`: test = site `bus` + camera `school.G421`; val = `school.G423`; train = the rest.
  Camera sets (shared fields of view) don't cross splits, so nothing is excluded. MEVA
  actor ids are per clip, so actor-disjoint splits are impossible (stated in reports).
  Self-supervised pretraining must use `train` clips only.
- `poselift`: official test split (STG-NF layout); val = 10% of official train.
- `smartspaces`: by scene (test 073, val 072). Scenes share one space and character set.

## Canonical-format additions (T09; all optional, see `eval/canonical.py`)

`ClipLabels`: `dataset, camera_id, site_id, group_id, view_group, actor_ids, width, height,
n_frames, duration_s, continuous, synthetic, video, supports{events, subtypes, actors,
journeys, gt_tracks, keypoints, frame_labels, exhaustive}, extra`. `LabelEvent`:
`source_label, event_id, extra`. Label vocabulary adds `item_put_down`, `item_transfer`
(MEVA; not forced into `item_returned`) and `shoplifting` (dataset-level label, no
subtype). Optional `frame_labels/<clip>.npy`.

## Rules

- Splits are by **actor and clip**, never by frame. Held-out actors never appear in training.
  All derivatives of a clip (crops, overlapping windows, emulated variants, augmentations)
  share a `group_id` and stay in one split.
- No threshold tuning on the test split. Calibrate on val and freeze. Site-specific
  calibration is reported separately from generalization results.
- **False alerts per hour are measured on continuous footage only**, never on curated clip sets.
- **End-to-end suites run the deployed streaming path** (sampling, causal smoothing, track
  resets, dedupe, alert suppression) from video files, not saved model scores.
- Report sample counts and 95% confidence intervals (bootstrap over clips/actors) for
  every headline metric. With few events, say so.
- Only report metrics the labels support; otherwise "unavailable".
- Positives for detection metrics are events labeled `visible: observed` for that camera.
- Every report records: git sha, model ids, dataset versions, camera profiles, GPU, driver.
- Sim/emulator results never count as production validation on their own. Only the real
  held-out set and store shadow mode do.
