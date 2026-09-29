# T09 — Eval harness & dataset converters

**Wave 1 · owned:** `eval/` framework (runner, metrics, dataset loaders/converters, splits, suite registry), `docs/EVAL.md`. Not owned: `eval/suites/public_pose.py` (T06) and `eval/suites/emu_matrix.py` (T07). Provide the suite base class/registry they plug into.

## Goal
One command produces the metrics in `docs/EVAL.md` for any suite, so every other agent's work is judged the same way.

## Deliverables
1. Converters to the canonical format: PoseLift, RetailS, MEVA (subset: indoor cams + object-interaction activities; boxes/activities only, no keypoints), MERL Shopping (if license OK), `quick_capture` and `lab_mock_aisle` (from the take log), sim output.
2. Split generator by actor/clip/camera; frozen `splits.json` checked into `eval/splits/` (IDs only, no data).
3. Metrics: event recall at FA budget + full curve, frame AUC-ROC/PR, per-subtype recall, IDF1/ID switches (via `motmetrics` or TrackEval), journey F1, latency percentiles.
4. Runner: `python -m eval.run --suite <s> [--models ...] [--policy ...]` → `runs/eval/<sha>/<suite>.json` + `summary.md`; `--compare main` produces a delta table for PRs.
5. `eval/README.md` explaining how to add a suite.

## Acceptance
- [ ] Unit tests for each metric against hand-computed toy cases (including overlapping and duplicate alerts).
- [ ] `public_pose` runs on PoseLift in < 5 min on the 5090 with a dummy model.
- [ ] Reports include git sha, model ids, dataset versions, driver/GPU.

## Addendum (review round 1)
- **Derivative groups:** every clip gets a `group_id`; crops, overlapping windows, emulated variants, and augmentations inherit it and stay in the same split. A test fails if any group spans splits.
- **Streaming path:** `lab_e2e` and `soak` run the real pipeline from video files (causal smoothing, track resets, dedupe, suppression), not saved scores.
- **False alerts per hour** only from continuous footage suites.
- **Uncertainty:** bootstrap 95% CIs over clips/actors and sample counts on every headline metric.
- **Label honesty:** metrics needing labels a dataset lacks report `unavailable`. Positives = `visible: observed` for that camera.
- **Thresholds:** fit on val, frozen for test; site calibration reported separately.
- **Licensing:** RetailS is `pending`. Write the converter against its documented format but don't download it until docs/DATA.md says approved.
- Implement the `sim_transfer` suite (train-set variants, one fixed real test set).
