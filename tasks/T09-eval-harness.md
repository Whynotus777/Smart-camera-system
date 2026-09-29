# T09 — Eval harness & dataset converters

**Wave 1 · owned:** `eval/`, `docs/EVAL.md`

## Goal
One command produces the metrics in `docs/EVAL.md` for any suite, so every other agent's work is judged the same way.

## Deliverables
1. Converters to the canonical format: PoseLift, RetailS, MEVA (subset: indoor cams + object-interaction activities), MERL Shopping (if license OK), `lab_mock_aisle` (from the tablet take log), sim output.
2. Split generator by actor/clip/camera; frozen `splits.json` checked into `eval/splits/` (IDs only, no data).
3. Metrics: event recall at FA budget + full curve, frame AUC-ROC/PR, per-subtype recall, IDF1/ID switches (via `motmetrics` or TrackEval), journey F1, latency percentiles.
4. Runner: `python -m eval.run --suite <s> [--models ...] [--policy ...]` → `runs/eval/<sha>/<suite>.json` + `summary.md`; `--compare main` produces a delta table for PRs.
5. `eval/README.md` explaining how to add a suite.

## Acceptance
- [ ] Unit tests for each metric against hand-computed toy cases (including overlapping and duplicate alerts).
- [ ] `public_pose` runs on PoseLift in < 5 min on the 5090 with a dummy model.
- [ ] Reports include git sha, model ids, dataset versions, driver/GPU.
