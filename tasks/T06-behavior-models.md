# T06 — Behavior models on pose sequences

**Wave 2 · owned:** `src/scs/behavior/`, `eval/suites/public_pose.py` (built on T09's framework)

## Goal
A `BehaviorModel` that outputs a calibrated suspiciousness score per track window, trained with public + synthetic data now and staged lab data when it arrives.

## Deliverables
1. Baselines reproduced on PoseLift/RetailS via T09 loaders: STG-NF (unsupervised), plus a supervised skeleton classifier (ST-GCN or PoseC3D-lite) trained on PoseLift/RetailS thefts + `sim_store` labeled conceal actions.
2. Input normalization that makes models camera-agnostic: center on hips, scale by torso length, include keypoint confidences, optional camera-tilt feature from `CameraInstall`.
3. Training recipes as configs; runs queued via `tsp`; artifacts to `models/<model_id>/` with `MANIFEST.yaml` entry (data versions, license lineage).
4. Calibration (temperature/isotonic on val) so `score` is comparable across models.
5. Ablation: public-only vs public+sim vs public+sim+emu; report the effect of sim data on real test sets (this answers "is Isaac Sim worth it?").

## Acceptance
- [ ] Reproduced STG-NF within ±3 AUC points of published PoseLift (67.5) and RetailS-real (63.2).
- [ ] Our best model ≥ STG-NF on both, reported with the val-frozen threshold.
- [ ] Inference < 2 ms per 24-frame window on GPU; model implements `BehaviorModel`.
- [ ] Every model's license lineage recorded (RetailS-trained weights stay R&D until H2 resolves).

## Out of scope
Pixel/video models (optional follow-up for alert candidates only).
