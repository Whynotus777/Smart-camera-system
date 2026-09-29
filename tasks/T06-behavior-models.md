# T06 — Behavior models on pose sequences

**Wave 2 · owned:** `src/scs/behavior/`, `eval/suites/public_pose.py` (built on T09's framework)

## Goal
A `BehaviorModel` that outputs calibrated scores for **observable interactions** (item to bag, item to clothing, item returned, obscured interaction) per track window. Pose is a baseline; the model is allowed to use pixels (ARCHITECTURE D3).

## Deliverables
1. Three candidates, same splits and same budget:
   - **Pose**: STG-NF reproduced on PoseLift (RetailS only once licensed) + a supervised skeleton classifier.
   - **Visual**: a small video model on person/hand-region crops over time (from T04's crop export).
   - **Fused**: visual + pose + track + zone context.
   The question is whether pixels cut nuisance alerts at equal recall. Key hard negatives: own phone out of pocket, item to basket, item returned.
2. Input normalization that makes models camera-agnostic: center on hips, scale by torso length, include keypoint confidences, optional camera-tilt feature from `CameraInstall`.
3. Training recipes as configs; runs queued via `tsp`; artifacts to `models/<model_id>/` with `MANIFEST.yaml` entry (data versions, license lineage).
4. Calibration (temperature/isotonic on val) so `score` is comparable across models.
5. `sim_transfer` experiment (docs/EVAL.md): {real/mock only, sim only, real+sim} evaluated on the **same untouched real test set**. Sim gets more investment only if it improves this number.

## Acceptance
- [ ] Reproduced STG-NF within ±3 AUC points of published PoseLift (67.5).
- [ ] Pose vs visual vs fused compared on `lab_e2e` (once recorded) at the same false-alert budget, with CIs; recommendation in `docs/reports/T06-*.md`.
- [ ] Inference < 2 ms per 24-frame window on GPU; model implements `BehaviorModel`.
- [ ] Every model's license lineage recorded (RetailS-trained weights stay R&D until H2 resolves).

## Out of scope
Large foundation-model pretraining. Small task-specific models on top of pretrained backbones only.
