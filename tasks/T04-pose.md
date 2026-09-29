# T04 — Pose on high-resolution crops

**Wave 1 · owned:** `src/scs/perception/pose*.py`

## Goal
Accurate COCO17 keypoints for gated tracks, especially **wrists and hands near shelves and the torso**, from ceiling-mounted views.

## Deliverables
1. `PoseEstimator` top-down implementations: RTMPose (Apache-2.0) and YOLO11-pose (AGPL, R&D). Crops come from the **main stream** frame using track boxes scaled from detector resolution.
2. Batched TRT inference with a fixed max batch; crops padded/letterboxed to 256×192 (or model native).
3. Temporal smoothing (One-Euro filter) per track, with confidence-aware gap filling up to 8 frames. Keep raw and smoothed.
4. Zone gating helper: pose only for tracks whose box intersects SHELF/HIGH_VALUE/CHECKOUT/ENTRY_EXIT polygons (expanded by 10%).
5. Report: PCK@0.2 for wrists on MEVA/sim/lab annotated frames by camera profile and mount height; latency at 30 / 100 / 300 crops per second.

## Acceptance
- [ ] Outputs validate as `Pose` (17 kps) and match PoseLift keypoint conventions (same COCO order), so T06 models transfer.
- [ ] 300 crops/s sustained with < 15 ms per batch on the 5090.
- [ ] Wrist PCK on 1440p crops beats the same model on 360p sub-stream crops (quantifies ARCHITECTURE D1). Include the numbers.

## Out of scope
Hand keypoints (21-pt). Note in the report whether they're worth it.
