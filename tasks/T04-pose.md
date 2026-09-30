# T04 — Pose on high-resolution crops

**Wave 1 · owned:** `src/scs/perception/pose*.py`

## Goal
Accurate COCO17 keypoints for gated tracks, especially **wrists and hands near shelves and the torso**, from ceiling-mounted views.

## Deliverables
1. `PoseEstimator` top-down implementations: RTMPose (Apache-2.0) and YOLO11-pose (AGPL, R&D). Crops come from the **main-stream** frame using track boxes **as-is**: they're already main-stream full-res pixels (ADR 0002), so no rescaling. Keypoints are emitted in the same main-stream pixels.
2. Batched TRT inference with a fixed max batch; crops padded/letterboxed to 256×192 (or model native).
3. Temporal smoothing (One-Euro filter) per track, with confidence-aware gap filling up to 8 frames. Keep raw and smoothed.
4. Zone gating helper: pose only for tracks whose box intersects SHELF/HIGH_VALUE/CHECKOUT/ENTRY_EXIT polygons (expanded by 10%). **Use `scs.geometry`** (`bbox_touches_zone(..., expand=0.10)`); don't write polygon math here.
5. Report: PCK@0.2 for wrists on (a) COCO-keypoints val (the person-crop protocol), (b) sim ground truth from T08 by camera profile and mount height, and (c) a human-annotated wrist set from `quick_capture` *(when data exists)*. Latency at 30 / 100 / 300 crops per second. MEVA isn't used here: it has no keypoint labels.

## Acceptance
- [ ] Outputs validate as `Pose` (17 kps) and match PoseLift keypoint conventions (same COCO order), so T06 models transfer.
- [ ] 300 crops/s sustained with < 15 ms per batch on the 5090.
- [ ] Wrist PCK on 1440p crops beats the same model on 360p sub-stream crops (quantifies ARCHITECTURE D1). Include the numbers. Substitute now: COCO val and sim frames downscaled to the sub-stream resolution. Final: `quick_capture` simultaneous main+sub *(when data exists)*.

## Out of scope
Hand keypoints (21-pt). Note in the report whether they're worth it.
