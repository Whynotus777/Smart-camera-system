# T03 — Detection + tracking

**Wave 1 · owned:** `src/scs/perception/detect*.py`, `src/scs/perception/track*.py`

## Goal
Batched person detection across cameras and per-camera tracking with low ID switches, behind swappable interfaces.

## Deliverables
1. `Detector` implementations: (a) Ultralytics YOLO11 (R&D, AGPL) and (b) one Apache-2.0 option (RTMDet or RT-DETR/D-FINE). Both export to TensorRT FP16 with batch = number of cameras.
2. `Tracker` implementations: ByteTrack (MIT reference impl) and one appearance-aware tracker (e.g. BoT-SORT-style with a lightweight ReID) for occlusions at shelves.
3. Optional `CrossCameraAssociator` (in-memory, TTL 30 min, no persistence). Leave it disabled by default; document when it helps.
4. Fine-tuning script for detector on MEVA indoor + sim + (later) lab data; overhead/steep-angle views are the domain gap.
5. Comparison report: mAP-person on a held-out CCTV set, IDF1 and ID switches on `lab_e2e`/sim clips, latency per batch of 10, and **license** per option.

## Acceptance
- [ ] Batch of 10 × 640 px in < 10 ms on the 5090 (TRT FP16) for at least one detector.
- [ ] ID switches per person-minute at least 50% lower than legacy ByteTrack config on the same clips (legacy demo ≈ 2 switches per 10 s).
- [ ] Both detectors pass the same interface test suite.
- [ ] Report committed to `docs/reports/T03-detect-track.md`.

## Out of scope
Hand/item detection (possible follow-up; note ideas in the report).
