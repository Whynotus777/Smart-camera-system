# T03 — Detection + tracking

**Wave 1 · owned:** `src/scs/perception/detect*.py`, `src/scs/perception/track*.py`

## Goal
Batched person detection across cameras and per-camera tracking with low ID switches, behind swappable interfaces.

## Deliverables
0. Coordinates (ADR 0002): detectors run on a resized input (~640 px) but **map boxes back to main-stream full-res pixels** before returning `Detection`s. Keep the letterbox/scale parameters and unit-test the round trip. Nothing leaves T03 in detector-input pixels.
1. `Detector` implementations: (a) Ultralytics YOLO11 (R&D, AGPL) and (b) one Apache-2.0 option (RTMDet or RT-DETR/D-FINE). Both export to TensorRT FP16 with dynamic batch. Inference uses **micro-batching with a deadline** (ARCHITECTURE D9): batch whatever frames are ready within ~20 ms, up to a max; a slow or dead camera never delays others. Benchmark batch sizes 1/4/8/16 and pick from latency + throughput.
2. `Tracker` implementations: ByteTrack (MIT reference impl) and one appearance-aware tracker (e.g. BoT-SORT-style with a lightweight ReID) for occlusions at shelves.
3. Optional `CrossCameraAssociator` (in-memory, TTL 30 min, no persistence). Leave it disabled by default; document when it helps.
4. Fine-tuning script for detector on MEVA indoor + sim + (later) lab data; overhead/steep-angle views are the domain gap.
5. Comparison report: mAP-person on a held-out CCTV set (MEVA), IDF1 and ID switches on sim clips now and on `quick_capture` / `lab_e2e` *(when data exists)*, latency per batch of 10, and **license** per option.

## Acceptance
- [ ] Batch of 10 × 640 px in < 10 ms on the 5090 (TRT FP16) for at least one detector, measured under `scripts/gpu exclusive`.
- [ ] Test: one camera stops sending frames → other cameras' detection latency unchanged.
- [ ] Report ends with **one recommended production candidate**; the other stays as a comparison only.
- [ ] ID switches per person-minute at least 50% lower than the **baseline: default ByteTrack settings** on the same clips. Clips now: `tests/fixtures/video/demo_*.mp4` (smoke-level only: the PoC's boxes are burned into the pixels) plus MEVA indoor and SmartSpaces retail clips; then `quick_capture` *(when data exists)*.
- [ ] Both detectors pass the same interface test suite.
- [ ] Report committed to `docs/reports/T03-detect-track.md`.

## Out of scope
Hand/item detection (possible follow-up; note ideas in the report).

## Addendum (free data)
Fine-tune/evaluate on `smartspaces` retail scenes (overhead retail, synthetic, CC-BY) and `meva` indoor (real CCTV). Report both separately; the gap between them is itself a finding. Suite: `smartspaces_track`.
