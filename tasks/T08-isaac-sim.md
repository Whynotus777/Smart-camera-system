# T08 — Isaac Sim synthetic convenience store

**Wave 1 (long-running) · owned:** `sim/` · **depends on:** T00, T02 (`DirectorySource`), T07 (profiles)

## Goal
Generate labeled multi-camera video + ground truth (boxes, 2D/3D skeletons, journeys, theft subtype) across camera profiles and mount heights, to (a) pretrain/augment T06 and (b) compare camera placements.

## Deliverables
1. Environment: Isaac Sim (current release) in the official container on the 5090, driver pinned to the validated version. `sim/README.md` documents exact versions. Blackwell + Isaac Sim is driver-sensitive; test the pin before anything else.
2. **First deliverable is narrow (v0):** one shelf, one camera, one actor, matched pairs of interactions (pick → return vs pick → bag; phone out of own pocket vs item → pocket), varied over camera angle, lighting, occlusion, clothing, object size. Same actors/clothes/backgrounds on both sides of each pair. Hand v0 to T06 for `sim_transfer` before scaling up.
2b. Scene (after v0 proves useful) `sim/scenes/cstore_v1.usd`: ~8×12 m c-store (2–3 aisles, candy rack by counter, coolers, counter, door), SimReady/purchased assets with licenses recorded in `docs/DATA.md`.
3. Actors via Isaac Sim Replicator Agent (IRA): routines for browse, pick-inspect-return, pick-to-checkout, and staff restock. **Theft behaviors** (pocket, bag, waistband, jacket, grab-and-run) need custom animations. Try in order: existing animation libraries with clear licenses; mocap from our own staged video (video→3D human motion model, then retarget), noting that SMPL-based pipelines are non-commercial unless licensed.
4. Camera rigs generated from `configs/camera_profiles/*` (intrinsics, distortion, resolution) at mount heights {2.4, 2.7, 3.0 m} and tilts {30, 45, 60°}, plus randomized lighting and clothing.
5. Writer → canonical format (`docs/DATA.md`). Scripted truth labels (`label_source: script`) **plus per-camera `visible`** computed from the renderer (hands/item occluded or out of frame at the decisive moment → `not_observed` / `partially_observed`). Event labels start at the decisive moment, never at the start of a sequence that later ends in theft.
6. Batch generation CLI with seeds. Scale to the v1 corpus (≥ 2,000 clips) **only after** `sim_transfer` shows v0 improves the real test set.

## Acceptance
- [ ] Ground-truth skeletons project correctly onto frames (overlay check on 50 random frames).
- [ ] `sim_matrix` suite runs end-to-end through the pipeline from `DirectorySource`.
- [ ] Generation throughput and GPU hours documented; runs scheduled so they don't collide with T12 benchmarks.

## Out of scope
Robot dispatch simulation (legacy `simulation_trigger.py` idea) — parked.
