# T07 — Camera emulator & profiles

**Wave 1 · owned:** `src/scs/camera_emu/`, `configs/camera_profiles/`, `eval/suites/emu_matrix.py` (built on T09's framework)

## Goal
Answer "which camera, which stream, where mounted?" before we're in the store, by re-rendering high-quality footage as specific cameras would see it.

## Deliverables
1. `emulate(src_video, src_profile, dst_profile, opts) -> path` that applies, in order: FOV crop/scale, lens distortion (Brown-Conrady, or equidistant fisheye), sensor noise vs. lux, exposure/motion blur by shutter, IR night mode (grayscale + IR response + bloom), resolution, fps decimation, codec re-encode at the profile bitrate (ffmpeg, H.264/H.265, CBR/VBR), optional packet-loss artifacts.
2. Profiles: extend `configs/camera_profiles/` (Reolink models we might deploy, a 1080p enterprise dome, a fisheye) using the `verification` levels from contracts v0.2: `approximation` → `spec_sourced` (datasheet link in `sources`) → `measured` (rtsp_probe / test footage from the physical camera) → `emulator_calibrated` (emulator output matched against that camera's real footage). A datasheet never makes noise/IR/low-light behavior calibrated.
3. `EmulatedSource` implementing `FrameSource` so the whole pipeline can run on emulated streams.
4. `emu_matrix` suite driver (`eval/suites/emu_matrix.py`): for each source clip × profile × {day, dusk, IR} → run pipeline → metrics via T09.

## Acceptance
- [ ] Emulated sub-stream output from a 1440p Reolink main-stream recording is statistically close to the real sub-stream recorded at the same time (SSIM/LPIPS + detector-score distribution, documented). Data: `quick_capture` simultaneous main+sub *(when data exists)*. Substitute now: per-transform unit checks against synthetic targets (e.g. known blur kernel, known bitrate).
- [ ] Deterministic given a seed. Unit tests per transform.
- [ ] Report `docs/reports/T07-camera-matrix.md`: **sensitivity** analysis (how recall/FA move with resolution, bitrate, codec, lighting), using MEVA/SmartSpaces clips now and `quick_capture` *(when data exists)*. No purchase or mounting recommendation until predictions are checked against real footage from that camera; mark any draft recommendation as provisional.

## Limits (state these in the report)
2D emulation can't change viewpoint or mount height. That's T08's job.
