# T02 — Ingest

**Wave 1 · owned:** `src/scs/ingest/`

## Goal
Robust, timestamped frames from RTSP cameras, video files, sim output, and emulated streams through one `FrameSource` interface.

## Deliverables
1. `RtspSource`: GStreamer (preferred, NVDEC via `nvh264dec`/`nvh265dec` or `nvv4l2decoder`) with a PyAV fallback. Main stream by default (ARCHITECTURE D1). Handles H.264 and H.265.
2. Reconnect with exponential backoff (cap 30 s); emits `Event(type=CAMERA_HEALTH)` on disconnect/reconnect/stall (> 3 s no frames) to `Streams.HEALTH`.
3. Timestamps: wall-clock at decode and monotonic, stored in `FrameRef.ts`; record RTSP/RTP timestamps in `data` when available for drift checks.
4. `FileSource` (loops optionally, `realtime=True` paces to fps) and `DirectorySource` for sim frames.
5. Multi-camera manager: one decode worker per camera, bounded queue (drop-oldest), per-camera fps/latency stats.
6. Frames available as GPU tensors (torch, via DLPack or CUDA memory) when GPU decode is used; CPU numpy otherwise.
7. `scripts/rtsp_probe.py`: prints stream specs for a URL from an env var (for filling camera profiles).

## Acceptance
- [ ] 10 simultaneous `FileSource`s from 1440p H.264 files sustain 15 fps each on the 5090, decode on NVDEC (show `nvidia-smi dmon` output).
- [ ] Integration test with a local RTSP server (`mediamtx` in Docker) serving a file: kill the server for 10 s → source recovers, HEALTH events emitted, no thread leak.
- [ ] Unit tests for backoff, queue drop policy, timestamp monotonicity.
- [ ] Works headless (no `cv2.imshow` anywhere).

## Out of scope
Recording and clips (T10). Detection.

## Gotchas
Reolink RTSP can stall silently; detect frame staleness rather than relying on read errors. Some Reolinks default the main stream to H.265.
