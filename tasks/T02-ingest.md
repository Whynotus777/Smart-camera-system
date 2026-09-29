# T02 — Ingest

**Wave 1 · owned:** `src/scs/ingest/`, `scripts/rtsp_probe.py`

## Goal
Robust, timestamped frames from RTSP cameras, video files, sim output, and emulated streams through one `FrameSource` interface.

## Deliverables
1. `RtspSource`: GStreamer (preferred, NVDEC via `nvh264dec`/`nvh265dec` or `nvv4l2decoder`) with a PyAV fallback. Main stream by default (ARCHITECTURE D1). Handles H.264 and H.265.
2. Reconnect with exponential backoff (cap 30 s); emits `Event(type=CAMERA_HEALTH)` on disconnect/reconnect/stall (> 3 s no frames) to `Streams.HEALTH`.
3. Frame identity and time (ADR 0002, ARCHITECTURE D10): every frame carries `(camera_id, epoch, seq)`, `ts` = decode wall clock, `ts_mono` = host monotonic clock at decode, `source_ts` = RTP/camera capture time when available, and the `transform` applied. `FrameRef.width/height` are always the main-stream dimensions; set `stream` to the stream actually decoded. Latency reports must say which clock they measure (capture→decode vs decode→result).
4. `FileSource` (loops optionally, `realtime=True` paces to fps) and `DirectorySource` for sim frames.
5. Multi-camera manager: one decode worker per camera, bounded analytics queue (drop-oldest), per-camera stats: decoded frames, dropped frames, fresh-frame age, errors, fps.
5b. **Encoded packet tap** for evidence (ARCHITECTURE §2): a callback/queue of encoded packets with the same frame identity, taken *before* any analytics dropping. T10 builds the ring buffer on it.
6. Frames available as GPU tensors (torch, via DLPack or CUDA memory) when GPU decode is used; CPU numpy otherwise.
7. `scripts/rtsp_probe.py`: prints stream specs for a URL from an env var (for filling camera profiles).

## Acceptance
- [ ] 10 simultaneous `FileSource`s from 1440p H.264 **and** H.265 files sustain 15 fps each on the 5090, decoding on NVDEC. Evidence = per-camera decoded/dropped counts, fresh-frame age p95, and errors over 10 min; `nvidia-smi dmon` is supporting only. Run under `scripts/gpu exclusive`.
- [ ] Packet tap receives 100% of packets even when the analytics queue is dropping (test by stalling the consumer).
- [ ] Integration test with a local RTSP server (`mediamtx` in Docker) serving a file: kill the server for 10 s → source recovers, HEALTH events emitted, no thread leak.
- [ ] Unit tests for backoff, queue drop policy, timestamp monotonicity.
- [ ] Works headless (no `cv2.imshow` anywhere).

## Out of scope
Recording and clips (T10). Detection.

## Gotchas
Reolink RTSP can stall silently; detect frame staleness rather than relying on read errors. Some Reolinks default the main stream to H.265.
