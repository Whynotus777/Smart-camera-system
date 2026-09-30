# T02 — Ingest: report

Status: implementation complete, and every acceptance criterion is met (evidence below).
Branch stack: `agent/T02-ingest` → `agent/T02-source` → `agent/T02-backends` →
`agent/T02-rtsp` → `agent/T02-bench`. Code: `src/scs/ingest/`,
`scripts/rtsp_probe.py`, tests in `tests/ingest/`.

## Acceptance

| Criterion | Status | Evidence |
|---|---|---|
| 10 × 1440p15 H.264 **and** H.265 files sustain 15 fps each on NVDEC, 10 min, `scripts/gpu exclusive` | ✅ | [Decode benchmark](#decode-benchmark-10--1440p15-10-min-gpu-exclusive) |
| Packet tap receives 100 % of packets while the analytics queue drops (stalled consumer) | ✅ | `test_tap_complete_and_decodable_while_consumer_stalls` ×4 (GStreamer/PyAV × H.264/H.265, NVDEC) + `test_packet_tap_gets_every_packet_while_analytics_drops` (CPU) |
| mediamtx server killed 10 s → source recovers, HEALTH events, no thread leak | ✅ | `test_server_killed_10s_recovers_with_health_events_and_no_thread_leak` ×2 backends; [MEVA fault run](#meva-fault-run-reconnect--stall) |
| Unit tests: backoff, queue drop policy, timestamp monotonicity | ✅ | `tests/ingest/test_ingest_core.py` |
| Headless (no `cv2.imshow`) | ✅ | no GUI calls in `src/scs/ingest/` or `scripts/rtsp_probe.py` |

Test runs (5090, commit `aa43999`; `.env`: GStreamer from the user-prefix plugins-bad):

```
$ pytest -m "not gpu and not data and not slow" -q           # dev-only venv, = CI
91 passed, 23 skipped
$ scripts/gpu shared -- pytest tests/ingest -m "gpu and not slow" -v
test_torch_matches_opencv[nv12|i420]                                      2 passed
test_file_decode_identity_and_tap[{gstreamer,pyav}-{h264,h265}-nvdec]      4 passed
test_tap_complete_and_decodable_while_consumer_stalls[×4]                  4 passed
$ scripts/gpu shared -- pytest tests/ingest/test_ingest_meva_faults.py tests/ingest/test_ingest_rtsp.py \
      tests/ingest/test_ingest_probe.py -v
test_meva_farm_faults_recover_per_camera                                  PASSED
test_server_killed_10s_recovers_with_health_events_and_no_thread_leak[gstreamer|pyav]   2 passed
test_silent_stall_detected_within_3s_then_recovers[gstreamer|pyav]                     2 passed
test_probe_*                                                              3 passed
```

## Decode benchmark (10 × 1440p15, 10 min, `gpu exclusive`)

### H264 — PASS

- 10 × 2560x1440 @ 15 fps, backend `gstreamer`, consumer `numpy` (cv2 threads 1), measured window **600.0 s** after a 2.52 s warm-up (all cameras decoding).
- GPU at start: `NVIDIA GeForce RTX 5090, 580.173.02, 300 MiB, 32607 MiB, 4 %, 0 %, 52` (`scripts/gpu exclusive`). Supporting dmon: SM 7.1 % avg · NVDEC 7.5 % avg / 14 % max · 5.8 GiB used (121 samples @ 5 s).
- Process CPU: **0.93 cores** avg (decode + copy + tap + stats + consumer). Consumer took 89983 frames in 16136 micro-batches.

| camera | decoder | decoded | expected | dropped | errors | reconnects | stalls | decode fps | fresh-frame age p50 / p95 / max (s) | first frame (s) | reordered / unmatched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam01 | `nvh264dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0334 / 0.0632 / 0.0828 | 2.21 | 3 / 15 |
| cam02 | `nvh264dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0332 / 0.0629 / 0.0836 | 2.5 | 5 / 24 |
| cam03 | `nvh264dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0334 / 0.0632 / 0.0892 | 2.01 | 3 / 7 |
| cam04 | `nvh264dec` | 9000 | 9000 | 0 | 0 | 0 | 0 | 15.000 | 0.0332 / 0.0631 / 0.0709 | 2.34 | 4 / 67 |
| cam05 | `nvh264dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0334 / 0.0631 / 0.0704 | 1.86 | 1 / 16 |
| cam06 | `nvh264dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0333 / 0.0633 / 0.0884 | 2.42 | 4 / 18 |
| cam07 | `nvh264dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0333 / 0.0632 / 0.1339 | 1.76 | 4 / 8 |
| cam08 | `nvh264dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0332 / 0.0631 / 0.0956 | 1.94 | 1 / 14 |
| cam09 | `nvh264dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0333 / 0.0627 / 0.1044 | 2.11 | 0 / 15 |
| cam10 | `nvh264dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0331 / 0.0623 / 0.131 | 1.7 | 0 / 19 |
| **total** | | **89985** | 90000 | **0** | **0** | | | | worst p95 **0.0633** | | |

### H265 — PASS

- 10 × 2560x1440 @ 15 fps, backend `gstreamer`, consumer `numpy` (cv2 threads 1), measured window **600.0 s** after a 2.5 s warm-up (all cameras decoding).
- GPU at start: `NVIDIA GeForce RTX 5090, 580.173.02, 300 MiB, 32607 MiB, 2 %, 0 %, 55` (`scripts/gpu exclusive`). Supporting dmon: SM 6.2 % avg · NVDEC 5.9 % avg / 7 % max · 5.8 GiB used (121 samples @ 5 s).
- Process CPU: **0.92 cores** avg (decode + copy + tap + stats + consumer). Consumer took 89985 frames in 16271 micro-batches.

| camera | decoder | decoded | expected | dropped | errors | reconnects | stalls | decode fps | fresh-frame age p50 / p95 / max (s) | first frame (s) | reordered / unmatched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam01 | `nvh265dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0333 / 0.0635 / 0.0678 | 2.41 | 8 / 14 |
| cam02 | `nvh265dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0332 / 0.0625 / 0.0945 | 2.21 | 3 / 13 |
| cam03 | `nvh265dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0333 / 0.0633 / 0.1079 | 1.84 | 1 / 15 |
| cam04 | `nvh265dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0333 / 0.0635 / 0.1145 | 1.76 | 2 / 23 |
| cam05 | `nvh265dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0333 / 0.0632 / 0.1299 | 2.48 | 3 / 17 |
| cam06 | `nvh265dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0333 / 0.0629 / 0.1249 | 2.1 | 0 / 13 |
| cam07 | `nvh265dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0334 / 0.0631 / 0.1049 | 1.94 | 8 / 20 |
| cam08 | `nvh265dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0332 / 0.0629 / 0.1313 | 1.67 | 0 / 6 |
| cam09 | `nvh265dec` | 8999 | 9000 | 0 | 0 | 0 | 0 | 14.998 | 0.0334 / 0.0634 / 0.0711 | 2.31 | 8 / 18 |
| cam10 | `nvh265dec` | 8998 | 9000 | 0 | 0 | 0 | 0 | 14.996 | 0.0334 / 0.0626 / 0.1052 | 2.0 | 2 / 12 |
| **total** | | **89986** | 90000 | **0** | **0** | | | | worst p95 **0.0635** | | |

Clocks: fresh-frame age = host monotonic now − last decode, sampled at 10 Hz over the window (one frame period at 15 fps is 0.067 s). Decode fps = decoded / wall time. Inputs: `python -m scs.ingest.bench prepare` (MEVA, CC-BY-4.0, **upscaled** 1080p→1440p, H.264 ≈ 6 Mbit/s, H.265 ≈ 4–4.7 Mbit/s, 2 s GOP, no B-frames).


## MEVA fault run (reconnect / stall)

Four MEVA 1440p15 RTSP streams (cam01/03 H.264, cam02/04 H.265) through a private mediamtx, with `RtspSource` on GStreamer + NVDEC and health on an `InMemoryBus`. `tests/ingest/test_ingest_meva_faults.py` produced this; the JSON is at `runs/T02/meva_faults.json`.

| t (s) | fault | detected as | recovered (first frame of a new epoch) |
|---|---|---|---|
| 12 | cam01 publisher killed 10 s (path offline) | `disconnected` at 12.06 (session EOS) | +17.64 s (publisher back at +10 s; the rest is backoff) |
| 34 | cam02 publisher SIGSTOP 6 s (RTSP up, no frames) | `stalled` at 36.99 (+2.99 s) | +6.13 s |
| 56 | mediamtx killed 10 s, restarted | all 4 `disconnected` within 0.24 s | +16.13–18.17 s (server back at +10 s; backoff) |

Isolation: no other camera changed state during cam01's drop or cam02's stall, and no `stalled` was reported anywhere else, including during backoff waits. 
Identity: every camera has `decoded == packets`, `reordered == unmatched == 0`, and 0 tap overflow. Tap packets per camera: {'cam01': 943, 'cam02': 1186, 'cam03': 1210, 'cam04': 1241}. OS threads before/after: 32/34.

Capture→decode (`ts − source_ts`: decode wall clock minus mediamtx's RTCP NTP clock, the same host) p50/p95 ms: {'cam01': (3.0439, 9.9003), 'cam02': (2.8007, 10.0992), 'cam03': (3.526, 10.3126), 'cam04': (3.7851, 9.6493)}.

Note: after SIGCONT, the ffmpeg publisher bursts its 6 s backlog. rtspsrc's jitterbuffer then goes quiet once more, so cam02 stalls a second time and settles in its next epoch. A resuming camera sends live frames, not a backlog, so this is an artifact of the test publisher.

## What was built

- **`RtspSource`** (main stream by default, URL from an env var only), **`FileSource`**
  (`loop`, `realtime` pacing), **`DirectorySource`** (sim frames), all `FrameSource`.
- **Backends.** GStreamer primary: `rtspsrc`/`parsebin` → `h26Xparse` → `tee` → packet
  tap ∥ `nvh26Xdec` (NVDEC) → NV12 → appsink. PyAV fallback: demux → Annex-B bitstream
  filter → packet tap → `h264_cuvid`/`hevc_cuvid` (NVDEC) or software. H.264 and H.265 are
  detected from caps. `stats.decoder` names the decoder that actually ran.
- **One decode worker thread per camera**, pushing into a bounded drop-oldest analytics
  queue. Decoding never waits on the consumer. `CameraManager.next_batch()` gathers
  micro-batches with a deadline (D9).
- **Stall detection:** a watchdog samples fresh-frame age at 10 Hz. If no frame arrives for
  3 s (or no first frame within 10 s of connecting), it emits `stalled` and aborts the
  session. Reconnect uses exponential backoff with jitter, capped at 30 s.
- **CAMERA_HEALTH** `Event`s go to `Streams.HEALTH`: `connected`, `stalled`,
  `disconnected`, `reconnecting`, `eos`, `stopped`. Each carries `epoch`, a redacted
  reason, and a stats snapshot.
- **Per-camera stats:** decoded, delivered, dropped, packets, errors, reconnects, stalls,
  fps, fresh-frame age (now, p50, p95, max), capture→decode ms (p50, p95), skipped pre-key
  packets, reordered/unmatched frames.
- **`scripts/rtsp_probe.py`:** reads the URL from `--env VAR`, redacts credentials in all
  output, and prints codec, profile, size, nominal and measured fps, bitrate, GOP,
  B-frames and audio, plus a `StreamSpec` snippet.
- **`python -m scs.ingest.bench`:** `prepare` builds 1440p15 MEVA loop files on NVENC;
  `run` is the acceptance benchmark.

## Frame identity and time (ADR 0002/0003, D10)

Every `FrameRef` from T02 carries:

| Field | Meaning |
|---|---|
| `camera_id, epoch, seq` | Identity. `epoch` increases by 1 for each *session that delivers data* (first = 0). `seq` restarts at 0 per epoch. |
| `frame_idx` | Per-source counter across epochs and file loops (not an identity) |
| `ts` / `ts_mono` | Decode wall clock / host monotonic clock, taken at the same instant |
| `source_ts` | **Live RTSP (GStreamer):** sender NTP time from RTCP SR (`rtspsrc add-reference-timestamp-meta`). `None` until the first SR arrives, and always `None` with PyAV. With mediamtx it's the server's ingest clock, not the shutter. **Files / directories:** the recording's own timeline, `origin_ts` + media time. The default origin is wall clock at the first packet. It stays continuous across loops and reopens, so durations stay right in unpaced replay (T13 B1). |
| `transform` | `None`: T02 hands out the decoded main-stream frame as-is (RGB conversion isn't geometric) |
| `width/height, stream` | Main-stream dims; `stream="sub"` requires `main_size=` |

`seq` is assigned per **access unit in decode order**, upstream of the tap and the decoder.
A frame inherits its packet's `seq` through a PTS match, so `packet.identity ==
frame.identity`. It held for every frame in the live RTSP runs, including all reconnects.
A frame that matches no packet is counted in `unmatched`; it still gets a unique,
increasing `seq` (see Known gaps for looping files).
Packets before the first keyframe of an epoch are not tapped and get no seq
(`skipped_prekey`). A packet the decoder rejects produces no frame, which leaves a gap in
frame `seq`. That gap is deliberate: it shows the frame was lost (see Blockers). With
B-frames, frame seqs are forced monotonic and `reordered` counts the frames whose packet
seq may differ. Reolink and the T14 farm use no B-frames.

**Clocks named in reports:**

- `fresh_frame_age_*`: host monotonic, now − last decode.
- `capture_to_decode_ms_*`: `ts − source_ts`, i.e. decode wall clock minus the sender's
  NTP clock. It includes any camera↔host clock offset.
- Decode→result latency belongs downstream and uses `ts_mono`.

## Cross-agent notes (T13 RELEASE.md items addressed to T02)

- **B1, the replay clock: T02 signs off on the proposal, and it's implemented.**
  `FileSource` and `DirectorySource` set `source_ts` = `origin_ts` + media time (continuous
  across loops and reopens; tested unpaced on both backends). Pass one `origin_ts` to
  several files to replay them on a shared timeline. T09 and T05 still need to agree on
  `source_ts ?? ts` for durations.
- **R5, the image contract:** T02 yields HxWx3 uint8 RGB (numpy, or a CUDA tensor with
  `output="torch"`). `ref.width/height` are main-stream dims.
- **R7, credentials in argv:** T02 decodes in-process (GStreamer/PyAV), so the URL never
  reaches a subprocess argv. Every error, health reason and `last_error` goes through
  `redact_text`.
- **R10, clip boundaries:** the packet tap gives per-packet `pts_ns`, `ts`, `ts_mono` and
  `source_ts` (below).
- **R11, reconnect time:** T02 follows the brief (backoff 0.5 s × 2ⁿ with ±10 % jitter,
  cap 30 s). After a 10 s outage the next attempt lands within the current backoff step:
  recovery was +18 s after a 10 s publisher drop and +17.7 s after a 10 s server kill (MEVA
  fault run). If stores need faster recovery, lower `Backoff(cap_s=…)` per camera.

## Packet tap API (for T10 evidence ring buffer, T13 durability)

```python
from scs.ingest.sources import RtspSource
from scs.ingest.packets import EncodedPacket, PacketQueue

src = RtspSource("cam01", url_env="SCS_CAM1_RTSP_MAIN", bus=bus)   # or FileSource(...)
unsubscribe = src.tap.subscribe(callback)       # callback(pkt: EncodedPacket) -> None
q = PacketQueue(max_bytes=256 * 2**20)          # or: decouple with a byte-bounded queue
src.tap.subscribe(q); pkt = q.get(timeout=1.0)  # q.overflow counts refused packets (never silent)
src.start()
```

`EncodedPacket` is frozen (slots) and has these fields:

- `camera_id, epoch, seq` (`.identity`, the identity of the frame it decodes to; see the
  loop caveat in Known gaps)
- `pts_ns`: stream clock in ns. It restarts per epoch and per file loop.
- `ts`, `ts_mono`: when the packet left the demuxer/depayloader
- `source_ts`: RTP/NTP capture time, if the sender provides it
- `keyframe`, `codec` (`"h264" | "h265"`)
- `data`: one access unit, Annex-B, AU-aligned. Parameter sets (SPS/PPS, plus VPS for
  H.265) are in-band on every keyframe, so a clip can start at any `keyframe=True` packet
  with no other state.

Guarantees and caveats:

- The tap runs on the decode thread **before** the analytics queue. A slow analytics
  consumer never causes tap loss (tested at 100 % under a fully stalled consumer, and
  the tapped bytes re-decode to one picture per packet).
- Delivery is synchronous. A callback that blocks stalls that camera's decode (and then
  its RTSP session), so T10 should either append to its ring buffer cheaply or use
  `PacketQueue`.
- A raising subscriber is isolated (`tap.errors`); the other subscribers still get the
  packet.
- A reconnect gives a new epoch with `seq` from 0, and its first tapped packet is a
  keyframe. A ring buffer should key on `identity` and treat an epoch change as a
  discontinuity when cutting clips.
- Muxing: concatenating `data` in order gives a valid H.264/H.265 elementary stream.
  Mux to MP4 with `pts_ns` (e.g. `ffmpeg -f h264 -i - -c copy` with timestamps, or PyAV).

## Dev environment on this box (no root)

GStreamer's NVDEC plugin lives in `gstreamer1.0-plugins-bad`, which isn't installed and
there's no sudo. For development, the Ubuntu 24.04 debs are extracted to a user prefix
(`apt-get download gstreamer1.0-plugins-bad libgstreamer-plugins-bad1.0-0
gir1.2-gst-plugins-bad-1.0` → `dpkg -x … ~/.local/gst-bad`) with:

```
export GST_PLUGIN_PATH=~/.local/gst-bad/usr/lib/x86_64-linux-gnu/gstreamer-1.0
export LD_LIBRARY_PATH=~/.local/gst-bad/usr/lib/x86_64-linux-gnu
export GI_TYPELIB_PATH=~/.local/gst-bad/usr/lib/x86_64-linux-gnu/girepository-1.0
```

PyGObject 3.50.0 is built into the uv (managed CPython 3.11) venv against extracted
`libglib2.0-dev`/`libgirepository-1.0-dev` headers, with `-Dpycairo=disabled`. On the
edge box (T12) this is simply
`apt install gstreamer1.0-plugins-bad gstreamer1.0-libav python3-gi` plus PyGObject.
Without GStreamer, `backend="auto"` falls back to PyAV (the `av` wheel ships cuvid).

## Performance notes

- **GIL:** reading a 1440p NV12 frame through PyGObject's `MapInfo.data` copies 5.5 MB into
  `bytes` with the GIL held (2.1 ms/frame). That serialized 10 cameras at ~100 fps total.
  Mapping via ctypes on the `GstBuffer*` and making one numpy copy (GIL released) raised
  unpaced throughput to ~1,000 fps across 10 × 1440p H.265.
- **NVDEC headroom:** 10 parallel ffmpeg `hevc_cuvid` decodes of the 1440p files ran at
  ~136 fps each (dec ≈ 95 %). 10 × 15 fps is ~11 % of NVDEC.
- **Startup burst:** realtime file playback began with a ~1.8 s burst, because the clock
  started before NVDEC had initialized. Fixed by prerolling until the decoder branch has
  a frame; `max-display-delay=0` also removes decoder-side batching.
- **Consumer CPU:** ingest itself (decode, copy, tap, stats) costs ~0.8 cores for
  10 × 1440p15. OpenCV's default thread pool spins ~8 cores converting 150 fps of NV12→RGB;
  `cv2.setNumThreads(1)` brings that to ~0.15 cores. T12's runtime should set it
  process-wide, or use `output="torch"`.

## Known gaps / follow-ups

- **Zero-copy GPU frames:** `output="torch"` uploads NV12 from host memory and converts
  on the GPU. Frames still round-trip NVDEC → host → GPU (~0.8 GB/s at 10 × 1440p15). True
  zero-copy needs GstCuda memory mapped to DLPack. It's a follow-up if T12's budget needs it.
- **`output="torch"` has no long run.** It's correctness-tested (GPU test: matches OpenCV
  within 3 levels), but the 10-minute runs used the numpy consumer; a torch-consumer run
  queued behind other agents' exclusive jobs never ran.
- **PyAV has no live `source_ts`:** PyAV doesn't expose RTCP SR, so PyAV RTSP frames have `source_ts=None`. Files get media time on both backends. GStreamer is primary.
- **T14 farm:** it isn't on `main` yet (`agent/T14-replay`), so the fault run uses a local
  mediamtx with the same topology and fault types on MEVA transcodes. Once the farm lands,
  point it at `rtsp://127.0.0.1:8554/meva/<codec>/<cam>`; the sources need no change.
- **Looping files on NVDEC:** 0.17–0.23 % of frames (≈ 15–20 per camera per 10 min) arrived
  with a PTS that matched no tapped packet: `unmatched` 203 (H.264) and 151 (H.265) of ~90,000,
  plus 25/35 `reordered`. They still get a unique, increasing `seq`, but their packet may
  carry a different one. It never happened on live RTSP (0 in the fault run and the
  RTSP tests), unpaced loops, or software-decode loops. The counts fit ~8 frames at each
  file-loop seek on NVDEC. Moving PTS→seq reservation upstream of the tee cut it from
  ~150/camera to ~15–20; the NVDEC-at-loop cause is still to be confirmed (the GPU was busy
  with other agents' exclusive jobs). T10 should key on identity and treat a missing
  packet match as a gap, not an error. Follow-up: diagnose, then start a new epoch at each
  file loop if it's inherent to NVDEC.
- **1440p inputs are MEVA 1080p upscaled.** Decode cost depends on resolution and
  bitrate, not scene detail, but the numbers aren't from a real Reolink stream. Re-run
  `rtsp_probe` and the bench once cameras exist.

## Blockers

1. **`FrameRef.seq` semantics when the decoder loses a frame (interface/correctness, owner
   decision).** `contracts.py` says `seq` "increases by 1 per decoded frame". T02 assigns
   `seq` per **access unit**, so `packet.identity == frame.identity` always holds, which
   T10 needs to cut clips by identity. In normal operation the two readings agree. They
   differ only when the decoder rejects an AU: the frame `seq` then skips a number, which
   marks a real lost frame. The ADR 0003 alternatives cite exactly this ("how many frames
   were missed").
   - **Option A (implemented, recommended):** allow gaps. Needs a one-line clarification
     in the `contracts.py` Conventions and ADR 0003 (owner/ADR; outside T02 paths).
   - **Option B:** strictly contiguous frame `seq`. Packets would then need a separate
     field to name their frame (e.g. optional `FrameRef.pts_ns`, an additive contract
     change), and T10 would join on it.
   Nothing else in T02 depends on the choice; only the gap-on-loss edge case is on hold.
2. **Docker daemon on the 5090 box is wedged (environment; needs root).**
   - A `docker rm -f` of one of T02's test containers (`scs-t02-test-1906056-34981`,
     still listed as "Up" with no containerd shim) has been hanging since ~00:23.
   - T14's `docker kill scs-replay-mediamtx` also hangs, so this affects T14's farm too.
   - Likely fix: `sudo systemctl restart docker`, then `docker rm -f scs-t02-test-1906056-34981`.
   - T02's tests no longer depend on Docker: they use the mediamtx release binary via
     `SCS_MEDIAMTX_BIN`, and every Docker call has a timeout.
3. Not blocking, just requests: pyproject extras/mypy override, DATA.md registration of
   `ingest_bench`, edge-box packages. See `docs/reports/T02-HANDOFF.md`.
