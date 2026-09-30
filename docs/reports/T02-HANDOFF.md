# T02 hand-off: changes needed outside T02's owned paths

T02 owns `src/scs/ingest/`, `scripts/rtsp_probe.py`, `tests/ingest/` and
`docs/reports/T02-*.md`. The items below touch files owned by others, so they are
requests and T02 hasn't edited them.

## pyproject.toml (owner: T00 / T12)

- Add a GStreamer extra so the primary backend installs the same way everywhere:
  `gst = ["pygobject>=3.48,<3.51"]`. 3.51+ needs girepository-2.0; Ubuntu 24.04 ships
  1.80, which is the 1.0 API. PyGObject needs system GLib/GI headers at build time; on this
  box it was built with `-Dpycairo=disabled` (see the T02 report's "Dev environment" section).
  License: LGPL-2.1+, dynamically imported, fine for commercial use.
- mypy: `[[tool.mypy.overrides]] module = ["gi", "gi.*"] ignore_missing_imports = true`.
  `gst.py` currently uses per-line `type: ignore[import-untyped]`.
- `av` (PyAV, BSD-3; the wheel bundles LGPL FFmpeg with cuvid) is already in `perception`.

## docs/DATA.md (owner: T14 / data registry)

Register the derived benchmark set (built by `python -m scs.ingest.bench prepare`):

| id | what | license | use | where |
|---|---|---|---|---|
| `ingest_bench` | 10 MEVA clips (one per camera), 5 min, upscaled 1080p→2560×1440, 15 fps, 2 s GOP, no B-frames, H.264 ≈ 6 Mbit/s + H.265 ≈ 4 Mbit/s (NVENC) | CC-BY-4.0 (derived from MEVA; attribution in `ingest_bench/MANIFEST.json`) | prod (decode benchmarking only) | `<data_root>/ingest_bench/` (3.7 GB) |

## Edge box / runtime (owner: T12)

- System packages: `gstreamer1.0-plugins-{base,good,bad}` (rtspsrc, h26Xparse, nvcodec),
  `gstreamer1.0-libav` (software fallback decoders), GI typelibs, and an NVIDIA driver that
  provides `libnvcuvid`. On Jetson the decoder is `nvv4l2decoder`; `gst.pick_decoder`
  already tries it, but it's untested.
- Call `cv2.setNumThreads(1)` (or a small number) process-wide. OpenCV's default pool
  spins ~8 cores on 150 fps of 1440p NV12→RGB; one thread costs ~0.15 cores.
- Budget inputs: 10 × 1440p15 decode uses ~11 % of the 5090's NVDEC (ffmpeg cuvid saturates
  at ~1,360 fps of 1440p HEVC) and ~0.8 CPU cores for ingest itself.
- One Python process per ≤ 10 cameras is fine. Measured GIL headroom is ~1,000 fps of 1440p
  after the ctypes buffer-map fix.

## docs/ARCHITECTURE.md §4 (owner: architecture)

The `FrameSource` signature is unchanged. The docstring in `src/scs/ingest/base.py` now
says what ADR 0002 already requires: images are HxWx3 **RGB** uint8,
and `ref.width/height` are **main-stream** dims (they equal the image size unless
`stream == "sub"` or `transform` is set). Worth mirroring in §4.

## T10 (evidence) / T13 (durability)

Packet tap API and guarantees: see "Packet tap API" in `docs/reports/T02-ingest.md`.

## T14 (replay farm)

Nothing is required. `RtspSource(url=farm.url(codec, cam))` works against the farm as-is.
T02's fault test uses its own mediamtx on a free port (never :8554), so it can run next to a
live farm.
