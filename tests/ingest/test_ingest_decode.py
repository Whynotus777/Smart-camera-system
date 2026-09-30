"""Real decoders on real H.264/H.265 files: identity, packet tap contents, pacing, looping.

Software-decode cases run wherever GStreamer (with plugins-bad parsers) or PyAV is
installed; NVDEC cases are marked `gpu`. Everything skips cleanly in CI (no av/gi there).
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest
from ingest_helpers import nal_types

from scs.ingest import gst as gst_backend
from scs.ingest import pyav as pyav_backend
from scs.ingest.packets import PacketQueue
from scs.ingest.sources import FileSource

PARAM_SETS = {"h264": {7, 8}, "h265": {32, 33, 34}}  # SPS+PPS / VPS+SPS+PPS
NVDEC_NAMES = {"nvh264dec", "nvh265dec", "h264_cuvid", "hevc_cuvid"}


def _need(backend: str) -> None:
    if backend == "gstreamer" and not gst_backend.available():
        pytest.skip("GStreamer with plugins-bad parsers not available")
    if backend == "pyav" and not pyav_backend.available():
        pytest.skip("PyAV not installed")
    pytest.importorskip("cv2")


CASES = [pytest.param(b, c, d, marks=[pytest.mark.gpu] if d == "nvdec" else [], id=f"{b}-{c}-{d}")
         for b in ("gstreamer", "pyav") for c in ("h264", "h265") for d in ("software", "nvdec")]


@pytest.mark.parametrize(("backend", "codec", "decode"), CASES)
def test_file_decode_identity_and_tap(clips, backend, codec, decode):
    _need(backend)
    q = PacketQueue()
    src = FileSource(clips[codec], "camA", backend=backend, decode=decode, queue_size=1000)
    src.tap.subscribe(q)
    frames = list(src.frames())
    src.close()
    pkts = q.drain()
    refs = [r for r, _ in frames]
    assert len(refs) == 60 and len(pkts) == 60  # 4 s @ 15 fps, no drops with a big queue
    assert [r.seq for r in refs] == list(range(60)) and {r.epoch for r in refs} == {0}
    assert [p.identity for p in pkts] == [r.identity for r in refs]
    assert all(b.ts_mono >= a.ts_mono for a, b in zip(refs, refs[1:], strict=False))
    img = frames[0][1]
    assert img.shape == (360, 640, 3) and (refs[0].width, refs[0].height) == (640, 360)
    keys = [p for p in pkts if p.keyframe]
    assert len(keys) == 4 and pkts[0].keyframe
    for p in keys:  # every keyframe is self-contained for clip cutting
        assert PARAM_SETS[codec] <= nal_types(p.data, codec), p.seq
    assert all(p.codec == codec for p in pkts)
    assert src.stats.unmatched == 0 and src.stats.reordered == 0
    if decode == "nvdec":
        assert src.stats.decoder in NVDEC_NAMES
    else:
        assert src.stats.decoder not in NVDEC_NAMES


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["gstreamer", "pyav"])
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_tap_complete_and_decodable_while_consumer_stalls(clips, tmp_path, backend, codec):
    """Acceptance: tap gets 100 % of packets while the analytics queue drops (consumer stalled)."""
    _need(backend)
    ffprobe = shutil.which("ffprobe") or pytest.skip("ffprobe not installed")
    q = PacketQueue()
    src = FileSource(clips[codec], "camA", backend=backend, decode="nvdec", queue_size=1, loop=True)
    src.tap.subscribe(q)
    src.start()
    deadline = time.monotonic() + 30
    while src.stats.packets < 300 and time.monotonic() < deadline:  # 5 loops, nobody consuming
        time.sleep(0.05)
    src.close()
    s = src.stats
    pkts = q.drain()
    assert s.dropped > 0 and s.dropped >= s.decoded - 1  # analytics side dropped nearly everything
    assert len(pkts) == s.packets >= 300 and q.overflow == 0
    assert [p.seq for p in pkts] == list(range(len(pkts)))  # 100 %, no gaps, one epoch across loops
    # the tapped bytes alone are a valid stream with one picture per packet
    es = tmp_path / f"tap.{codec}"
    es.write_bytes(b"".join(p.data for p in pkts))
    out = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",  # noqa: S603
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(es)],
                         capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) == len(pkts)


@pytest.mark.parametrize("backend", ["gstreamer", "pyav"])
def test_realtime_pacing_and_loop(clips, backend):
    _need(backend)
    src = FileSource(clips["h264"], "camA", backend=backend, decode="software", realtime=True, loop=True,
                     queue_size=1000)
    q = PacketQueue()
    src.tap.subscribe(q)
    t0 = time.monotonic()
    refs = []
    for ref, _ in src.frames():
        refs.append(ref)
        if len(refs) == 75:  # 5 s of 15 fps: crosses one loop boundary at 4 s
            break
    elapsed = time.monotonic() - t0
    src.close()
    assert 4.3 < elapsed < 7.0, elapsed  # paced (unpaced decode of 75 tiny frames takes < 1 s)
    assert [r.seq for r in refs] == list(range(75)) and {r.epoch for r in refs} == {0}
    assert refs[74].frame_idx == 74
    span = refs[74].ts_mono - refs[0].ts_mono
    assert 4.3 < span < 6.5
    # paced playback: the tap and frame appsinks render at the same clock time, so this is
    # where a frame could race ahead of its packet's PTS→seq entry (fixed by reserve_seq)
    assert src.stats.unmatched == 0 and src.stats.reordered == 0
    tapped = {p.identity for p in q.drain()}
    assert all(r.identity in tapped for r in refs)


@pytest.mark.parametrize("backend", ["gstreamer", "pyav"])
def test_file_source_ts_is_media_time_even_unpaced(clips, backend):
    """T13 B1: durations on source_ts are right when replay runs faster than realtime."""
    _need(backend)
    src = FileSource(clips["h265"], "camA", backend=backend, decode="software", loop=True, queue_size=1000,
                     origin_ts=5000.0)
    refs = []
    t0 = time.monotonic()
    for ref, _ in src.frames():
        refs.append(ref)
        if len(refs) == 100:  # crosses the 60-frame loop boundary
            break
    wall = time.monotonic() - t0
    src.close()
    ts = [r.source_ts for r in refs]
    assert ts[0] == pytest.approx(5000.0, abs=0.07)
    steps = [b - a for a, b in zip(ts, ts[1:], strict=False)]
    assert all(s == pytest.approx(1 / 15, abs=2e-3) for s in steps), steps  # incl. across the loop
    assert ts[-1] - ts[0] == pytest.approx(99 / 15, abs=0.01)
    assert wall < ts[-1] - ts[0]  # unpaced: faster than realtime, media clock unaffected
