"""StreamSource and CameraManager with a scripted fake backend: identity, tap, stall, health (CI)."""

from __future__ import annotations

import threading
import time

import pytest
from ingest_helpers import FakeSession, SessionScript

from scs.bus import InMemoryBus
from scs.contracts import Event, Streams
from scs.ingest.backoff import Backoff
from scs.ingest.base import FrameSource
from scs.ingest.manager import CameraManager
from scs.ingest.packets import PacketQueue, PacketTap
from scs.ingest.source import StreamSource


def _src(*sessions: FakeSession, camera_id: str = "cam1", **kw) -> StreamSource:
    kw.setdefault("output", "raw")
    kw.setdefault("backoff", Backoff(initial_s=0.01, cap_s=0.05))
    return StreamSource(camera_id, SessionScript(*sessions), **kw)


def _states(src: StreamSource) -> list[str]:
    return [e.data["state"] for e in src.health.events]


# ------------------------------------------------------------------ tap subscribers


def test_packet_queue_refuses_newest_on_overflow_and_counts():
    q = PacketQueue(max_bytes=10)
    tap = PacketTap()
    tap.subscribe(q)
    src = _src(FakeSession(n=5), reconnect=False)  # 6-byte packets: only one fits in 10 bytes
    src.tap = tap
    list(src.frames())
    src.close()
    assert q.received == 1 and q.overflow == 4
    assert q.get(0).seq == 0


def test_tap_isolates_failing_subscriber():
    tap = PacketTap()
    good = PacketQueue()
    tap.subscribe(lambda p: 1 / 0)
    unsub = tap.subscribe(good)
    src = _src(FakeSession(n=7), reconnect=False)
    src.tap = tap
    list(src.frames())
    src.close()
    assert good.received == 7 and tap.errors == 7
    unsub()
    assert len(tap._subs) == 1


# ------------------------------------------------------------------ identity & time


def test_frame_identity_and_timestamps_monotonic():
    src = _src(FakeSession(n=50), reconnect=False, queue_size=100)
    assert isinstance(src, FrameSource)
    refs = [r for r, _ in src.frames()]
    src.close()
    assert [r.seq for r in refs] == list(range(50))
    assert [r.frame_idx for r in refs] == list(range(50))
    assert {r.epoch for r in refs} == {0}
    assert all(b.ts_mono >= a.ts_mono and b.ts >= a.ts for a, b in zip(refs, refs[1:], strict=False))
    assert all(r.transform is None and r.stream == "main" and (r.width, r.height) == (8, 4) for r in refs)
    assert refs[3].identity == ("cam1", 0, 3)


def test_epoch_increments_and_seq_restarts_on_reconnect():
    src = _src(FakeSession(n=5, end="error"), FakeSession(n=5, end="error"), FakeSession(n=5, end="hang"),
               queue_size=100)
    got = []
    for ref, _ in src.frames():
        got.append(ref.identity)
        if len(got) == 15:
            break
    src.close()
    assert got == [("cam1", e, s) for e in range(3) for s in range(5)]
    assert len(set(got)) == 15  # identity never repeats across reconnects
    assert src.stats.reconnects == 2 and src.stats.errors == 2


def test_packets_carry_frame_identity_and_skip_prekey():
    q = PacketQueue()
    src = _src(FakeSession(n=20, lead_delta=3), reconnect=False, queue_size=100)
    src.tap.subscribe(q)
    refs = [r for r, _ in src.frames()]
    src.close()
    pkts = q.drain()
    assert src.stats.skipped_prekey == 3
    assert [p.identity for p in pkts] == [r.identity for r in refs]
    assert pkts[0].keyframe and pkts[0].seq == 0
    assert [p.keyframe for p in pkts].count(True) == 2


def test_sub_stream_requires_main_size():
    with pytest.raises(ValueError, match="main_size"):
        _src(FakeSession(), stream="sub")
    src = _src(FakeSession(n=2), stream="sub", main_size=(2560, 1440), reconnect=False)
    ref, img = next(iter(src.frames()))
    src.close()
    assert (ref.width, ref.height, ref.stream) == (2560, 1440, "sub") and img.width == 8


# ------------------------------------------------------------------ packet tap under consumer stall


def test_packet_tap_gets_every_packet_while_analytics_drops():
    n = 600
    q = PacketQueue()
    src = _src(FakeSession(n=n, fps=600), reconnect=False, queue_size=2)
    src.tap.subscribe(q)
    src.start()
    src._thread.join(10)  # consumer never reads: fully stalled
    s = src.stats
    assert s.decoded == n and s.packets == n
    assert s.dropped == n - 2  # analytics queue kept only the 2 newest
    pkts = q.drain()
    assert len(pkts) == n and [p.seq for p in pkts] == list(range(n))  # 100 %, in order, no gaps
    left = [r.seq for r, _ in src.frames()]
    assert left == [n - 2, n - 1]
    src.close()


# ------------------------------------------------------------------ stall, health, threads


def test_stall_detected_and_reconnects_with_health_events():
    bus = InMemoryBus()
    src = _src(FakeSession(n=100, fps=100, hang_after=5), FakeSession(n=5, end="hang"),
               stall_timeout_s=0.3, bus=bus, queue_size=100)
    got = []
    t0 = time.monotonic()
    for ref, _ in src.frames():
        got.append(ref.identity)
        if len(got) == 10:
            break
    elapsed = time.monotonic() - t0
    src.close()
    assert got[:5] == [("cam1", 0, i) for i in range(5)] and got[5:] == [("cam1", 1, i) for i in range(5)]
    assert 0.3 <= elapsed < 3.0
    st = _states(src)
    assert st[:5] == ["connected", "stalled", "disconnected", "reconnecting", "connected"]
    assert st[-1] == "stopped" and src.stats.stalls == 1
    on_bus = [ev.data["state"] for _, ev in bus.consume(Streams.HEALTH, "t", "t", Event)]
    assert on_bus == st
    stalled = src.health.events[1]
    assert stalled.type == "camera_health" and stalled.data["fresh_frame_age_s"] >= 0.3


def test_first_frame_timeout_triggers_reconnect():
    src = _src(FakeSession(n=0, end="hang"), FakeSession(n=3, end="hang"), connect_timeout_s=0.3,
               queue_size=100)
    refs = []
    for ref, _ in src.frames():
        refs.append(ref)
        if len(refs) == 3:
            break
    src.close()
    assert refs[0].epoch == 0  # the silent session delivered nothing, so it never got an epoch
    assert "no first frame" in src.health.events[0].data["reason"]


def test_no_thread_leak_after_close():
    before = threading.active_count()
    srcs = [_src(FakeSession(n=10**6, fps=200), camera_id=f"cam{i}") for i in range(5)]
    mgr = CameraManager(srcs).start()
    time.sleep(0.3)
    batch = mgr.next_batch(deadline_s=0.05)
    assert 1 <= len(batch) <= 5
    mgr.close()
    time.sleep(0.1)
    assert threading.active_count() == before
    assert not mgr.alive


def test_manager_rejects_duplicate_ids_and_reports_stats():
    with pytest.raises(ValueError, match="duplicate"):
        CameraManager([_src(FakeSession()), _src(FakeSession())])
    src = _src(FakeSession(n=40, fps=200), reconnect=False, queue_size=1)
    mgr = CameraManager([src]).start()
    src._thread.join(5)
    snap = mgr.snapshot()["cam1"]
    mgr.close()
    assert snap["decoded"] == 40 and snap["dropped"] == 39 and snap["backend"] == "fake"
    assert snap["decoder"] == "fakedec" and snap["fresh_frame_age_samples"] >= 1


# ------------------------------------------------------------------ credentials in health events

# Built at runtime so the repo's secret scanners never see a literal credential URL.
FAKE_CRED = "admin" + ":" + "hunter2" + "@"


def test_health_reason_never_contains_credentials():
    src = _src(FakeSession(n=1, end="error"), reconnect=False)

    class Leaky(FakeSession):
        def run(self, sink):
            return "error", "rtspsrc: could not connect to rtsp://" + FAKE_CRED + "cam/1"

    src._factory = lambda: Leaky()
    list(src.frames())
    src.close()
    assert all("hunter2" not in str(e.data) for e in src.health.events)


def test_finite_source_retries_errors_but_ends_at_eos():
    src = _src(FakeSession(n=3, end="error"), FakeSession(n=4, end="eos"), reconnect=True, end_on_eos=True,
               queue_size=100)
    refs = [r for r, _ in src.frames()]
    src.close()
    expected = [("cam1", 0, i) for i in range(3)] + [("cam1", 1, i) for i in range(4)]
    assert [r.identity for r in refs] == expected
    assert _states(src)[-2:] == ["eos", "stopped"] and src.stats.reconnects == 1


def test_no_stall_reported_while_waiting_in_backoff():
    class Refused(FakeSession):
        def run(self, sink):
            return "error", "connection refused"

    src = _src(Refused(), connect_timeout_s=0.2, stall_timeout_s=0.2,
               backoff=Backoff(initial_s=0.8, factor=1.0, cap_s=0.8, jitter=0.0))
    src.start()
    time.sleep(2.0)  # two long backoff waits, each longer than the connect timeout
    src.close()
    st = _states(src)
    assert "stalled" not in st and src.stats.stalls == 0
    assert st.count("disconnected") >= 2


def test_media_time_source_ts_is_continuous_across_loops():
    class Looping(FakeSession):
        def run(self, sink):
            for _ in range(2):  # two passes of the same 5-frame file, like FileSource(loop=True)
                super().run(sink)
                sink.on_loop()
            return "eos", ""

    src = _src(Looping(n=5), reconnect=False, queue_size=100, media_origin=1000.0)
    refs = [r for r, _ in src.frames()]
    src.close()
    ts = [r.source_ts for r in refs]
    assert [r.seq for r in refs] == list(range(10))
    step = 66_666_667 / 1e9
    assert ts[0] == pytest.approx(1000.0)
    assert all(b - a == pytest.approx(step, abs=1e-6) for a, b in zip(ts, ts[1:], strict=False))


def test_live_sources_have_no_media_time():
    src = _src(FakeSession(n=3), reconnect=False, queue_size=10)
    refs = [r for r, _ in src.frames()]
    src.close()
    assert all(r.source_ts is None for r in refs)
