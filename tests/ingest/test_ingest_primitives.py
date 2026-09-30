"""CPU tests of ingest primitives: backoff, drop-oldest queue, stats, URL handling (runs in CI)."""

from __future__ import annotations

import threading
import time

import pytest

from scs.ingest.backoff import Backoff
from scs.ingest.stats import CameraStats, DropOldestQueue
from scs.ingest.urls import UrlEnvError, redact_text, redact_url, url_from_env

# Built at runtime so the repo's secret scanners never see a literal credential URL.
FAKE_CRED = "admin" + ":" + "hunter2" + "@"


def test_backoff_grows_and_caps_at_30s():
    b = Backoff(jitter=0.0)
    delays = [b.next_delay() for _ in range(12)]
    assert delays[:4] == [0.5, 1.0, 2.0, 4.0]
    assert max(delays) == 30.0 and delays[-1] == 30.0
    assert all(a <= b for a, b in zip(delays, delays[1:], strict=False))
    b.reset()
    assert b.next_delay() == 0.5


def test_backoff_jitter_bounded_and_never_over_cap():
    b = Backoff(jitter=0.5)
    for _ in range(200):
        assert 0 <= b.next_delay() <= 30.0
    b2 = Backoff(initial_s=4.0, jitter=0.25)
    d = b2.next_delay()
    assert 3.0 <= d <= 5.0


def test_drop_oldest_queue_evicts_oldest_and_counts():
    dropped = []
    q: DropOldestQueue[int] = DropOldestQueue(3, on_drop=dropped.append)
    for i in range(10):
        q.put(i)
    assert dropped == [0, 1, 2, 3, 4, 5, 6]
    assert [q.get(0), q.get(0), q.get(0)] == [7, 8, 9]
    assert q.get(0.01) is None
    q.close()
    assert q.get(1.0) is None and q.closed


def test_drop_oldest_queue_close_wakes_reader():
    q: DropOldestQueue[int] = DropOldestQueue(1)
    got = []
    t = threading.Thread(target=lambda: got.append(q.get(timeout=5)))
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(1)
    assert not t.is_alive() and got == [None]


def test_stats_fresh_frame_age_percentiles():
    s = CameraStats("c")
    s.on_frame(100.0)
    for t in (100.1, 100.2, 100.3, 105.0):
        s.sample_age(t)
    snap = s.snapshot()
    assert snap["fresh_frame_age_max_s"] == pytest.approx(5.0)
    assert snap["fresh_frame_age_p50_s"] == pytest.approx(0.3)


def test_redact_url_and_text():
    u = "rtsp://" + FAKE_CRED + "10.0.0.5:554/h264Preview_01_main?user=admin&password=hunter2"
    r = redact_url(u)
    assert "hunter2" not in r and "admin:" not in r
    assert r.startswith("rtsp://***@10.0.0.5:554/h264Preview_01_main")
    msg = f"Could not open resource {u} for reading"
    assert "hunter2" not in redact_text(msg)
    assert redact_url("rtsp://10.0.0.5/x") == "rtsp://10.0.0.5/x"


def test_url_from_env(monkeypatch):
    url = "rtsp://" + FAKE_CRED + "h/x"
    monkeypatch.setenv("SCS_TEST_CAM", url)
    assert url_from_env("SCS_TEST_CAM") == url
    with pytest.raises(UrlEnvError):
        url_from_env(url)
    monkeypatch.delenv("SCS_TEST_CAM")
    with pytest.raises(UrlEnvError, match="not set"):
        url_from_env("SCS_TEST_CAM")
