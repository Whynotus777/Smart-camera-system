"""Fast unit checks of the M1 store guarantees the integration tests rely on (no ffmpeg)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from scs.app.evidence import SegmentEvidence
from scs.app.store import Checkpoint, ReviewConflict, Store
from scs.contracts import Alert, Event, EventType


def _event_alert(ts: float, n: int = 0) -> tuple[Event, Alert]:
    ev = Event(event_id=f"{n:032x}", type=EventType.ZONE_ENTER, camera_id="cam01", ts=ts, source="test")
    al = Alert(
        alert_id=f"{n + 1000:032x}",
        site_id="s",
        ts_open=ts,
        camera_ids=["cam01"],
        score=1.0,
        reason_codes=["t"],
        event_ids=[ev.event_id],
    )
    return ev, al


def test_commit_is_idempotent_on_replay(tmp_path: Path) -> None:
    st = Store(tmp_path / "db")
    ev, al = _event_alert(100.0)
    cp = Checkpoint("cam01", 0, 10, 100.0, {"x": 1})
    assert st.commit_frames(cp, [ev], [al], {al.alert_id: (85.0, 110.0)}) == 1
    # a restarted ingest re-derives the same event from the same frames
    assert st.commit_frames(cp, [ev], [al], {al.alert_id: (85.0, 110.0)}) == 0
    assert st.event_ids() == [ev.event_id] and len(st.alerts()) == 1 and len(st.clip_jobs()) == 1


def test_review_once_retry_ok_conflict_refused(tmp_path: Path) -> None:
    st = Store(tmp_path / "db")
    ev, al = _event_alert(100.0)
    st.inject(ev, al, (85.0, 110.0))
    _, created = st.record_review(al.alert_id, "confirmed", "r1")
    assert created
    rev, created = st.record_review(al.alert_id, "confirmed", "r2")  # client retry
    assert not created and rev.request_id == "r1"
    with pytest.raises(ReviewConflict):
        st.record_review(al.alert_id, "dismissed", "r3")
    with pytest.raises(KeyError):
        st.record_review("f" * 32, "confirmed", "r4")
    assert len(st.reviews()) == 1


def test_segment_prune_keeps_footage_pending_clips_need(tmp_path: Path) -> None:
    st = Store(tmp_path / "db")
    now = time.time()
    for i in range(10):  # 50 s segments ending 550 s ago ... 50 s ago
        f = tmp_path / f"{i}.ts"
        f.write_bytes(b"x")
        st.add_segments([("cam01", 0, i, now - 600 + i * 50, now - 550 + i * 50, str(f))])
    ev, al = _event_alert(now - 480)
    st.inject(ev, al, (now - 495, now - 470))  # pending clip: needs segment 2
    assert SegmentEvidence(st, retain_s=120).prune() == 8
    assert sorted(int(p.stem) for p in tmp_path.glob("*.ts")) == [2, 9]
