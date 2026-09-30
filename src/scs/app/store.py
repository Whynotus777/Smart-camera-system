"""SQLite system of record for the walking skeleton.

One database file, shared by every role process in WAL mode. The tables are an
outbox chain: the ingest role writes `events` + `alerts` + `clips(pending)` +
`checkpoint` in one transaction; the clipper turns `clips` rows to `done`; the review
page writes `reviews`. Each row's primary key is deterministic (see `scs.app`), so
replaying after a crash is idempotent.

T10 replaces this with its own review-queue storage; `tests/integration/` checks the
same invariants against whatever store is behind the flow.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scs.contracts import Alert, AlertStatus, Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS epochs (
  camera_id TEXT NOT NULL, epoch INTEGER NOT NULL, started REAL NOT NULL,
  PRIMARY KEY (camera_id, epoch));
CREATE TABLE IF NOT EXISTS checkpoint (
  camera_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, media_frame INTEGER NOT NULL,
  ts REAL NOT NULL, state TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, ts REAL NOT NULL, json TEXT NOT NULL,
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
  camera_id TEXT NOT NULL, ts REAL NOT NULL, json TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS clips (
  alert_id TEXT PRIMARY KEY REFERENCES alerts(alert_id), camera_id TEXT NOT NULL,
  t0 REAL NOT NULL, t1 REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
  path TEXT, clip_start REAL, clip_end REAL, attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
  alert_id TEXT PRIMARY KEY REFERENCES alerts(alert_id), decision TEXT NOT NULL,
  reason TEXT, reviewer TEXT, request_id TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS segments (
  camera_id TEXT NOT NULL, epoch INTEGER NOT NULL, idx INTEGER NOT NULL,
  t0 REAL NOT NULL, t1 REAL NOT NULL, path TEXT NOT NULL,
  PRIMARY KEY (camera_id, epoch, idx));
CREATE INDEX IF NOT EXISTS segments_t ON segments(camera_id, t0);
"""

DECISIONS = ("confirmed", "dismissed")


@dataclass(frozen=True)
class Checkpoint:
    camera_id: str
    epoch: int
    media_frame: int  # last frame fully processed (file sources resume at media_frame + 1)
    ts: float
    state: dict[str, Any]


@dataclass(frozen=True)
class ClipJob:
    alert_id: str
    camera_id: str
    t0: float
    t1: float
    status: str
    path: str | None
    clip_start: float | None
    clip_end: float | None
    attempts: int


@dataclass(frozen=True)
class Review:
    alert_id: str
    decision: str
    reason: str | None
    reviewer: str | None
    request_id: str
    ts: float


class ReviewConflict(Exception):
    """A different outcome is already recorded for this alert."""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: we issue BEGIN IMMEDIATE ourselves so writers never deadlock
        # on a read→write upgrade across processes.
        self._db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=30000")
        with self.tx() as c:  # executescript() would COMMIT our BEGIN; run statements one by one
            for stmt in _statements(SCHEMA):
                c.execute(stmt)

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # ------------------------------------------------------------------ ingest

    def new_epoch(self, camera_id: str) -> int:
        with self.tx() as c:
            row = c.execute("SELECT MAX(epoch) FROM epochs WHERE camera_id=?", (camera_id,)).fetchone()
            epoch = 0 if row[0] is None else row[0] + 1
            c.execute("INSERT INTO epochs VALUES (?,?,?)", (camera_id, epoch, time.time()))
        return epoch

    def load_checkpoint(self, camera_id: str) -> Checkpoint | None:
        r = self._db.execute("SELECT * FROM checkpoint WHERE camera_id=?", (camera_id,)).fetchone()
        if r is None:
            return None
        return Checkpoint(r["camera_id"], r["epoch"], r["media_frame"], r["ts"], json.loads(r["state"]))

    def commit_frames(
        self,
        cp: Checkpoint,
        events: list[Event],
        alerts: list[Alert],
        clip_windows: dict[str, tuple[float, float]],
    ) -> int:
        """Atomically persist a checkpoint and whatever the frames since the last one produced.

        Returns how many events were new (re-derived duplicates are ignored).
        """
        now = time.time()
        new = 0
        with self.tx() as c:
            for e in events:
                cur = c.execute(
                    "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?)",
                    (e.event_id, e.camera_id, e.ts, e.model_dump_json(), now),
                )
                new += cur.rowcount
            for a in alerts:
                c.execute(
                    "INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?)",
                    (a.alert_id, a.event_ids[0], a.camera_ids[0], a.ts_open, a.model_dump_json(), now),
                )
                t0, t1 = clip_windows[a.alert_id]
                c.execute(
                    "INSERT OR IGNORE INTO clips (alert_id, camera_id, t0, t1, updated) VALUES (?,?,?,?,?)",
                    (a.alert_id, a.camera_ids[0], t0, t1, now),
                )
            c.execute(
                "INSERT OR REPLACE INTO checkpoint VALUES (?,?,?,?,?,?)",
                (cp.camera_id, cp.epoch, cp.media_frame, cp.ts, json.dumps(cp.state), now),
            )
        return new

    def inject(self, event: Event, alert: Alert, window: tuple[float, float]) -> bool:
        """Insert a test event + alert + clip job outside the ingest path (M1 'injected test event')."""
        now = time.time()
        with self.tx() as c:
            new = c.execute(
                "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?)",
                (event.event_id, event.camera_id, event.ts, event.model_dump_json(), now),
            ).rowcount
            c.execute(
                "INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?)",
                (
                    alert.alert_id,
                    event.event_id,
                    alert.camera_ids[0],
                    alert.ts_open,
                    alert.model_dump_json(),
                    now,
                ),
            )
            c.execute(
                "INSERT OR IGNORE INTO clips (alert_id, camera_id, t0, t1, updated) VALUES (?,?,?,?,?)",
                (alert.alert_id, alert.camera_ids[0], window[0], window[1], now),
            )
        return bool(new)

    # --------------------------------------------------------------- segments

    def add_segments(self, rows: list[tuple[str, int, int, float, float, str]]) -> None:
        with self.tx() as c:
            c.executemany("INSERT OR IGNORE INTO segments VALUES (?,?,?,?,?,?)", rows)

    def segments(self, camera_id: str, t0: float, t1: float) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM segments WHERE camera_id=? AND t1>? AND t0<? ORDER BY t0, epoch, idx",
            (camera_id, t0, t1),
        ).fetchall()

    def prune_segments(self, older_than: float) -> list[str]:
        """Delete segment rows ending before `older_than` that no pending clip needs; return their paths."""
        with self.tx() as c:
            rows = c.execute(
                "SELECT s.camera_id, s.epoch, s.idx, s.path FROM segments s WHERE s.t1 < ? AND NOT EXISTS ("
                " SELECT 1 FROM clips j WHERE j.status='pending' AND j.camera_id=s.camera_id"
                " AND j.t0 < s.t1 AND s.t0 < j.t1)",
                (older_than,),
            ).fetchall()
            c.executemany(
                "DELETE FROM segments WHERE camera_id=? AND epoch=? AND idx=?",
                [(r[0], r[1], r[2]) for r in rows],
            )
        return [r[3] for r in rows]

    def segment_count(self, camera_id: str, epoch: int) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM segments WHERE camera_id=? AND epoch=?", (camera_id, epoch)
            ).fetchone()[0]
        )

    # ------------------------------------------------------------------ clips

    def clip_jobs(self, status: str | None = None) -> list[ClipJob]:
        q = "SELECT * FROM clips" + (" WHERE status=?" if status else "") + " ORDER BY t0"  # noqa: S608
        rows = self._db.execute(q, (status,) if status else ()).fetchall()
        return [
            ClipJob(
                r["alert_id"],
                r["camera_id"],
                r["t0"],
                r["t1"],
                r["status"],
                r["path"],
                r["clip_start"],
                r["clip_end"],
                r["attempts"],
            )
            for r in rows
        ]

    def clip_attempt(self, alert_id: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE clips SET attempts=attempts+1, updated=? WHERE alert_id=?", (time.time(), alert_id)
            )

    def clip_done(self, alert_id: str, path: str, clip_start: float, clip_end: float) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE clips SET status='done', path=?, clip_start=?, clip_end=?, error=NULL, updated=? "
                "WHERE alert_id=?",
                (path, clip_start, clip_end, time.time(), alert_id),
            )

    def clip_failed(self, alert_id: str, error: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE clips SET status='failed', error=?, updated=? WHERE alert_id=?",
                (error[:2000], time.time(), alert_id),
            )

    # ----------------------------------------------------------------- alerts

    def alerts(self) -> list[tuple[Alert, ClipJob | None, Review | None]]:
        rows = self._db.execute(
            "SELECT a.json, c.alert_id AS c_id, c.camera_id, c.t0, c.t1, c.status, c.path, c.clip_start,"
            " c.clip_end, c.attempts, r.decision, r.reason, r.reviewer, r.request_id, r.ts AS r_ts"
            " FROM alerts a LEFT JOIN clips c USING(alert_id) LEFT JOIN reviews r USING(alert_id)"
            " ORDER BY a.ts"
        ).fetchall()
        out = []
        for r in rows:
            alert = Alert.model_validate_json(r["json"])
            job = (
                ClipJob(
                    r["c_id"],
                    r["camera_id"],
                    r["t0"],
                    r["t1"],
                    r["status"],
                    r["path"],
                    r["clip_start"],
                    r["clip_end"],
                    r["attempts"],
                )
                if r["c_id"]
                else None
            )
            rev = (
                Review(alert.alert_id, r["decision"], r["reason"], r["reviewer"], r["request_id"], r["r_ts"])
                if r["decision"]
                else None
            )
            if rev is not None:
                alert = alert.model_copy(update={"status": AlertStatus(rev.decision)})
            if job is not None and job.status == "done":
                alert = alert.model_copy(update={"clip_uris": [f"/clips/{alert.alert_id}.mp4"]})
            out.append((alert, job, rev))
        return out

    def event(self, event_id: str) -> Event:
        r = self._db.execute("SELECT json FROM events WHERE event_id=?", (event_id,)).fetchone()
        return Event.model_validate_json(r[0])

    def event_ids(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT event_id FROM events ORDER BY ts")]

    # ---------------------------------------------------------------- reviews

    def record_review(
        self,
        alert_id: str,
        decision: str,
        request_id: str,
        reason: str | None = None,
        reviewer: str | None = None,
    ) -> tuple[Review, bool]:
        """Record one outcome per alert. Returns (review, created).

        Idempotent: repeating the same decision (e.g. a client retry after a lost response)
        returns the stored row. A *different* decision raises `ReviewConflict`; changing an
        outcome is an explicit, audited operation that M1 doesn't offer.
        """
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        with self.tx() as c:
            if c.execute("SELECT 1 FROM alerts WHERE alert_id=?", (alert_id,)).fetchone() is None:
                raise KeyError(alert_id)
            r = c.execute("SELECT * FROM reviews WHERE alert_id=?", (alert_id,)).fetchone()
            if r is not None:
                rev = Review(
                    r["alert_id"], r["decision"], r["reason"], r["reviewer"], r["request_id"], r["ts"]
                )
                if rev.decision != decision:
                    raise ReviewConflict(f"{alert_id} already {rev.decision}")
                return rev, False
            rev = Review(alert_id, decision, reason, reviewer, request_id, time.time())
            c.execute(
                "INSERT INTO reviews VALUES (?,?,?,?,?,?)",
                (rev.alert_id, rev.decision, rev.reason, rev.reviewer, rev.request_id, rev.ts),
            )
        return rev, True

    def reviews(self) -> list[Review]:
        return [
            Review(r["alert_id"], r["decision"], r["reason"], r["reviewer"], r["request_id"], r["ts"])
            for r in self._db.execute("SELECT * FROM reviews ORDER BY ts")
        ]


def _statements(script: str) -> list[str]:
    return [s.strip() for s in script.split(";") if s.strip()]
