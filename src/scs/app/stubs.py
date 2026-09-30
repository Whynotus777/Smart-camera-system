"""Deliberately dumb perception + rule stand-ins behind the real interfaces (M1).

Detection quality is out of scope for M1 (T13 brief); what matters is that the pipeline
has the same *shape* as the real one (Detector → Tracker → JourneyEngine) so T03 and T05
can swap their implementations in, and that every stage is deterministic and its state
serializable, so a crashed ingest role can resume from a checkpoint and re-derive
identical events.

- `BackgroundDiffDetector` (Detector): pixels that differ from a fixed background image,
  grouped into blobs. Stateless per frame given the background.
- `IouTracker` (Tracker): greedy IoU matching; state is a small dict.
- `DwellEngine` (JourneyEngine): "some person's foot point stays in the zone for
  ≥ `min_dwell_s`" → one `Event` per dwell episode, with a persisted cooldown per zone,
  and one `Alert` per event. Event/alert IDs are hashes of the dwell's first frame
  identity, so re-processing the same frames yields the same IDs.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

import numpy as np

from scs.contracts import Alert, BBox, BehaviorScore, Detection, Event, EventType, FrameRef, Pose, Track, Zone
from scs.geometry import foot_point, normalize_point, point_in_polygon

SOURCE = "scs.app.stubs@m1"


def stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:32]  # noqa: S324 (not security)


class BackgroundDiffDetector:
    """Blob detector over |frame - background|. Boxes are mapped back to main-stream pixels."""

    def __init__(
        self,
        background: np.ndarray | None = None,
        thresh: int = 30,
        block: int = 4,
        block_frac: float = 0.3,
        min_blocks: int = 8,
        max_dets: int = 3,
    ) -> None:
        self.background = background
        self.thresh, self.block, self.block_frac = thresh, block, block_frac
        self.min_blocks, self.max_dets = min_blocks, max_dets

    def detect(self, batch: list[tuple[FrameRef, Any]]) -> list[list[Detection]]:
        return [self._detect_one(ref, np.asarray(img)) for ref, img in batch]

    def _detect_one(self, ref: FrameRef, img: np.ndarray) -> list[Detection]:
        if self.background is None:
            return []
        h, w = img.shape
        b = self.block
        diff = np.abs(img.astype(np.int16) - self.background.astype(np.int16)) > self.thresh
        gh, gw = h // b, w // b
        grid = diff[: gh * b, : gw * b].reshape(gh, b, gw, b).mean(axis=(1, 3)) > self.block_frac
        blobs = sorted(_components(grid), key=len, reverse=True)
        sx, sy = ref.width / w, ref.height / h
        out = []
        for blob in blobs[: self.max_dets]:
            if len(blob) < self.min_blocks:
                break
            ys = [p[0] for p in blob]
            xs = [p[1] for p in blob]
            bbox: BBox = (min(xs) * b * sx, min(ys) * b * sy, (max(xs) + 1) * b * sx, (max(ys) + 1) * b * sy)
            score = min(1.0, 0.5 + len(blob) / (gh * gw))
            out.append(Detection(frame=ref, bbox=bbox, score=round(score, 4)))
        return out


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a small boolean grid."""
    seen = np.zeros_like(grid, dtype=bool)
    comps = []
    gh, gw = grid.shape
    for y, x in zip(*np.nonzero(grid), strict=True):
        if seen[y, x]:
            continue
        comp, q = [], deque([(int(y), int(x))])
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append(comp)
    return comps


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Greedy IoU tracker. Confirmed after 2 hits, dropped after `max_misses` frames."""

    def __init__(self, iou_min: float = 0.2, max_misses: int = 10) -> None:
        self.iou_min, self.max_misses = iou_min, max_misses
        self.next_id = 0
        self.tracks: list[dict[str, Any]] = []

    def update(self, dets: list[Detection], frame: Any = None) -> list[Track]:
        unmatched = list(range(len(dets)))
        for t in self.tracks:
            best, best_iou = None, self.iou_min
            for j in unmatched:
                if (v := _iou(tuple(t["bbox"]), dets[j].bbox)) >= best_iou:  # type: ignore[arg-type]
                    best, best_iou = j, v
            if best is None:
                t["misses"] += 1
                continue
            unmatched.remove(best)
            t.update(bbox=list(dets[best].bbox), hits=t["hits"] + 1, misses=0, det=best)
        for j in unmatched:
            self.tracks.append(
                {"id": self.next_id, "bbox": list(dets[j].bbox), "hits": 1, "misses": 0, "det": j}
            )
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t["misses"] <= self.max_misses]
        out = []
        for t in self.tracks:
            if t["misses"] == 0 and t["hits"] >= 2:
                d = dets[t["det"]]
                out.append(Track(frame=d.frame, track_id=t["id"], bbox=d.bbox, score=d.score))
        return out

    def state(self) -> dict[str, Any]:
        return {
            "next_id": self.next_id,
            "tracks": [{k: v for k, v in t.items() if k != "det"} for t in self.tracks],
        }

    def restore(self, s: dict[str, Any]) -> None:
        self.next_id = s["next_id"]
        self.tracks = [{**t, "det": -1} for t in s["tracks"]]


class DwellEngine:
    """JourneyEngine stand-in: one dwell-in-zone rule (see module docstring)."""

    def __init__(
        self,
        site_id: str,
        zone: Zone,
        min_dwell_s: float = 5.0,
        cooldown_s: float = 30.0,
        max_gap_s: float = 0.5,
    ) -> None:
        self.site_id, self.zone = site_id, zone
        self.min_dwell_s, self.cooldown_s, self.max_gap_s = min_dwell_s, cooldown_s, max_gap_s
        self.st: dict[str, Any] = {
            "start_ts": None,
            "start_clock": None,
            "start_key": None,
            "last_clock": None,
            "fired": False,
            "last_fire_clock": None,
        }
        self._pending: list[Event] = []

    def on_track(self, t: Track) -> list[Event]:
        if t.state != "confirmed":
            return []
        if not point_in_polygon(normalize_point(foot_point(t.bbox), t.frame), self.zone.polygon):
            return []
        # Durations are measured on the capture clock (`source_ts`, e.g. file PTS) when the
        # source provides one: in unpaced file replay the decode wall clock (`ts`) runs far
        # faster than the video, and a 5 s dwell would never elapse. Emitted times stay `ts`
        # values of seen frames (T09's e2e driver maps those back to media time).
        st, ts = self.st, t.frame.ts
        clock = t.frame.source_ts if t.frame.source_ts is not None else ts
        if st["last_clock"] is None or clock - st["last_clock"] > self.max_gap_s:
            st.update(start_ts=ts, start_clock=clock, start_key=[t.frame.epoch, t.frame.seq], fired=False)
        st["last_clock"] = clock
        cooled = st["last_fire_clock"] is None or clock - st["last_fire_clock"] >= self.cooldown_s
        if st["fired"] or clock - st["start_clock"] < self.min_dwell_s or not cooled:
            return []
        st.update(fired=True, last_fire_clock=clock)
        epoch, seq = st["start_key"]
        ev = Event(
            event_id=stable_id(t.frame.camera_id, "dwell", self.zone.id, epoch, seq),
            type=EventType.ZONE_ENTER,
            camera_id=t.frame.camera_id,
            ts=ts,
            track_id=t.track_id,
            zone_id=self.zone.id,
            confidence=t.score,
            source=SOURCE,
            data={
                "rule": "dwell",
                "min_dwell_s": self.min_dwell_s,
                "dwell_start_ts": st["start_ts"],
                "t_start": st["start_ts"],  # eval.e2e: optional span, as FrameRef.ts values
                "t_end": ts,
                "dwell_start": {"epoch": epoch, "seq": seq},
                "fire": {"epoch": t.frame.epoch, "seq": t.frame.seq},
            },
        )
        self._pending.append(ev)
        return [ev]

    def on_pose(self, p: Pose) -> list[Event]:
        return []

    def on_behavior(self, b: BehaviorScore) -> list[Event]:
        return []

    def poll_alerts(self, now: float) -> list[Alert]:
        out = [
            Alert(
                alert_id=stable_id("alert", e.event_id),
                site_id=self.site_id,
                ts_open=e.ts,
                global_id=e.global_id,
                camera_ids=[e.camera_id],
                score=e.confidence,
                reason_codes=["dwell_in_zone"],
                event_ids=[e.event_id],
            )
            for e in self._pending
        ]
        self._pending = []
        return out

    def reset_transient(self) -> None:
        """After a gap in a live stream the dwell episode is broken; the cooldown survives."""
        self.st.update(start_ts=None, start_clock=None, start_key=None, last_clock=None, fired=False)

    def state(self) -> dict[str, Any]:
        return dict(self.st)

    def restore(self, s: dict[str, Any]) -> None:
        self.st = dict(s)
