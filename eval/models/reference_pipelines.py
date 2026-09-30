"""Harness-validation pipelines with hand-predictable answers. NOT product models.

They implement the `eval.e2e.StreamingPipeline` contract so the whole e2e path
(decode -> pipeline -> media-time mapping -> matching -> CIs -> report) can be checked on
real video with known expected numbers:

- `NullPipeline`: emits nothing -> recall 0, 0 false alerts/hour.
- `PeriodicAlertPipeline(period_s)`: one Alert every `period_s` media seconds (not at
  t=0) -> exactly floor(duration/period) alerts per clip; on theft-free footage FA/h =
  that count / hours.
- `GTReplayPipeline`: reads the clip's canonical labels and emits each labeled event
  as an interaction Event (and an Alert for theft types) once the event has ended,
  i.e. causally -> recall 1.0, 0 false detections, latency = one frame.

Factory signature used by eval.e2e: `factory(camera_id=, fps=, policy=, models=, **_)`.
`GTReplayPipeline` also needs `clip_id` and reads labels from the shared data root.
"""

from __future__ import annotations

from typing import Any

from eval.canonical import converted_dir, read_labels
from scs.contracts import Alert, Event, EventType, FrameRef

THEFT = {"item_to_clothing", "item_to_bag", "grab_run", "exit_no_checkout", "shoplifting"}


class NullPipeline:
    def __init__(self, **_: Any) -> None:
        pass

    def process(self, ref: FrameRef, image: Any) -> list:
        return []

    def flush(self, now: float) -> list:
        return []


class PeriodicAlertPipeline:
    def __init__(self, camera_id: str, fps: float, policy: dict | None = None, **_: Any) -> None:
        self.camera_id, self.fps = camera_id, fps
        self.period = float((policy or {}).get("period_s", 600.0))
        self.next = self.period

    def process(self, ref: FrameRef, image: Any) -> list:
        t = ref.frame_idx / self.fps
        if t + 1e-9 >= self.next:
            self.next += self.period
            return [
                Alert(
                    site_id="eval",
                    ts_open=ref.ts,
                    camera_ids=[self.camera_id],
                    score=0.5,
                    reason_codes=["periodic_test"],
                )
            ]
        return []

    def flush(self, now: float) -> list:
        return []


class GTReplayPipeline:
    def __init__(
        self, camera_id: str, fps: float, clip_id: str, policy: dict | None = None, **_: Any
    ) -> None:
        ds = (policy or {}).get("dataset", "meva")
        self.labels = read_labels(converted_dir(ds) / "labels" / f"{clip_id}.json")
        self.camera_id, self.fps = camera_id, fps
        self.pending = sorted(self.labels.events, key=lambda e: e.t_end)
        self.seen_ts: list[float] = []  # FrameRef.ts per frame seen, to timestamp spans with real frames

    def _ts_at(self, t_media: float) -> float:
        i = min(max(int(round(t_media * self.fps)), 0), len(self.seen_ts) - 1)
        return self.seen_ts[i]

    def process(self, ref: FrameRef, image: Any) -> list:
        self.seen_ts.append(ref.ts)
        now = ref.frame_idx / self.fps
        out: list = []
        while self.pending and self.pending[0].t_end <= now:
            e = self.pending.pop(0)
            ev = Event(
                type=EventType.ITEM_PICKUP,
                camera_id=self.camera_id,
                ts=ref.ts,
                confidence=0.9,
                source="eval.gt_replay",
                data={
                    "interaction": e.type,
                    "t_start": self._ts_at(e.t_start),
                    "t_end": self._ts_at(e.t_end),
                },
            )
            out.append(ev)
            if e.type in THEFT:
                out.append(
                    Alert(
                        site_id="eval",
                        ts_open=ref.ts,
                        camera_ids=[self.camera_id],
                        score=0.9,
                        reason_codes=["gt_replay"],
                        event_ids=[ev.event_id],
                    )
                )
        return out

    def flush(self, now: float) -> list:
        out: list = []
        for e in self.pending:  # events still open at end of stream
            ev = Event(
                type=EventType.ITEM_PICKUP,
                camera_id=self.camera_id,
                ts=self.seen_ts[-1],
                confidence=0.9,
                source="eval.gt_replay",
                data={"interaction": e.type, "t_start": self._ts_at(e.t_start), "t_end": self.seen_ts[-1]},
            )
            out.append(ev)
        self.pending = []
        return out
