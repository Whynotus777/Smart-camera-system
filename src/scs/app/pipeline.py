"""The per-frame streaming path, shared by the ingest role and T09's e2e eval driver.

EVAL.md: end-to-end suites must run the *deployed* streaming path, not a re-implementation.
`M1Pipeline` is that path for M1 (detector → tracker → journey engine → alerts), and it
satisfies T09's `eval.e2e.StreamingPipeline` protocol (`process(ref, image)` /
`flush(now)`), so `eval.run --pipeline scs.app.pipeline:factory` scores exactly the
code the ingest role runs. When T03/T05 land, they are swapped in here, once.

Everything is causal (AGENTS.md rule 11): `process` only sees the current and past frames.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from scs.app.config import DEFAULT_ZONE
from scs.app.stubs import BackgroundDiffDetector, DwellEngine, IouTracker
from scs.contracts import Alert, Event, FrameRef, Zone


def analytics_image(image: Any, width: int) -> np.ndarray:
    """Any `FrameSource` image (HxWx3 or HxW, any size; a GPU tensor is copied to host)
    → small HxW uint8 grayscale, by channel mean + integer block averaging (deterministic).

    The M1 ffmpeg source already delivers this; T02's sources deliver full-size color
    frames (FrameSource contract), and so does T09's fallback decoder.
    """
    a = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
    if a.ndim == 3:
        a = a.mean(axis=2)
    k = a.shape[1] // width
    if k > 1:
        h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
        a = a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    return a.astype(np.uint8) if a.dtype != np.uint8 else a


class M1Pipeline:
    def __init__(
        self,
        detector: BackgroundDiffDetector,
        engine: DwellEngine,
        tracker_factory: Callable[[], IouTracker] = IouTracker,
        learn_background: bool = False,
        warmup_frames: int = 40,
        last_epoch: int | None = None,
        analytics_width: int = 160,
    ) -> None:
        """`learn_background`: build the detector's background from the first frames of each
        epoch (live cameras, eval clips); otherwise the detector already has a fixed one."""
        self.detector, self.engine = detector, engine
        self.tracker_factory = tracker_factory
        self.tracker = tracker_factory()
        self.learn_background, self.warmup_frames = learn_background, warmup_frames
        self._warm: list[np.ndarray] = []
        self.last_epoch = last_epoch
        self.analytics_width = analytics_width

    def process(self, ref: FrameRef, image: Any) -> list[Event | Alert]:
        image = analytics_image(image, self.analytics_width)
        if ref.epoch != self.last_epoch:
            if self.last_epoch is not None:
                # A new epoch (reconnect, or a file loop) is a discontinuity: tracks and any
                # running dwell are reset; the per-zone cooldown is kept.
                self.tracker = self.tracker_factory()
                self.engine.reset_transient()
            if self.learn_background:
                self.detector.background, self._warm = None, []
            self.last_epoch = ref.epoch
        if self.learn_background and self.detector.background is None:
            self._warm.append(np.asarray(image))
            if len(self._warm) >= self.warmup_frames:
                self.detector.background = np.median(np.stack(self._warm), axis=0).astype(np.uint8)
                self._warm = []
        out: list[Event | Alert] = []
        for t in self.tracker.update(self.detector.detect([(ref, image)])[0], None):
            out += self.engine.on_track(t)
        out += self.engine.poll_alerts(ref.ts)
        return out

    def flush(self, now: float) -> list[Event | Alert]:
        return list(self.engine.poll_alerts(now))

    def state(self) -> dict[str, Any]:
        return {"tracker": self.tracker.state(), "engine": self.engine.state()}

    def restore(self, s: dict[str, Any]) -> None:
        self.tracker.restore(s["tracker"])
        self.engine.restore(s["engine"])


def factory(
    camera_id: str = "cam01",
    site_id: str = "eval",
    zone: Zone | dict | None = None,
    min_dwell_s: float = 5.0,
    cooldown_s: float = 10.0,
    **_: Any,
) -> M1Pipeline:
    """`eval.e2e` pipeline factory (accepts and ignores the driver's other kwargs)."""
    z = Zone.model_validate(zone) if isinstance(zone, dict) else (zone or DEFAULT_ZONE)
    return M1Pipeline(
        BackgroundDiffDetector(None), DwellEngine(site_id, z, min_dwell_s, cooldown_s), learn_background=True
    )
