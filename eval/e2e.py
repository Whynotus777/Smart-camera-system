"""Streaming end-to-end driver: video file -> the deployed streaming pipeline -> Events/Alerts.

docs/EVAL.md: end-to-end suites run the deployed streaming path (sampling, causal
smoothing, track resets, dedupe, alert suppression) from video files, never saved
model scores. This module is the only way e2e suites get predictions:

- A *source* decodes the file frame by frame in order (T02's `FileSource` via `--source`;
  `VideoFileSource` below is a fallback so eval isn't blocked on T02).
- A *pipeline* is built per camera stream by a factory (`--pipeline module:factory`):
      factory(camera_id=..., fps=..., policy=..., models=...) -> StreamingPipeline
  and sees each frame exactly once, in order (causal by construction), returning the
  `Event`s/`Alert`s (or `Track`s) it emits at that frame; `flush()` drains at end of stream.
  Factories get `clip_id=` too and must accept unknown kwargs (`**_`).
- Timestamps in emitted items (`Event.ts`, `Alert.ts_open`, optional `Event.data["t_start"/
  "t_end"]`) must be `FrameRef.ts` values of frames the pipeline has already seen. In
  file replay, decode wall clock isn't proportional to media time, so only real frame
  timestamps map back to media time exactly (the driver interpolates between them).
  `ProtocolPipeline` composes the ARCHITECTURE §4 interfaces; T13's app can replace it.
- Everything emitted is recorded with the *media time* of the frame being processed
  when it came out (`t_emit`), so latency is "event end -> alert emit" in stream time,
  independent of how fast the replay ran.

Crashes are caught per clip and reported (soak: crashes must be 0), never hidden.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from eval.metrics.events import Pred
from scs.contracts import Alert, Event, EventType, FrameRef, Track

# ---------------------------------------------------------------------------
# Streaming contract
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamingPipeline(Protocol):
    def process(self, ref: FrameRef, image: Any) -> list[Event | Alert | Track]: ...

    def flush(self, now: float) -> list[Event | Alert | Track]: ...


PipelineFactory = Callable[..., StreamingPipeline]


class VideoFileSource:
    """Fallback `FrameSource` over a video file (PyAV, else OpenCV). Not T02's FileSource:
    reports say which source was used. `ts` = wall clock at decode; media time = frame_idx/fps."""

    def __init__(
        self, path: str | Path, camera_id: str, max_frames: int | None = None, realtime: bool = False
    ) -> None:
        self.path, self.camera_id, self.max_frames, self.realtime = (
            Path(path),
            camera_id,
            max_frames,
            realtime,
        )
        self.fps = 0.0
        self.width = self.height = 0

    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray]]:
        try:
            import av as pyav
        except ImportError:
            pyav = None  # type: ignore[assignment]
        if pyav is not None:
            yield from self._pyav(pyav)
        else:
            yield from self._cv2()

    def _ref(self, i: int, w: int, h: int) -> FrameRef:
        return FrameRef(
            camera_id=self.camera_id,
            epoch=0,
            seq=i,
            frame_idx=i,
            ts=time.time(),
            ts_mono=time.monotonic(),
            width=w,
            height=h,
            transform=None,
        )

    def _pace(self, i: int, t0: float) -> None:
        if self.realtime and self.fps:
            dt = t0 + i / self.fps - time.monotonic()
            if dt > 0:
                time.sleep(dt)

    def _pyav(self, av: Any) -> Iterator[tuple[FrameRef, np.ndarray]]:
        with av.open(str(self.path)) as c:
            st = c.streams.video[0]
            self.fps = float(st.average_rate or st.guessed_rate or 0)
            t0 = time.monotonic()
            for i, fr in enumerate(c.decode(st)):
                if self.max_frames is not None and i >= self.max_frames:
                    break
                self._pace(i, t0)
                img = fr.to_ndarray(format="bgr24")
                yield self._ref(i, img.shape[1], img.shape[0]), img

    def _cv2(self) -> Iterator[tuple[FrameRef, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        t0, i = time.monotonic(), 0
        try:
            while self.max_frames is None or i < self.max_frames:
                ok, img = cap.read()
                if not ok:
                    break
                self._pace(i, t0)
                yield self._ref(i, img.shape[1], img.shape[0]), img
                i += 1
        finally:
            cap.release()


class ProtocolPipeline:
    """Reference composition of the ARCHITECTURE §4 interfaces for one camera.

    detect every `detect_every` frames -> tracker.update -> (pose on confirmed tracks) ->
    per-track pose buffer -> behavior.score every `behavior_stride` frames once the buffer
    holds `behavior.window` poses -> journey.on_* -> journey.poll_alerts(now=frame ts).
    Lost tracks drop their buffers (track reset). All causal.
    """

    def __init__(
        self,
        detector: Any,
        tracker: Any,
        journey: Any = None,
        pose: Any = None,
        behavior: Any = None,
        detect_every: int = 1,
        behavior_stride: int = 1,
        emit_tracks: bool = False,
    ) -> None:
        self.detector, self.tracker, self.journey = detector, tracker, journey
        self.emit_tracks = emit_tracks
        self.pose, self.behavior = pose, behavior
        self.detect_every, self.behavior_stride = max(1, detect_every), max(1, behavior_stride)
        self.buffers: dict[int, deque] = {}
        self.n = 0

    def process(self, ref: FrameRef, image: Any) -> list[Event | Alert]:
        out: list[Event | Alert] = []
        self.n += 1
        if (self.n - 1) % self.detect_every:
            return self.journey.poll_alerts(ref.ts) if self.journey is not None else []
        dets = self.detector.detect([(ref, image)])[0]
        tracks = self.tracker.update(dets, image)
        if self.emit_tracks:
            out += tracks
        if self.journey is None:  # detection+tracking only (smartspaces_track)
            return out
        for t in tracks:
            out += self.journey.on_track(t)
            if t.state == "lost":
                self.buffers.pop(t.track_id, None)
        live = [t for t in tracks if t.state == "confirmed"]
        if self.pose is not None and live:
            for p in self.pose.estimate(ref, image, live):
                out += self.journey.on_pose(p)
                if self.behavior is not None:
                    buf = self.buffers.setdefault(p.track_id, deque(maxlen=int(self.behavior.window)))
                    buf.append(p)
                    if len(buf) == buf.maxlen and self.n % self.behavior_stride == 0:
                        out += self.journey.on_behavior(self.behavior.score(list(buf)))
        out += self.journey.poll_alerts(ref.ts)
        return out

    def flush(self, now: float) -> list[Event | Alert | Track]:
        return self.journey.poll_alerts(now) if self.journey is not None else []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class Emitted:
    kind: str  # event | alert | track
    t_emit: float  # media seconds of the frame being processed when emitted
    t: float  # media seconds of Event.ts / Alert.ts_open
    payload: dict[str, Any]


@dataclass
class ClipOutput:
    clip_id: str
    camera_id: str
    fps: float
    n_frames: int = 0
    wall_s: float = 0.0
    items: list[Emitted] = field(default_factory=list)
    error: str | None = None
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ClipOutput:
        items = [Emitted(**x) for x in d.pop("items")]
        return cls(items=items, **d)


def _media_time(ts: float, ts_arr: list[float], fi_arr: list[int], fps: float) -> float:
    """Map a wall-clock ts from the pipeline to media seconds via the frames seen so far."""
    if not ts_arr:
        return 0.0
    return float(np.interp(ts, ts_arr, fi_arr)) / fps


def run_clip(
    clip_id: str,
    camera_id: str,
    video: Path,
    fps_hint: float,
    factory: PipelineFactory,
    factory_kw: dict[str, Any],
    source_factory: Callable[..., Any] | None = None,
    max_frames: int | None = None,
    realtime: bool = False,
) -> ClipOutput:
    src = (source_factory or VideoFileSource)(video, camera_id, max_frames=max_frames, realtime=realtime)
    out = ClipOutput(clip_id, camera_id, fps_hint, source=type(src).__name__)
    ts_arr: list[float] = []
    fi_arr: list[int] = []
    t0 = time.perf_counter()
    pipe = None
    try:
        pipe = factory(camera_id=camera_id, fps=fps_hint, clip_id=clip_id, **factory_kw)
        for ref, img in src.frames():
            fps = getattr(src, "fps", 0) or fps_hint
            out.fps = fps
            ts_arr.append(ref.ts)
            fi_arr.append(ref.frame_idx)
            now_media = ref.frame_idx / fps
            for item in pipe.process(ref, img):
                out.items.append(_record(item, now_media, ts_arr, fi_arr, fps))
            out.n_frames += 1
        now_media = (fi_arr[-1] / out.fps) if fi_arr else 0.0
        for item in pipe.flush(ts_arr[-1] if ts_arr else 0.0):
            out.items.append(_record(item, now_media, ts_arr, fi_arr, out.fps))
    except Exception:  # noqa: BLE001 - any pipeline crash is a result, not a harness failure
        out.error = traceback.format_exc(limit=8)
    out.wall_s = time.perf_counter() - t0
    return out


def _record(
    item: Event | Alert | Track, now_media: float, ts_arr: list[float], fi_arr: list[int], fps: float
) -> Emitted:
    if isinstance(item, Alert):
        return Emitted(
            "alert",
            now_media,
            _media_time(item.ts_open, ts_arr, fi_arr, fps),
            json.loads(item.model_dump_json()),
        )
    if isinstance(item, Event):
        d = json.loads(item.model_dump_json())
        for k in ("t_start", "t_end"):  # optional event span in wall ts -> media seconds
            if isinstance(d["data"].get(k), int | float):
                d["data"][k + "_media"] = _media_time(d["data"][k], ts_arr, fi_arr, fps)
        return Emitted("event", now_media, _media_time(item.ts, ts_arr, fi_arr, fps), d)
    if isinstance(item, Track):  # tracking suites: pipelines may also emit their Tracks
        return Emitted(
            "track",
            now_media,
            _media_time(item.frame.ts, ts_arr, fi_arr, fps),
            {
                "frame_idx": item.frame.frame_idx,
                "track_id": item.track_id,
                "bbox": list(item.bbox),
                "score": item.score,
                "state": item.state,
            },
        )
    raise TypeError(f"pipeline emitted {type(item).__name__}; expected Event, Alert or Track")


def _worker(args: tuple) -> dict[str, Any]:
    from eval.suites.base import load_model, resolve_attr

    clip_id, camera_id, video, fps, pipeline_spec, model_specs, policy, source_spec, max_frames, realtime = (
        args
    )
    factory = resolve_attr(pipeline_spec)
    models = {r: load_model(r, s) for r, s in model_specs.items()}
    src = resolve_attr(source_spec) if source_spec else None
    return run_clip(
        clip_id,
        camera_id,
        Path(video),
        fps,
        factory,
        {"policy": policy, "models": models},
        src,
        max_frames,
        realtime,
    ).to_json()


@dataclass
class E2EJob:
    pipeline: str
    models: dict[str, str]
    policy: dict[str, Any] | None
    source: str | None = None
    max_frames: int | None = None
    realtime: bool = False

    def key(self, sha: str) -> str:
        blob = json.dumps({"sha": sha, **asdict(self)}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_clips(
    clips: list[tuple[str, str, Path, float]],
    job: E2EJob,
    cache_dir: Path | None,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> dict[str, ClipOutput]:
    """Run the pipeline on every (clip_id, camera_id, video, fps); reuse cached outputs of the
    same job key (same code sha, pipeline, models, policy, source, frame cap)."""
    results: dict[str, ClipOutput] = {}
    todo = []
    for c in clips:
        p = cache_dir / f"{c[0]}.json" if cache_dir else None
        if p is not None and p.exists():
            results[c[0]] = ClipOutput.from_json(json.loads(p.read_text()))
        else:
            todo.append(c)
    args = [
        (
            cid,
            cam,
            str(v),
            fps,
            job.pipeline,
            job.models,
            job.policy,
            job.source,
            job.max_frames,
            job.realtime,
        )
        for cid, cam, v, fps in todo
    ]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(workers) if workers > 1 and len(args) > 1 else None
    try:
        it = pool.map(_worker, args) if pool else map(_worker, args)
        for i, d in enumerate(it):
            o = ClipOutput.from_json(d)
            results[o.clip_id] = o
            if cache_dir and o.error is None:
                (cache_dir / f"{o.clip_id}.json").write_text(json.dumps(o.to_json()))
            log(
                f"e2e: {i + 1}/{len(todo)} {o.clip_id}: {o.n_frames} frames, {len(o.items)} items"
                + (f", CRASH: {o.error.splitlines()[-1]}" if o.error else "")
            )
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    return results


# ---------------------------------------------------------------------------
# Outputs -> metric inputs
# ---------------------------------------------------------------------------

INTERACTION_EVENT_TYPES = {EventType.ITEM_PICKUP.value, EventType.SHELF_INTERACTION.value}


def interaction_preds(o: ClipOutput) -> list[Pred]:
    """Scored interaction detections: Events of type item_pickup/shelf_interaction, or any
    Event whose `data["interaction"]` names a canonical interaction label. Score =
    `Event.confidence`; span = `data.t_start/t_end` if given, else the instant `ts`."""
    out = []
    for it in o.items:
        if it.kind != "event":
            continue
        d = it.payload
        if d["type"] not in INTERACTION_EVENT_TYPES and not d["data"].get("interaction"):
            continue
        a = d["data"].get("t_start_media", it.t)
        b = d["data"].get("t_end_media", it.t)
        out.append(
            Pred(
                o.clip_id,
                min(a, b),
                max(a, b),
                float(d["confidence"]),
                t_emit=it.t_emit,
                pred_id=d["event_id"],
            )
        )
    return out


def alert_preds(o: ClipOutput) -> list[Pred]:
    """Alerts on this camera. Window = [earliest referenced event seen on this stream, ts_open];
    if it references none, the instant ts_open. Score = Alert.score."""
    ev_t = {it.payload["event_id"]: it.t for it in o.items if it.kind == "event"}
    out = []
    for it in o.items:
        if it.kind != "alert" or o.camera_id not in it.payload["camera_ids"]:
            continue
        refs = [ev_t[e] for e in it.payload["event_ids"] if e in ev_t]
        start = min(refs) if refs else it.t
        out.append(
            Pred(
                o.clip_id,
                min(start, it.t),
                it.t,
                float(it.payload["score"]),
                t_emit=it.t_emit,
                pred_id=it.payload["alert_id"],
            )
        )
    return out
