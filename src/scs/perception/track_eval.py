"""T03 detection + tracking evaluation: mAP-person, IDF1, ID switches per person-minute.

Why a T03-local runner: the acceptance criteria compare detectors and trackers *before*
T09's harness exists. It follows docs/EVAL.md anyway: splits by scene/camera, the
causal (online) tracker path, bootstrap 95% CIs over clips, and a record of git sha,
model ids, GPU and driver in every JSON.

Detections (and their appearance features) are computed once per (detector, clip) and
cached under runs/t03/, so every tracker is scored on identical detections; the only
difference between tracker rows is the tracker.

Usage (GPU; wrap in scripts/gpu):
  python -m scs.perception.track_eval --dataset smartspaces --split test --detector dfine \
      --engine models/dfine-s/dfine_bf16_b16.engine --out runs/t03/eval
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scs.contracts import Detection, FrameRef
from scs.perception.detect import iou_matrix
from scs.perception.detect_trt import env_record
from scs.perception.track_botsort import AppearanceTrackConfig, AppearanceTracker, HsvEmbedder
from scs.perception.track_bytetrack import ByteTrackConfig, ByteTracker
from scs.perception.track_data import GtClip, meva_clips, smartspaces_clips

EVAL_FPS = 10.0  # ARCHITECTURE §5: detection at 10 fps

# --------------------------------------------------------------------------- tracker zoo


class PrecomputedEmbedder:
    """Eval-only: `frame` passed to the tracker *is* the (N,D) feature array for that frame's dets."""

    dim = HsvEmbedder().dim

    def embed(self, frame: Any, boxes: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
        return np.asarray(frame)


def tracker_zoo(fps: float) -> dict[str, Callable[[], Any]]:
    return {
        # baseline for the acceptance criterion: reference defaults, frame_rate = actual input fps
        "bytetrack_default": lambda: ByteTracker(ByteTrackConfig(frame_rate=fps)),
        "bytetrack_tuned": lambda: ByteTracker(
            ByteTrackConfig(frame_rate=fps, track_buffer=90, track_thresh=0.45, match_thresh=0.85)
        ),
        "appearance": lambda: AppearanceTracker(AppearanceTrackConfig(frame_rate=fps), PrecomputedEmbedder()),
        "appearance_noreid": lambda: AppearanceTracker(
            AppearanceTrackConfig(frame_rate=fps, reid_lost=False), PrecomputedEmbedder()
        ),
    }


# --------------------------------------------------------------------------- detection pass


def read_frames(video: Path, indices: Iterable[int]) -> Iterable[tuple[int, np.ndarray]]:
    """Decode the requested frame indices (ascending) as HxWx3 RGB uint8."""
    import cv2

    want = sorted(set(indices))
    cap = cv2.VideoCapture(str(video))
    i, k = 0, 0
    try:
        while k < len(want):
            if not cap.grab():
                break
            if i == want[k]:
                ok, bgr = cap.retrieve()
                if ok:
                    yield i, bgr[:, :, ::-1]
                k += 1
            i += 1
    finally:
        cap.release()


def detect_clip(
    detector: Any, clip: GtClip, stride: int, cache: Path, batch: int = 16
) -> dict[int, np.ndarray]:
    """{frame_idx: (N, 5 + D) x1,y1,x2,y2,score,feat...} for the clip's eval frames, cached as .npz."""
    safe = clip.clip_id.replace("/", "__")
    f = cache / f"{safe}_s{stride}.npz"
    if f.exists():
        z = np.load(f)
        return {int(k): z[k] for k in z.files}
    emb = HsvEmbedder()
    out: dict[int, np.ndarray] = {}
    buf: list[tuple[FrameRef, np.ndarray]] = []

    def flush() -> None:
        dets = detector.detect(buf)
        for (ref, img), ds in zip(buf, dets, strict=True):
            boxes = np.array([d.bbox for d in ds]).reshape(-1, 4)
            scores = np.array([d.score for d in ds]).reshape(-1, 1)
            feats = emb.embed(img, boxes, (ref.width, ref.height)) if len(ds) else np.zeros((0, emb.dim))
            out[ref.frame_idx] = np.c_[boxes, scores, feats].astype(np.float32)
        buf.clear()

    for idx, img in read_frames(clip.video, clip.eval_frames(stride)):
        ref = FrameRef(
            camera_id=clip.camera_id,
            epoch=0,
            seq=idx // stride,
            frame_idx=idx,
            ts=idx / clip.fps,
            width=clip.width,
            height=clip.height,
        )
        buf.append((ref, img))
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **{str(k): v for k, v in out.items()})  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------- metrics


@dataclass
class ClipResult:
    clip_id: str
    person_frames: int
    person_minutes: float
    det_tp: dict[float, np.ndarray]  # iou thr -> tp flags (sorted by nothing; paired with det_scores)
    det_scores: np.ndarray
    n_gt: int
    track: dict[str, dict[str, float]]


def _match_frame(pred: np.ndarray, gt: np.ndarray, thr: float) -> np.ndarray:
    """COCO-style greedy matching by score; returns tp flag per prediction (in pred order)."""
    tp = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    order = np.argsort(-pred[:, 4])
    ious = iou_matrix(pred[order, :4], gt)
    taken = np.zeros(len(gt), bool)
    for r, p in enumerate(order):
        cand = np.where(~taken & (ious[r] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[r, cand])]
            taken[j] = True
            tp[p] = True
    return tp


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side="left")
        ap += prec[k] if k < len(prec) else 0.0
    return ap / 101


def _track_metrics(
    clip: GtClip, frames: list[int], pred: dict[int, tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for idx in frames:
        gid, gbox = clip.gt(idx)
        pid, pbox = pred.get(idx, (np.zeros(0, int), np.zeros((0, 4))))
        d = 1 - iou_matrix(gbox, pbox)
        d[d > 0.5] = np.nan
        acc.update(gid.tolist(), pid.tolist(), d)
    mh = mm.metrics.create()
    s = mh.compute(
        acc,
        metrics=[
            "idtp",
            "idfp",
            "idfn",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_objects",
            "mota",
            "idf1",
            "num_fragmentations",
        ],
        name="x",
    )
    out = {k: float(s[k].iloc[0]) for k in s.columns}
    out.update(_switch_gaps(acc))
    return out


REENTRY_GAP_FRAMES = 30  # 3 s at 10 fps


def _switch_gaps(acc: Any) -> dict[str, float]:
    """Split ID switches by how long the GT person went unmatched before it.

    < 3 s: crossings and short (shelf) occlusions, the per-camera tracker's job.
    >= 3 s: mostly re-entries after leaving the view; needs long-term re-ID.
    """
    ev = acc.mot_events.reset_index()
    last: dict[Any, int] = {}
    short = long = 0
    for fr, typ, oid in zip(ev.FrameId, ev.Type, ev.OId, strict=True):
        if typ not in ("MATCH", "SWITCH"):
            continue
        if typ == "SWITCH":
            if fr - last.get(oid, fr) - 1 >= REENTRY_GAP_FRAMES:
                long += 1
            else:
                short += 1
        last[oid] = fr
    return {"switches_short": float(short), "switches_reentry": float(long)}


def evaluate_clip(
    clip: GtClip,
    dets: dict[int, np.ndarray],
    stride: int,
    trackers: dict[str, Callable[[], Any]],
    score_thresh: float = 0.1,
    det_metrics: bool = True,
) -> ClipResult:
    frames = clip.eval_frames(stride)
    fps = clip.fps / stride
    n_gt = 0
    person_frames = 0
    tps: dict[float, list[np.ndarray]] = {t: [] for t in np.round(np.arange(0.5, 0.96, 0.05), 2)}
    scores = []
    for idx in frames:
        _, gbox = clip.gt(idx)
        p = dets.get(idx, np.zeros((0, 5)))
        n_gt += len(gbox)
        person_frames += len(gbox)
        if det_metrics:
            scores.append(p[:, 4])
        for t in tps if det_metrics else ():
            tps[t].append(_match_frame(p[:, :5], gbox, float(t)))
    res: dict[str, dict[str, float]] = {}
    for name, make in trackers.items():
        trk = make()
        pred: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for seq, idx in enumerate(frames):
            p = dets.get(idx, np.zeros((0, 5)))
            p = p[p[:, 4] >= score_thresh]
            ref = FrameRef(
                camera_id=clip.camera_id,
                epoch=0,
                seq=seq,
                frame_idx=idx,
                ts=idx / clip.fps,
                width=clip.width,
                height=clip.height,
            )
            ds = [Detection(frame=ref, bbox=tuple(map(float, r[:4])), score=float(min(r[4], 1.0))) for r in p]
            out = trk.update(ds, p[:, 5:] if p.shape[1] > 5 else None)
            if out:
                pred[idx] = (np.array([t.track_id for t in out]), np.array([t.bbox for t in out]))
        res[name] = _track_metrics(clip, frames, pred)
    return ClipResult(
        clip.clip_id,
        person_frames,
        person_frames / fps / 60.0,
        {float(t): np.concatenate(v) if v else np.zeros(0, bool) for t, v in tps.items()},
        np.concatenate(scores) if scores else np.zeros(0),
        n_gt,
        res,
    )


def _bootstrap(
    values: Callable[[np.ndarray], float], n: int, reps: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    xs = [values(rng.integers(0, n, n)) for _ in range(reps)]
    xs = [x for x in xs if np.isfinite(x)]
    return (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))) if xs else (float("nan"),) * 2


def summarize(results: list[ClipResult], ci: bool = True) -> dict[str, Any]:
    n = len(results)
    boot = _bootstrap if ci else (lambda *a, **k: (float("nan"), float("nan")))

    def ap_at(sel: np.ndarray, thr: float) -> float:
        tp = np.concatenate([results[i].det_tp[thr] for i in sel])
        sc = np.concatenate([results[i].det_scores for i in sel])
        return average_precision(tp, sc, sum(results[i].n_gt for i in sel))

    def ap5095(sel: np.ndarray) -> float:
        return float(np.mean([ap_at(sel, float(t)) for t in results[0].det_tp]))

    allsel = np.arange(n)
    rec_tp = np.concatenate([r.det_tp[0.5][r.det_scores >= 0.3] for r in results])
    out: dict[str, Any] = {
        "clips": n,
        "gt_boxes": int(sum(r.n_gt for r in results)),
        "person_minutes": round(sum(r.person_minutes for r in results), 2),
        "det": {
            "AP50": ap_at(allsel, 0.5),
            "AP50_ci95": boot(lambda s: ap_at(s, 0.5), n, 200),
            "AP50_95": ap5095(allsel),
            "recall50_at_0.3": float(rec_tp.sum() / max(sum(r.n_gt for r in results), 1)),
        },
        "track": {},
    }
    for name in results[0].track:

        def tot(key: str, sel: np.ndarray, name: str = name) -> float:
            return float(sum(results[i].track[name][key] for i in sel))

        def idsw_pm(sel: np.ndarray) -> float:
            pm = sum(results[i].person_minutes for i in sel)
            return tot("num_switches", sel) / pm if pm else float("nan")

        def idf1(sel: np.ndarray) -> float:
            tp = tot("idtp", sel)
            return 2 * tp / max(2 * tp + tot("idfp", sel) + tot("idfn", sel), 1e-9)

        def mota(sel: np.ndarray) -> float:
            return 1 - (
                tot("num_misses", sel) + tot("num_false_positives", sel) + tot("num_switches", sel)
            ) / max(tot("num_objects", sel), 1)

        def rate(key: str, sel: np.ndarray) -> float:
            pm = sum(results[i].person_minutes for i in sel)
            return tot(key, sel) / pm if pm else float("nan")

        out["track"][name] = {
            "idsw_short_per_person_min": rate("switches_short", allsel),
            "idsw_short_per_person_min_ci95": boot(lambda s: rate("switches_short", s), n),
            "idsw_reentry_per_person_min": rate("switches_reentry", allsel),
            "idsw": int(tot("num_switches", allsel)),
            "idsw_per_person_min": idsw_pm(allsel),
            "idsw_per_person_min_ci95": boot(idsw_pm, n),
            "idf1": idf1(allsel),
            "idf1_ci95": boot(idf1, n),
            "mota": mota(allsel),
            # share of GT person-frames covered by some track (IDSW can be gamed by tracking less)
            "track_recall": 1 - tot("num_misses", allsel) / max(tot("num_objects", allsel), 1),
            "frag": int(tot("num_fragmentations", allsel)),
        }
    return out


class _LazyDetector:
    """Builds the TRT detector on first use, so re-scoring cached detections needs no GPU."""

    def __init__(self, make: Callable[[], Any], fallback_id: str) -> None:
        self._make, self._fallback_id = make, fallback_id
        self._det: Any = None

    @property
    def model_id(self) -> str:
        return self._det.model_id if self._det is not None else self._fallback_id

    def detect(self, batch: list[Any]) -> list[list[Detection]]:
        if self._det is None:
            self._det = self._make()
        return self._det.detect(batch)  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> None:
    from scs.perception.detect import DetectorConfig, make_detector

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["smartspaces", "meva"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--resize", choices=["letterbox", "stretch"], default=None)
    ap.add_argument("--dur", type=float, default=120.0, help="seconds per SmartSpaces camera")
    ap.add_argument("--max-cams", type=int, default=None)
    ap.add_argument("--meva-per-camera", type=int, default=2)
    ap.add_argument("--trackers", default=None, help="comma list; default all")
    ap.add_argument(
        "--tracker-config",
        type=Path,
        default=None,
        help="frozen AppearanceTrackConfig JSON from track_tune → adds 'appearance_tuned'",
    )
    ap.add_argument("--out", type=Path, default=Path("runs/t03/eval"))
    a = ap.parse_args(argv)

    stride = 3  # 30 fps sources → 10 fps
    resize = a.resize or ("stretch" if a.detector.startswith("dfine") else "letterbox")
    # AP needs the low-score tail; trackers still only see >= 0.1 (evaluate_clip)
    cfg = DetectorConfig(resize_mode=resize, score_thresh=0.05)  # type: ignore[arg-type]
    det = _LazyDetector(
        lambda: make_detector(a.detector, engine=a.engine, cfg=cfg), f"{a.detector}:{a.engine}"
    )
    clips = (
        smartspaces_clips(a.split, a.dur, a.max_cams)
        if a.dataset == "smartspaces"
        else meva_clips(a.split, a.meva_per_camera)
    )
    zoo = tracker_zoo(30.0 / stride)
    if a.tracker_config:
        tuned = json.loads(a.tracker_config.read_text())
        zoo["appearance_tuned"] = lambda: AppearanceTracker(
            AppearanceTrackConfig(**tuned), PrecomputedEmbedder()
        )
    if a.trackers:
        zoo = {k: v for k, v in zoo.items() if k in a.trackers.split(",")}
    tag = f"{a.dataset}_{a.split}_{Path(a.engine).parent.name}_{Path(a.engine).stem}"
    cache = Path("runs/t03/dets") / tag
    results = []
    for clip in clips:
        t0 = time.time()
        dets = detect_clip(det, clip, stride, cache)
        results.append(evaluate_clip(clip, dets, stride, zoo))
        print(
            f"{clip.clip_id}: {len(dets)} frames, {results[-1].n_gt} gt, {time.time() - t0:.1f}s", flush=True
        )
    summary = {
        "dataset": a.dataset,
        "split": a.split,
        "detector": det.model_id,
        "engine": a.engine,
        "eval_fps": 30.0 / stride,
        "partial_labels": a.dataset == "meva",
        **summarize(results),
        "per_clip": [
            {
                "clip": r.clip_id,
                "gt": r.n_gt,
                "person_min": r.person_minutes,
                **{k: {m: v[m] for m in ("num_switches", "idf1")} for k, v in r.track.items()},
            }
            for r in results
        ],
        "env": env_record(),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(
        json.dumps(
            {k: summary[k] for k in ("detector", "clips", "person_minutes", "det", "track")},
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
