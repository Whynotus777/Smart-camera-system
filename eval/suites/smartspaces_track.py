"""`smartspaces_track`: person detection AP@0.5, IDF1, ID switches on overhead SmartSpaces views.

SmartSpaces is synthetic (Isaac Sim): results are labeled "synthetic retail, not
production validation"; the retail-ness is itself a proxy for a c-store.

GT scenes: `smartspaces` split `test` (scene_073), cameras whose scene is retail and has
GT. If no retail scene with GT is available (T14 is checking public GT coverage), the
suite FALLS BACK to labeled non-retail scenes in the split and says so in the labels
(`fallback: non-retail scenes`), so a number is never silently from the wrong domain.

Predictions (not an end-to-end suite, so saved outputs are allowed):
- `--predictions DIR`: canonical `tracks/<clip_id>.parquet` per clip (score = detection score), or
- `--pipeline module:factory`: a streaming pipeline that emits `Track`s (e.g.
  `ProtocolPipeline(detector, tracker, emit_tracks=True)`), run on the videos.
`--opt max_frames=N` caps frames per camera for smoke runs (GT is cut to match).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.canonical import read_tracks
from eval.datasets import load_dataset
from eval.metrics.tracking import TrackFrames, detection_ap, sequence_stats, tracking_metrics
from eval.suites._common import crash_summary, run_e2e
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite


def select_gt_clips(ds, split: str) -> tuple[list[str], bool]:
    """(clip ids, fallback_used): retail scenes with GT, else any labeled scenes."""
    have_gt = [
        c for c in ds.clip_ids(split) if ds.clip(c).labels.supports.gt_tracks and ds.clip(c).has_tracks()
    ]
    retail = [c for c in have_gt if ds.clip(c).labels.extra.get("scene_kind") == "retail"]
    return (retail, False) if retail else (have_gt, bool(have_gt))


def _frames(t, max_frames: int | None, with_scores: bool) -> TrackFrames:
    m = np.ones(len(t), dtype=bool) if max_frames is None else t.frame_idx < max_frames
    return TrackFrames(t.frame_idx[m], t.track_id[m], t.boxes[m], t.score[m] if with_scores else None)


@register_suite
class SmartSpacesTrack(Suite):
    name = "smartspaces_track"
    description = "Detection AP@0.5, IDF1, ID switches per person-minute on SmartSpaces held-out scenes"
    datasets = ("smartspaces",)
    labels = ("synthetic retail (Isaac Sim), not production validation",)
    gated = ("idf1",)

    def run(self, ctx: RunContext) -> SuiteResult:
        max_frames = ctx.opt("max_frames", None, int)
        iou = ctx.opt("iou", 0.5, float)
        try:
            ds = load_dataset("smartspaces", ctx.data_root, ctx.split_dir)
        except FileNotFoundError as e:
            return self.unavailable(str(e))
        clips, fallback = select_gt_clips(ds, ctx.split)
        if ctx.limit:
            clips = clips[: ctx.limit]
        if not clips:
            return self.unavailable(f"no clips with GT tracks in split {ctx.split!r}")
        labels = list(self.labels) + (
            ["fallback: non-retail scenes (no retail GT available)"] if fallback else []
        )
        pipeline_info = None
        preds: dict[str, TrackFrames] = {}
        if ctx.predictions is not None:
            for c in clips:
                p = Path(ctx.predictions) / "tracks" / f"{c}.parquet"
                if not p.exists():
                    return self.unavailable(f"missing prediction file {p}")
                preds[c] = _frames(read_tracks(p), max_frames, True)
        elif ctx.pipeline:
            outs = run_e2e(ctx, ds, clips, max_frames=max_frames)
            pipeline_info = crash_summary(outs)
            for c, o in outs.items():
                rows = [
                    (
                        it.payload["frame_idx"],
                        it.payload["track_id"],
                        *it.payload["bbox"],
                        it.payload["score"],
                    )
                    for it in o.items
                    if it.kind == "track" and it.payload["state"] != "lost"
                ]
                preds[c] = (
                    TrackFrames.from_rows(rows)
                    if rows
                    else TrackFrames(np.zeros(0, int), np.zeros(0, int), np.zeros((0, 4)), np.zeros(0))
                )
        else:
            return self.unavailable("needs --predictions DIR or --pipeline module:factory")
        ok = [c for c in clips if c in preds]
        gts = {c: _frames(ds.clip(c).tracks(), max_frames, False) for c in ok}
        fps = {c: ds.clip(c).labels.fps for c in ok}
        stats = [sequence_stats(gts[c], preds[c], iou, fps[c]) for c in ok]
        tm = tracking_metrics(stats, ctx.bootstrap)
        ap = detection_ap([gts[c] for c in ok], [preds[c] for c in ok], iou)
        notes = [
            "Scenes 071-080 share one retail space and synthetic character set: held-out scenes still "
            "share appearance with train scenes."
        ]
        if max_frames:
            notes.append(f"max_frames={max_frames}: truncated sequences (smoke run)")
        res = self.result(
            "partial" if (ctx.limit or max_frames or fallback) else "ok",
            {"idf1": tm["idf1"], "idsw_per_person_min": tm["idsw_per_person_min"], "det_ap50": ap},
            metrics={
                "mota": tm["mota"],
                "id_switches": tm["id_switches"],
                "clips": ok,
                "pipeline": pipeline_info,
            },
            datasets=[ds.version()],
            notes=notes,
            params={
                "iou": iou,
                "max_frames": max_frames,
                "split": ctx.split,
                "fallback_non_retail": fallback,
            },
        )
        res.labels = labels
        return res
