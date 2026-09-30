"""Helpers shared by the built-in suites (not a suite module: leading underscore)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from eval.canonical import ClipLabels
from eval.datasets import Dataset
from eval.e2e import ClipOutput, E2EJob, run_clips
from eval.metrics.events import GTEvent
from eval.suites.base import RunContext

# Theft-like interaction labels: an Alert overlapping one of these is a true alert.
THEFT_TYPES = frozenset({"item_to_clothing", "item_to_bag", "grab_run", "exit_no_checkout", "shoplifting"})


def git_sha(short: bool = False) -> str:
    try:
        here = Path(__file__).resolve().parent
        sha = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=here,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S603,S607
            capture_output=True,
            text=True,
            check=True,
            cwd=here,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    sha = sha[:12] if short else sha
    return sha + ("-dirty" if dirty else "")


def hours(lab: ClipLabels) -> float:
    if lab.duration_s:
        return lab.duration_s / 3600.0
    if lab.n_frames:
        return lab.n_frames / lab.fps / 3600.0
    raise ValueError(f"{lab.clip_id}: duration unknown; can't compute camera-hours")


def gt_for(
    lab: ClipLabels,
    positive_types: frozenset[str] | set[str],
    ignore_types: frozenset[str] | set[str] = frozenset(),
    subtype_from: str = "subtype",
) -> list[GTEvent]:
    """Roles per docs/EVAL.md: positive = target type AND visible 'observed' for this camera AND
    annotation status good; partially/not observed or flagged -> ignore; other types don't appear
    (alerts on them are false) unless listed in `ignore_types`."""
    out = []
    for i, e in enumerate(lab.events):
        good = e.extra.get("src_status", "good") == "good"
        if e.type in positive_types:
            role = "positive" if (e.visible == "observed" and good) else "ignore"
        elif e.type in ignore_types or (e.source_label or "") in ignore_types:
            role = "ignore"
        else:
            continue
        sub = e.subtype if subtype_from == "subtype" else getattr(e, subtype_from, None)
        out.append(
            GTEvent(
                lab.clip_id,
                e.t_start,
                e.t_end,
                role=role,
                type=e.type,
                subtype=sub,
                event_id=e.event_id or f"{lab.clip_id}:{i}",
            )
        )
    return out


def run_e2e(
    ctx: RunContext, ds: Dataset, clip_ids: list[str], max_frames: int | None = None, realtime: bool = False
) -> dict[str, ClipOutput]:
    """Drive the streaming pipeline over the clips' videos (e2e suites only)."""
    if not ctx.pipeline:
        raise ValueError("end-to-end suite needs --pipeline module:factory (the deployed streaming path)")
    clips = []
    for c in clip_ids:
        lab = ds.clip(c).labels
        video = ds.clip(c).video_path()
        if video is None or not video.exists():
            raise FileNotFoundError(f"{c}: video missing ({lab.video})")
        clips.append((c, lab.camera_id or c, video, lab.fps))
    job = E2EJob(
        ctx.pipeline, {r: m.spec for r, m in ctx.models.items()}, ctx.policy, ctx.source, max_frames, realtime
    )
    sha = git_sha()
    cache = (ctx.out_dir / "_e2e" / job.key(sha)) if (ctx.out_dir and not sha.endswith("-dirty")) else None
    return run_clips(clips, job, cache, ctx.workers, ctx.log)


def crash_summary(outputs: dict[str, ClipOutput]) -> dict[str, object]:
    crashed = {c: o.error.strip().splitlines()[-1] for c, o in outputs.items() if o.error}
    frames = sum(o.n_frames for o in outputs.values())
    wall = sum(o.wall_s for o in outputs.values())
    return {
        "clips": len(outputs),
        "crashed_clips": len(crashed),
        "crashes": crashed,
        "frames": frames,
        "pipeline_fps": round(frames / wall, 2) if wall else None,
        "sources": sorted({o.source for o in outputs.values()}),
    }


def lineage_trained_on(models: dict, dataset: str) -> bool | None:
    """True if any model's lineage lists `dataset` in training, None if some lineage is unknown.
    No declared models = unknown: the pipeline's own components may have trained on it."""
    if not models:
        return None
    unknown = False
    for m in models.values():
        lin = m.lineage
        if lin is None:
            unknown = True
            continue
        if any(t.get("dataset") == dataset for t in lin.get("train", [])):
            return True
    return None if unknown else False
