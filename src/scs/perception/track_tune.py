"""Tune the appearance tracker on the **val** split only, then freeze (docs/EVAL.md: no test tuning).

Why: the re-ID thresholds trade ID switches against wrong merges, and the right trade-off
depends on the detector and the view. This grid runs on cached val detections from
`track_eval` (CPU only, no GPU lock needed), scores every config on both val sets, and picks
the config with the lowest mean ID-switch rate (relative to default ByteTrack) whose IDF1 is
not below ByteTrack's and whose MOTA is within 2 pts of it on both sets (guard in `main`).
The chosen config is written to JSON and used as-is for the test run
(`track_eval --tracker-config`). Rounds 2-3 refine round 1's winner (see GRID2/GRID3).

  python -m scs.perception.track_tune --dets-smartspaces runs/t03/dets/smartspaces_val_... \
      --dets-meva runs/t03/dets/meva_val_... --out runs/t03/tune
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scs.perception.track_botsort import AppearanceTrackConfig, AppearanceTracker
from scs.perception.track_data import GtClip, meva_clips, smartspaces_clips
from scs.perception.track_eval import PrecomputedEmbedder, evaluate_clip, summarize, tracker_zoo

# Round 2 (after round 1 showed the re-ID knobs are flat): fix the round-1 winner's re-ID
# settings and vary duplicate suppression and the detection thresholds.
GRID2: dict[str, list[Any]] = {
    "track_buffer": [60],
    "reid_thresh": [0.075, 0.1],
    "reid_memory_s": [120.0],
    "reid_long_thresh": [0.03, 0.05],
    "reid_gate_growth": [1.0],
    "dedup_iou": [None, 0.6, 0.75],
    "track_thresh": [0.5, 0.6, 0.7],
}
# Round 3: round 2's winner sat at the track_thresh edge; extend it.
GRID3: dict[str, list[Any]] = {
    "track_buffer": [60],
    "reid_thresh": [0.1],
    "reid_memory_s": [120.0],
    "reid_long_thresh": [0.05],
    "reid_gate_growth": [1.0],
    "dedup_iou": [0.6, 0.75],
    "track_thresh": [0.7, 0.75, 0.8, 0.85],
}
GRID: dict[str, list[Any]] = {
    "track_buffer": [60, 150],
    "reid_thresh": [0.10, 0.15, 0.20],
    "reid_memory_s": [0.0, 30.0, 120.0],
    "reid_long_thresh": [0.05, 0.08, 0.12],
    "reid_gate_growth": [1.0, 3.0],
}


def load_dets(clip: GtClip, cache: Path) -> dict[int, np.ndarray]:
    z = np.load(cache / f"{clip.clip_id.replace('/', '__')}_s3.npz")
    return {int(k): z[k] for k in z.files}


def configs(grid: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    grid = grid or GRID
    keys = list(grid)
    out = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        c = dict(zip(keys, vals, strict=True))
        if c["reid_memory_s"] == 0.0 and c["reid_long_thresh"] != grid["reid_long_thresh"][0]:
            continue  # long threshold is unused without memory
        out.append(c)
    return out


_SETS: dict[str, list[tuple[GtClip, dict[int, np.ndarray]]]] = {}


def _init(sets: dict[str, list[tuple[GtClip, dict[int, np.ndarray]]]]) -> None:
    _SETS.update(sets)


def _score(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    def make() -> AppearanceTracker:
        return AppearanceTracker(AppearanceTrackConfig(frame_rate=10.0, **cfg), PrecomputedEmbedder())

    return cfg, {
        k: summarize([evaluate_clip(c, d, 3, {"a": make}, det_metrics=False) for c, d in v], ci=False)[
            "track"
        ]["a"]
        for k, v in _SETS.items()
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets-smartspaces", type=Path, required=True)
    ap.add_argument("--dets-meva", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/t03/tune"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3])
    a = ap.parse_args(argv)
    fps = 10.0
    sets = {
        "smartspaces": [(c, load_dets(c, a.dets_smartspaces)) for c in smartspaces_clips("val", 120.0)],
        "meva": [(c, load_dets(c, a.dets_meva)) for c in meva_clips("val", 2)],
    }
    base = {
        k: summarize([evaluate_clip(c, d, 3, {"b": tracker_zoo(fps)["bytetrack_default"]}) for c, d in v])[
            "track"
        ]["b"]
        for k, v in sets.items()
    }
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(sets,)) as ex:
        results = list(ex.map(_score, configs({1: GRID, 2: GRID2, 3: GRID3}[a.round])))
    for cfg, res in results:
        rel = {k: res[k]["idsw_per_person_min"] / max(base[k]["idsw_per_person_min"], 1e-9) for k in sets}
        # Guard: the switch rate alone improves by tracking fewer people (track_thresh 0.85 gives
        # ~0 switches and IDF1 0.02). So the summary metrics must not regress: IDF1 >= baseline and
        # MOTA (which charges misses) >= baseline - 2 pts (AGENTS.md rule 7). Track recall is
        # recorded and reported, not gated.
        ok = all(res[k]["idf1"] >= base[k]["idf1"] and res[k]["mota"] >= base[k]["mota"] - 0.02 for k in sets)
        rows.append(
            {
                "cfg": cfg,
                "rel_idsw": rel,
                "mean_rel_idsw": float(np.mean(list(rel.values()))),
                "idf1": {k: res[k]["idf1"] for k in sets},
                "track_recall": {k: res[k]["track_recall"] for k in sets},
                "mota": {k: res[k]["mota"] for k in sets},
                "idf1_ok": ok,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    feasible = [r for r in rows if r["idf1_ok"]]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "grid.json").write_text(
        json.dumps({"baseline_val": base, "rows": rows}, indent=2, default=float)
    )
    if not feasible:  # never fall back to a config that fails the guard
        print("chosen: none (no config passes the guard)")
        return
    best = min(feasible, key=lambda r: r["mean_rel_idsw"])
    chosen = asdict(AppearanceTrackConfig(frame_rate=fps, **best["cfg"]))
    (a.out / "appearance_tuned.json").write_text(json.dumps(chosen, indent=2))
    print("chosen:", json.dumps(best, default=float))


if __name__ == "__main__":
    main()
