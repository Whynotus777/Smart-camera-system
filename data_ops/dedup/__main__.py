"""python -m data_ops.dedup [--jobs N] [--no-phash]: write <data_root>/dedup/groups.jsonl for T09.

Sources scanned (whatever is present): MEVA raw clips, SmartSpaces camera videos, replay
loop files (derived from MEVA), generated clips (derived from SmartSpaces keyframes).
Video signatures are cached by file sha256 in dedup/signatures.jsonl, so re-runs only hash
new files. Near-duplicate candidates are also written to dedup/near_duplicates.jsonl
for a human to eyeball.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

from data_ops.dedup import phash
from data_ops.dedup.groups import Grouper, load_clip_table, meva_view_group
from data_ops.paths import data_root, dataset_dir


def _manifest(ds: str) -> dict:
    p = dataset_dir(ds, "MANIFEST.json")
    return json.loads(p.read_text())["files"] if p.exists() else {}


def collect(g: Grouper) -> dict[str, tuple[Path, str | None]]:
    """Register items + derivative links; return {item: (video path, sha256)} for hashing."""
    videos: dict[str, tuple[Path, str | None]] = {}
    root = data_root()
    table_path = dataset_dir("meva", "annotations", "meva-data-repo", "metadata",
                             "meva-clip-camera-and-time-table.txt")
    table = load_clip_table(table_path) if table_path.exists() else None
    for rel, info in _manifest("meva").items():
        clip = Path(rel).name.removesuffix(".r13.avi")
        item = f"meva:{clip}"
        g.add(item, "meva", meva_view_group(clip, table))
        videos[item] = (dataset_dir("meva", "raw", rel), info.get("sha256"))
    for rel, info in _manifest("smartspaces").items():
        if rel.endswith("video.mp4"):
            parts = rel.split("/")  # MTMC_Tracking_2024/test/scene_071/camera_0635/video.mp4
            item = f"smartspaces:{parts[2]}/{parts[3]}"
            g.add(item, "smartspaces", f"smartspaces:{parts[2]}")
            videos[item] = (dataset_dir("smartspaces", "raw", rel), info.get("sha256"))
    replay = root / "replay" / "MANIFEST.json"
    if replay.exists():
        for name, info in json.loads(replay.read_text())["files"].items():
            item = f"replay:{name}"
            g.add(item, "replay")
            g.link(item, f"meva:{info['source_clip']}", "derivative:replay_loop")
    gen = root / "generated"
    for lin in sorted(gen.glob("*/lineage/*.json")):
        d = json.loads(lin.read_text())
        item = f"generated:{lin.parent.parent.name}/{d['clip_id']}"
        g.add(item, "generated")
        src = d["source"]
        g.link(item, f"smartspaces:{src['scene']}/{src['camera']}", "derivative:generated_from_keyframe")
    return videos


def _sig(args: tuple[str, str]) -> tuple[str, list[int]]:
    path, sha = args
    return sha, phash.video_signature(Path(path))


def signatures(videos: dict[str, tuple[Path, str | None]], jobs: int, log=print) -> dict[str, list[int]]:
    cache_path = data_root() / "dedup" / "signatures.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[r["sha256"]] = r["sig"]
    todo = {sha: str(p) for p, sha in videos.values() if sha and sha not in cache and p.exists()}
    log(f"dedup: {len(videos)} videos, {len(todo)} need hashing")
    with cache_path.open("a") as f, ProcessPoolExecutor(jobs) as ex:
        for sha, sig in ex.map(_sig, [(p, s) for s, p in todo.items()], chunksize=4):
            cache[sha] = sig
            f.write(json.dumps({"sha256": sha, "sig": sig}) + "\n")
    return {item: cache[sha] for item, (_, sha) in videos.items() if sha in cache}


def near_duplicates(sigs: dict[str, list[int]],
                    threshold: float = phash.NEAR_DUP) -> list[tuple[str, str, float]]:
    out = []
    for a, b in combinations(sorted(sigs), 2):
        dist = phash.signature_distance(sigs[a], sigs[b])
        if dist <= threshold:
            out.append((a, b, dist))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_ops.dedup")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--no-phash", action="store_true", help="derivative links only")
    a = ap.parse_args(argv)
    g = Grouper()
    videos = collect(g)
    dups: list[tuple[str, str, float]] = []
    if not a.no_phash:
        dups = near_duplicates(signatures(videos, a.jobs))
        for x, y, _ in dups:
            g.link(x, y, "near_duplicate:phash")
    out = data_root() / "dedup"
    out.mkdir(parents=True, exist_ok=True)
    rows = g.rows()
    (out / "groups.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "near_duplicates.jsonl").write_text(
        "".join(json.dumps({"a": x, "b": y, "median_hamming": d}) + "\n" for x, y, d in dups))
    n_groups = len({r["group_id"] for r in rows})
    print(f"dedup: {len(rows)} items, {n_groups} groups, {len(dups)} near-duplicate pairs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
