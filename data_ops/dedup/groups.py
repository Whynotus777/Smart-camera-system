"""group_id assignment, handed to T09's split generator (docs/EVAL.md "derivative groups").

`group_id`: items that must ALWAYS share a split, because one is (nearly) the other:
1. **Derivatives:** a file made from another (re-encode, replay loop file, generated clip
   and its source keyframe/video, emulated variant). Taken from our manifests/lineage.
2. **Near-duplicates:** perceptual-hash distance below `phash.NEAR_DUP`, across any sources.

`view_group` (separate field, T09 decides how to use it): different cameras filming the
same people at the same time. SmartSpaces: the scene. MEVA: the clip table's 5-minute
reference slot + site. Folding these into `group_id` would make "held out by camera"
splits impossible, and that trade-off belongs to T09 (docs/EVAL.md).

Output, one JSON line per item:
{"item", "dataset", "group_id", "group_size", "view_group", "reasons"}.
group_id is a stable hash of the group's lexicographically smallest member, so it doesn't
change when unrelated items are added.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class Grouper:
    uf: UnionFind = field(default_factory=UnionFind)
    dataset: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    view: dict[str, str] = field(default_factory=dict)

    def add(self, item: str, dataset: str, view_group: str | None = None) -> None:
        self.dataset[item] = dataset
        if view_group:
            self.view[item] = view_group
        self.uf.find(item)

    def link(self, a: str, b: str, reason: str) -> None:
        for x in (a, b):
            self.uf.find(x)
            self.reasons[x].add(reason)
        self.uf.union(a, b)

    def link_all(self, items: list[str], reason: str) -> None:
        for other in items[1:]:
            self.link(items[0], other, reason)

    def rows(self) -> list[dict]:
        members: dict[str, list[str]] = defaultdict(list)
        for x in self.uf.parent:
            members[self.uf.find(x)].append(x)
        out = []
        for items in members.values():
            gid = "g_" + hashlib.sha1(min(items).encode(), usedforsecurity=False).hexdigest()[:12]
            for x in sorted(items):
                out.append({"item": x, "dataset": self.dataset.get(x), "group_id": gid,
                            "group_size": len(items), "view_group": self.view.get(x),
                            "reasons": sorted(self.reasons.get(x, ()))})
        return sorted(out, key=lambda r: (r["group_id"], r["item"]))


def meva_view_group(clip: str, clip_table: dict[str, str] | None = None) -> str:
    """'2018-03-07.17-35-06.17-40-06.school.G339' -> 'meva:2018-03-07.17-35-00.school'.

    Uses the clip table's reference slot when given; otherwise floors the start to 5 min.
    """
    date, start, _end, site, _cam = clip.split(".")
    if clip_table and clip in clip_table:
        return f"meva:{clip_table[clip]}.{site}"
    hh, mm, _ss = start.split("-")
    return f"meva:{date}.{hh}-{int(mm) // 5 * 5:02d}-00.{site}"


def load_clip_table(path) -> dict[str, str]:
    """meva-clip-camera-and-time-table.txt -> {clip name: reference time slot}."""
    out = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out
