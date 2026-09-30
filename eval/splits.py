"""Group-aware split generator and the frozen split files in `eval/splits/<dataset>.json`.

docs/EVAL.md: splits are by actor, clip and camera, never by frame, and every
derivative of a clip (crops, overlapping windows, emulated variants, augmentations)
shares its `group_id` and its split. How:

1. Clips are joined into *units* with union-find over `group_id`, plus shared actor
   ids (`link_actors`) and shared `view_group` (`link_views`: simultaneous overlapping
   views of the same people, e.g. MEVA camera sets, SmartSpaces scenes).
2. Hold-out rules put whole cameras/sites in `test`/`val`. In a unit that touches a
   held-out clip, the held-out clips and their derivatives (same `group_id`) go to that
   split; clips linked only by view/actor from *other* cameras/sites are **excluded**
   (recorded with a reason), never trained on. Test wins over val.
3. Remaining units are assigned by a stable hash of their smallest group id, so adding
   new clips never moves old ones.

Only IDs are committed. A derivative that isn't listed (e.g. T07's emulated variant of
a test clip) resolves through its `group_id`: `Splits.split_for(clip_id, group_id)`.
Frozen means frozen: regenerate only with a new `version` and say why in the PR.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPLITS_DIR = Path(__file__).resolve().parent / "splits"
SPLIT_NAMES = ("train", "val", "test")
FORMAT = "t09-splits/1"


@dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    group_id: str | None = None
    camera_id: str | None = None
    site_id: str | None = None
    view_group: str | None = None
    actor_ids: tuple[str, ...] | None = None

    @property
    def gid(self) -> str:
        return self.group_id or self.clip_id


@dataclass(frozen=True)
class SplitSpec:
    holdout: dict[str, dict[str, list[str]]] = field(
        default_factory=dict
    )  # split -> {cameras|sites|view_groups}
    fractions: dict[str, float] = field(default_factory=lambda: {"train": 1.0})  # for non-held-out units
    link_actors: bool = True
    link_views: bool = True
    seed: str = "t09"

    def to_json(self) -> dict[str, Any]:
        return {
            "holdout": self.holdout,
            "fractions": self.fractions,
            "link_actors": self.link_actors,
            "link_views": self.link_views,
            "seed": self.seed,
        }


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _held(m: ClipMeta, rule: dict[str, list[str]]) -> bool:
    return (
        (m.camera_id is not None and m.camera_id in rule.get("cameras", []))
        or (m.site_id is not None and m.site_id in rule.get("sites", []))
        or (m.view_group is not None and m.view_group in rule.get("view_groups", []))
        or m.clip_id in rule.get("clips", [])
    )


def _hash01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / float(1 << 48)


def generate(
    dataset: str, clips: Iterable[ClipMeta], spec: SplitSpec, version: str = "1", notes: str = ""
) -> dict[str, Any]:
    clips = sorted(clips, key=lambda m: m.clip_id)
    if len({m.clip_id for m in clips}) != len(clips):
        raise ValueError("duplicate clip ids")
    if abs(sum(spec.fractions.values()) - 1.0) > 1e-9 or not set(spec.fractions) <= set(SPLIT_NAMES):
        raise ValueError(f"fractions must be over {SPLIT_NAMES} and sum to 1: {spec.fractions}")
    uf = _UF()
    for m in clips:
        uf.union("c:" + m.clip_id, "g:" + m.gid)
        if spec.link_actors and m.actor_ids:
            for a in m.actor_ids:
                uf.union("c:" + m.clip_id, "a:" + a)
        if spec.link_views and m.view_group:
            uf.union("c:" + m.clip_id, "v:" + m.view_group)
    units: dict[str, list[ClipMeta]] = defaultdict(list)
    for m in clips:
        units[uf.find("c:" + m.clip_id)].append(m)

    assign: dict[str, str] = {}
    excluded: dict[str, str] = {}
    order = [s for s in ("test", "val", "train") if s in spec.holdout]
    for members in units.values():
        held_in: str | None = next((s for s in order if any(_held(m, spec.holdout[s]) for m in members)),
                                   None)
        if held_in is None:
            key = min(m.gid for m in members)
            x, acc, dest = _hash01(f"{spec.seed}:{dataset}:{key}"), 0.0, "train"
            for s in SPLIT_NAMES:
                acc += spec.fractions.get(s, 0.0)
                if x < acc:
                    dest = s
                    break
            for m in members:
                assign[m.clip_id] = dest
            continue
        target: str = held_in
        anchor = next(m for m in members if _held(m, spec.holdout[target]))
        held_groups = {m.gid for m in members if _held(m, spec.holdout[target])}
        for m in members:
            if _held(m, spec.holdout[target]) or m.gid in held_groups:  # derivatives follow their clip
                assign[m.clip_id] = target
            else:
                excluded[m.clip_id] = f"shares group/actor/view with held-out {target} clip {anchor.clip_id}"
    out = {
        "format": FORMAT,
        "dataset": dataset,
        "version": version,
        "spec": spec.to_json(),
        "notes": notes,
        "splits": {s: sorted(c for c, t in assign.items() if t == s) for s in SPLIT_NAMES},
        "excluded": dict(sorted(excluded.items())),
        "group_of": {m.clip_id: m.group_id for m in clips if m.group_id and m.group_id != m.clip_id},
        "counts": {s: sum(1 for t in assign.values() if t == s) for s in SPLIT_NAMES}
        | {"excluded": len(excluded)},
    }
    validate(out, clips if spec.link_actors else None)
    return out


def validate(d: dict[str, Any], metas: Iterable[ClipMeta] | None = None) -> None:
    """Raise if a clip is listed twice, a group spans splits, or (given metas) an actor does."""
    seen: dict[str, str] = {}
    for s in SPLIT_NAMES:
        for c in d["splits"].get(s, []):
            if c in seen:
                raise ValueError(f"clip {c} in both {seen[c]} and {s}")
            seen[c] = s
    for c in d.get("excluded", {}):
        if c in seen:
            raise ValueError(f"clip {c} is both excluded and in {seen[c]}")
    group_split: dict[str, str] = {}
    for c, s in seen.items():
        g = d.get("group_of", {}).get(c, c)
        if group_split.setdefault(g, s) != s:
            raise ValueError(f"group {g} spans splits {group_split[g]} and {s}")
    if metas is not None:
        actor_split: dict[str, str] = {}
        for m in metas:
            ms = seen.get(m.clip_id)
            if ms is None:
                continue
            for a in m.actor_ids or ():
                if actor_split.setdefault(a, ms) != ms:
                    raise ValueError(f"actor {a} spans splits {actor_split[a]} and {ms}")


@dataclass
class Splits:
    data: dict[str, Any]

    @classmethod
    def load(cls, dataset: str, directory: Path = SPLITS_DIR) -> Splits:
        p = directory / f"{dataset}.json"
        if not p.exists():
            raise FileNotFoundError(f"no frozen split for {dataset!r} at {p}")
        d = json.loads(p.read_text())
        validate(d)
        return cls(d)

    def save(self, directory: Path = SPLITS_DIR) -> Path:
        validate(self.data)
        p = directory / f"{self.data['dataset']}.json"
        p.write_text(json.dumps(self.data, indent=1, sort_keys=False) + "\n")
        return p

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.data["splits"], sort_keys=True).encode()).hexdigest()[:16]

    def clips(self, split: str) -> list[str]:
        return list(self.data["splits"].get(split, []))

    def split_for(self, clip_id: str, group_id: str | None = None) -> str | None:
        """Split of a clip, or of an unlisted derivative via its group. None = excluded/unknown."""
        for s in SPLIT_NAMES:
            if clip_id in self._index(s):
                return s
        if clip_id in self.data.get("excluded", {}):
            return None
        if group_id is not None:
            return self._group_index().get(group_id)
        return None

    def _index(self, s: str) -> set[str]:
        cache = self.__dict__.setdefault("_idx", {})
        if s not in cache:
            cache[s] = set(self.data["splits"].get(s, []))
        return cache[s]

    def _group_index(self) -> dict[str, str]:
        if "_gidx" not in self.__dict__:
            g = {}
            for s in SPLIT_NAMES:
                for c in self.data["splits"].get(s, []):
                    g[self.data.get("group_of", {}).get(c, c)] = s
            self.__dict__["_gidx"] = g
        return self.__dict__["_gidx"]
