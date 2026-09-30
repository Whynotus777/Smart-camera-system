"""Regenerate the frozen split files: python -m eval.make_splits {meva|smartspaces}.

The *policy* lives here in code (reviewable); the output in eval/splits/ is IDs only.
Bump `version` and explain in the PR whenever the output changes.

meva (proxy, not retail) — held out by camera AND site, no simultaneous-view leakage:
- test: site `bus` (G331, G508: unseen site) + camera `school.G421` (unseen camera at a
  seen site). Camera sets (shared fields of view) are 3-331, 3-508, 3-421, so no test
  camera shares a view with a train camera: nothing needs excluding.
- val: camera `school.G423` (own camera set) — thresholds are fit here, then frozen.
- train: school G299, G330, G419, G420; admin G326, G329.
- All indexed indoor clips are listed (annotated or not) so the file doesn't change as
  T14's downloads grow. Actor-disjointness is impossible: MEVA actor ids are per clip.
  Self-supervised pretraining (T06) must use `train` clips only, or `meva_fa` leaks.

poselift (real store, pose only) — the OFFICIAL test split (clips under STG-NF `pose/test`,
i.e. the ones with a test frame mask) so numbers compare with published STG-NF; val = a
stable 10% hash of official-train clips (calibration only). Generated from converted
labels (`extra.official_split`), so it needs the converted dataset.

smartspaces (synthetic retail) — held out by scene (all cameras of a scene are
simultaneous views): test scene_073, val scene_072, train scene_071; scenes without GT
(scene_074) still get listed by the same rule (train by default). All scenes share one
retail space and character set, so held-out scenes still share appearance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.canonical import converted_dir, data_root, read_labels
from eval.converters.meva import INDOOR_CAMERAS, load_clip_table, parse_clip_name, view_group
from eval.splits import ClipMeta, Splits, SplitSpec, generate

MEVA_SPEC = SplitSpec(
    holdout={"test": {"sites": ["bus"], "cameras": ["school.G421"]}, "val": {"cameras": ["school.G423"]}},
    fractions={"train": 1.0},
    link_actors=True,
    link_views=True,
)
SMARTSPACES_SPEC = SplitSpec(
    holdout={
        "test": {"view_groups": ["smartspaces:scene_073"]},
        "val": {"view_groups": ["smartspaces:scene_072"]},
    },
    fractions={"train": 1.0},
    link_actors=False,
    link_views=True,
)


def meva_metas(root: Path) -> list[ClipMeta]:
    d = root / "meva"
    table = load_clip_table(
        d / "annotations" / "meva-data-repo" / "metadata" / "meva-clip-camera-and-time-table.txt"
    )
    clips = sorted(
        {json.loads(line)["clip"] for line in (d / "index" / "clips.jsonl").read_text().splitlines()}
    )
    out = []
    for c in clips:
        m = parse_clip_name(c)
        if m["camera_id"] in INDOOR_CAMERAS:
            out.append(ClipMeta(c, None, m["camera_id"], m["site"], view_group(c, table)))
    return out


def smartspaces_metas(root: Path) -> list[ClipMeta]:
    man = json.loads((root / "smartspaces" / "MANIFEST.json").read_text())
    out = []
    for rel in man["files"]:
        if rel.endswith("/video.mp4"):
            scene, cam = rel.split("/")[2:4]
            out.append(
                ClipMeta(
                    f"{scene}.{cam}",
                    None,
                    f"smartspaces.{scene}.{cam}",
                    "smartspaces_retail",
                    f"smartspaces:{scene}",
                )
            )
    return out


def poselift_split(root: Path) -> dict:
    labs = [read_labels(p) for p in sorted((converted_dir("poselift", root) / "labels").glob("*.json"))]
    if not labs:
        raise FileNotFoundError("poselift not converted: python -m eval.converters poselift")
    unknown = [lab.clip_id for lab in labs if lab.extra.get("official_split") not in ("train", "test")]
    if unknown:
        raise ValueError(
            f"{len(unknown)} clips without an official split (e.g. {unknown[:3]}): refusing to guess"
        )
    metas = [ClipMeta(lab.clip_id, camera_id=lab.camera_id) for lab in labs]
    test = [lab.clip_id for lab in labs if lab.extra["official_split"] == "test"]
    spec = SplitSpec(
        holdout={"test": {"clips": test}},
        fractions={"train": 0.9, "val": 0.1},
        link_actors=False,
        link_views=False,
    )
    return generate(
        "poselift",
        metas,
        spec,
        version="1",
        notes="official PoseLift test split (STG-NF layout); val = 10% hash of official train",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["meva", "smartspaces", "poselift"])
    ap.add_argument("--out", type=Path, help="split dir (default eval/splits/)")
    a = ap.parse_args(argv)
    root = data_root()
    if a.dataset == "poselift":
        d = poselift_split(root)
    elif a.dataset == "meva":
        d = generate(
            "meva",
            meva_metas(root),
            MEVA_SPEC,
            version="1",
            notes="proxy, not retail. Source: T14 index/clips.jsonl + MEVA clip table (camera sets). "
            "Actor-disjoint split impossible (per-clip actor ids).",
        )
    else:
        d = generate(
            "smartspaces",
            smartspaces_metas(root),
            SMARTSPACES_SPEC,
            version="1",
            notes="synthetic retail (Isaac Sim), held out by scene; scenes share one space/character set.",
        )
    p = Splits(d).save(a.out) if a.out else Splits(d).save()
    print(f"{a.dataset}: {d['counts']} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
