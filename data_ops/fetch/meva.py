"""MEVA (Kitware/IARPA, CC-BY-4.0): indoor cameras from the public S3 bucket + activity annotations.

Why indoor only: the free real-footage proxies (`meva_interaction`, `meva_fa`, docs/EVAL.md)
need indoor object interactions and continuous indoor footage; outdoor parking-lot views
are a different problem. MEVA has no "indoor" flag, so the camera list below was made by
(1) MEVA's own camera-model README, which names G299/G330 (gym) as indoor, and (2) a
visual check of one mid-clip frame per camera (2026-09-29, see docs/reports/T14-data.md).
IR cameras (clip-table camera set "IR": G474, G475, G476, G479) are all outdoor.

Order of download: every indoor clip with official Kitware activity annotations first
(`kitware` + `kitware-meva-training`), then unannotated indoor clips (continuous footage
for false-alerts-per-hour), oldest first, until --max-gb.

Layout under <data_root>/meva/:
  raw/<date>/<hour>/<clip>.r13.avi      video, mirrors the S3 key under drops-123-r13/
  annotations/meva-data-repo/            sparse, pinned clone (LICENSE, metadata, documents,
                                         annotation/DIVA-phase-2/MEVA/{kitware,kitware-meva-training})
  index/clips.jsonl                      one row per selected clip: camera, annotated, event counts
  MANIFEST.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from data_ops.budget import GB
from data_ops.fetch.common import RemoteFile, fetch_all, run, tool
from data_ops.manifest import Manifest
from data_ops.paths import dataset_dir

BUCKET = "mevadata-public-01"
PREFIX = "drops-123-r13/"
REPO_URL = "https://gitlab.kitware.com/meva/meva-data-repo.git"
REPO_COMMIT = "421841a75577b697c314e952e585aecbb1b99e17"  # 2021-12-13, pinned
ANNOTATION_SETS = ("kitware", "kitware-meva-training")  # official Kitware annotations
SPARSE = ["/LICENSE", "/README.md", "/metadata/", "/documents/",
          *(f"/annotation/DIVA-phase-2/MEVA/{s}/" for s in ANNOTATION_SETS)]

INDOOR_CAMERAS = frozenset({
    "admin.G326", "admin.G329",  # stairwells/hallway
    "bus.G331",  # waiting room
    "bus.G508",  # covered platform/interior (RGB set 3-508, grey scene)
    "school.G299", "school.G330",  # gym (MEVA docs: indoor)
    "school.G419", "school.G420", "school.G421", "school.G423",  # stairwell, lobby, commons, hallway
})
TARGET_ACTIVITIES = ("person_picks_up_object", "person_puts_down_object",
                     "person_transfers_object", "person_steals_object")
_CLIP_RE = re.compile(
    r"(?P<clip>\d{4}-\d\d-\d\d\.\d\d-\d\d-\d\d\.\d\d-\d\d-\d\d\.(?P<site>\w+)\.(?P<cam>G\d+))\.r13\.avi$")

META = {
    "dataset_id": "meva",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": ('"Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the '
                    "Intelligence Advanced Research Projects Activity (IARPA), CC BY 4.0. https://mevadata.org"),
    "use": "prod",
    "notes": "Not retail; scripted actors. Numbers are a proxy (docs/DATA.md).",
}


def camera_of(key: str) -> tuple[str, str] | None:
    """S3 key -> (clip name, 'site.Gnnn'), or None for non-clip objects."""
    m = _CLIP_RE.search(key)
    return (m["clip"], f"{m['site']}.{m['cam']}") if m else None


def list_bucket() -> list[dict]:
    out = run([tool("aws"), "s3api", "list-objects-v2", "--no-sign-request", "--bucket", BUCKET,
               "--prefix", PREFIX, "--output", "json"]).stdout
    return json.loads(out).get("Contents", [])


def sync_annotations(root: Path) -> Path:
    repo = root / "annotations" / "meva-data-repo"
    git = tool("git")
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run([git, "clone", "--filter=blob:none", "--no-checkout", "--sparse", REPO_URL, str(repo)])
    run([git, "-C", str(repo), "sparse-checkout", "set", "--no-cone", *SPARSE])
    run([git, "-C", str(repo), "fetch", "--depth", "1", "origin", REPO_COMMIT])
    run([git, "-C", str(repo), "checkout", "--detach", REPO_COMMIT])
    return repo


def annotation_index(repo: Path) -> dict[str, Counter]:
    """clip name -> Counter of target activity occurrences (first annotation set wins on duplicates)."""
    idx: dict[str, Counter] = {}
    base = repo / "annotation" / "DIVA-phase-2" / "MEVA"
    for sub in ANNOTATION_SETS:
        for f in sorted((base / sub).rglob("*.activities.yml")):
            clip = f.name[: -len(".activities.yml")]
            if clip in idx:
                continue
            text = f.read_text()
            idx[clip] = Counter({a: text.count(f"'{a}'") for a in TARGET_ACTIVITIES})
    return idx


def select(objects: list[dict], ann: dict[str, Counter], max_bytes: int) -> list[tuple[dict, str, str]]:
    """Indoor clips, annotated first, then unannotated, each oldest-first, up to max_bytes."""
    rows = []
    for o in objects:
        cc = camera_of(o["Key"])
        if cc and cc[1] in INDOOR_CAMERAS:
            rows.append((o, cc[0], cc[1]))
    rows.sort(key=lambda r: (r[1] not in ann, r[1]))
    out, total = [], 0
    for r in rows:
        if total + r[0]["Size"] > max_bytes:
            continue
        out.append(r)
        total += r[0]["Size"]
    return out


def _etag_checksum(etag: str) -> str | None:
    e = etag.strip('"')
    return None if "-" in e else f"md5:{e}"  # multipart ETags aren't MD5s


def fetch(max_gb: float, jobs: int = 8, dry_run: bool = False, log=print) -> int:
    root = dataset_dir("meva")
    log(f"meva: annotations -> {root / 'annotations'}")
    repo = sync_annotations(root)
    ann = annotation_index(repo)
    objs = list_bucket()
    picked = select(objs, ann, int(max_gb * GB))
    n_ann = sum(1 for _, c, _ in picked if c in ann)
    events = sum((ann[c] for _, c, _ in picked if c in ann), Counter())
    log(f"meva: {len(picked)} indoor clips selected ({n_ann} annotated), "
        f"{sum(o['Size'] for o, _, _ in picked) / GB:,.1f} GB; target events {dict(events)}")
    idx = root / "index" / "clips.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    with idx.open("w") as f:
        for o, clip, cam in picked:
            f.write(json.dumps({"clip": clip, "camera": cam, "key": o["Key"], "bytes": o["Size"],
                                "annotated": clip in ann, "events": dict(ann.get(clip, {}))}) + "\n")
    if dry_run:
        return 0
    files = [RemoteFile(relpath=o["Key"][len(PREFIX):], source=f"s3://{BUCKET}/{o['Key']}",
                        nbytes=o["Size"], source_checksum=_etag_checksum(o["ETag"])) for o, _, _ in picked]
    man_path = root / "MANIFEST.json"
    man = Manifest.load_or_new(man_path, **META)
    man.notes = (f"{META['notes']} Annotations: {REPO_URL} @ {REPO_COMMIT} "
                 f"(sets: {', '.join(ANNOTATION_SETS)}). Indoor cameras: {sorted(INDOOR_CAMERAS)}.")
    aws = tool("aws")

    def download(rf: RemoteFile, dest: Path) -> None:
        run([aws, "s3", "cp", "--no-sign-request", "--only-show-errors", rf.source, str(dest)])

    stats = fetch_all(files, root / "raw", man, man_path, download, jobs=jobs, log=log)
    log(f"meva: done {stats}; manifest {man_path} ({man.total_bytes() / GB:,.1f} GB recorded)")
    return 1 if stats["failed"] else 0
