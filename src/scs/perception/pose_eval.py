"""T04 accuracy eval: wrist PCK@0.2 (torso-normalized), full-res vs sub-stream crops.

Metric (stated in every report): a predicted keypoint is correct when
||pred - gt|| <= 0.2 x torso diameter, torso diameter = ||left_shoulder - right_hip||
of the GROUND TRUTH (COCO kp 5 and 12). Instances without both torso points labeled
are excluded (counted). Wrists = COCO kp 9 and 10, pooled; a wrist counts only if
labeled (v >= 1); results are also split into visible (v=2) and occluded (v=1).

Protocol (COCO-keypoints val2017, person-crop / GT-box protocol): every non-crowd person
with >= 1 labeled keypoint is cropped from its GT box through the SAME
`TopDownPoseEstimator` path used live (crop -> TRT -> map back to frame pixels).

Main vs sub stream (ARCHITECTURE D1): the substitute for simultaneous main+sub footage
is to downscale the whole image by `factor` (INTER_AREA, 4 = 2560x1440 -> 640x360, 3 =
1920x1080 -> 640x360), crop the scaled box from the small image, and scale keypoints
back up. Same model, same box, fewer pixels. Results are binned by the person's box
height in the full-res image, because the sub-stream penalty depends on it.

Ground truth is human annotation only. Agreement with another model is never reported
as accuracy. Bootstrap CIs resample images.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scs.contracts import FrameRef, Track

L_SHO, R_HIP, WRISTS = 5, 12, (9, 10)
PCK_ALPHA = 0.2
HEIGHT_BINS = (0, 64, 128, 256, 100_000)


def torso_diameter(gt: np.ndarray) -> float | None:
    """GT (17, 3) COCO kps -> ||l_shoulder - r_hip|| or None if either is unlabeled."""
    if gt[L_SHO, 2] <= 0 or gt[R_HIP, 2] <= 0:
        return None
    d = float(np.hypot(*(gt[L_SHO, :2] - gt[R_HIP, :2])))
    return d if d > 1.0 else None


def pck_hits(pred: np.ndarray, gt: np.ndarray, alpha: float = PCK_ALPHA) -> np.ndarray | None:
    """(17,) array: 1 hit, 0 miss, -1 unlabeled. None if torso is unavailable."""
    d = torso_diameter(gt)
    if d is None:
        return None
    err = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
    return np.where(gt[:, 2] > 0, (err <= alpha * d).astype(int), -1)


def _bin(h: float) -> str:
    for lo, hi in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:], strict=True):
        if lo <= h < hi:
            return f"{lo}-{hi}" if hi < 100_000 else f">={lo}"
    return "?"


def summarize(rows: list[dict[str, Any]], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """rows: one per instance {img, h, vis (17,), hits (17,)} -> PCK summaries with image-bootstrap CIs."""

    def stat(sel_rows: list[dict[str, Any]], kps: tuple[int, ...], vis: tuple[int, ...]) -> dict[str, Any]:
        per_img: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for r in sel_rows:
            for j in kps:
                if r["hits"][j] >= 0 and r["vis"][j] in vis:
                    per_img[r["img"]][0] += r["hits"][j]
                    per_img[r["img"]][1] += 1
        if not per_img:
            return {"pck": None, "n": 0}
        arr = np.array(list(per_img.values()), dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(arr), (n_boot, len(arr)))
        boots = arr[idx, 0].sum(1) / np.maximum(arr[idx, 1].sum(1), 1)
        return {
            "pck": float(arr[:, 0].sum() / arr[:, 1].sum()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(arr[:, 1].sum()),
            "n_images": len(arr),
        }

    allk = tuple(range(17))
    out: dict[str, Any] = {
        "wrist": stat(rows, WRISTS, (1, 2)),
        "wrist_visible": stat(rows, WRISTS, (2,)),
        "wrist_occluded": stat(rows, WRISTS, (1,)),
        "all_kp": stat(rows, allk, (1, 2)),
        "n_instances": len(rows),
        "by_height": {},
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_bin(r["h"])].append(r)
    for k in sorted(groups, key=lambda s: int(s.split("-")[0].lstrip(">="))):
        out["by_height"][k] = {"wrist": stat(groups[k], WRISTS, (1, 2)), "n_instances": len(groups[k])}
    return out


def load_coco(ann_path: Path, limit: int | None = None) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    d = json.loads(ann_path.read_text())
    imgs = {im["id"]: im for im in d["images"]}
    anns: dict[int, list[dict]] = defaultdict(list)
    for a in d["annotations"]:
        if a["iscrowd"] == 0 and a["num_keypoints"] > 0 and a["bbox"][2] > 1 and a["bbox"][3] > 1:
            anns[a["image_id"]].append(a)
    ids = sorted(anns)[:limit] if limit else sorted(anns)
    return {i: imgs[i] for i in ids}, {i: anns[i] for i in ids}


def run_coco(
    estimator: Any,
    img_dir: Path,
    ann_path: Path,
    factors: tuple[int, ...] = (1, 3, 4),
    limit: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate `estimator` (a TopDownPoseEstimator) on COCO val at each downscale factor."""
    import cv2
    import torch

    imgs, anns = load_coco(ann_path, limit)
    rows: dict[int, list[dict[str, Any]]] = {f: [] for f in factors}
    no_torso = 0
    t0 = time.time()
    for n, (iid, im) in enumerate(imgs.items()):
        bgr = cv2.imread(str(img_dir / im["file_name"]), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        gts = [np.array(a["keypoints"], dtype=float).reshape(17, 3) for a in anns[iid]]
        for f in factors:
            small = (
                bgr if f == 1 else cv2.resize(bgr, (round(W / f), round(H / f)), interpolation=cv2.INTER_AREA)
            )
            sh, sw = small.shape[:2]
            sx, sy = W / sw, H / sh
            ref = FrameRef(camera_id="coco", epoch=0, seq=iid, frame_idx=iid, ts=0.0, width=sw, height=sh)
            tracks = []
            for k, a in enumerate(anns[iid]):
                x, y, w, h = a["bbox"]
                tracks.append(
                    Track(frame=ref, track_id=k, bbox=(x / sx, y / sy, (x + w) / sx, (y + h) / sy), score=1.0)
                )
            poses = estimator.estimate(ref, torch.from_numpy(small).to(device), tracks)
            by_id = {p.track_id: p for p in poses}
            for k, (a, gt) in enumerate(zip(anns[iid], gts, strict=True)):
                p = by_id.get(k)
                pred = np.array(p.keypoints) if p else np.full((17, 3), -1e9)
                pred[:, 0] *= sx
                pred[:, 1] *= sy
                hits = pck_hits(pred, gt)
                if hits is None:
                    no_torso += f == 1
                    continue
                rows[f].append(
                    {
                        "img": iid,
                        "h": a["bbox"][3],
                        "vis": gt[:, 2].astype(int).tolist(),
                        "hits": hits.tolist(),
                    }
                )
        if n % 500 == 0:
            print(f"  {n}/{len(imgs)} images, {time.time() - t0:.0f}s", flush=True)
    return {
        "dataset": "coco_kp val2017 (person_keypoints_val2017.json), GT boxes",
        "metric": "PCK@0.2, threshold = 0.2 x GT torso diameter (||l_shoulder - r_hip||)",
        "n_images": len(imgs),
        "excluded_no_torso": no_torso,
        "results": {f"x1/{f}" if f > 1 else "full": summarize(rows[f]) for f in factors},
        "rows": {str(f): rows[f] for f in factors},
    }


def smartspaces_heights(gt_files: list[Path], frame_stride: int = 30) -> np.ndarray:
    """Person box heights (px, native 1080p) from SmartSpaces MTMC `ground_truth.txt` (GT, not a model).

    Columns: camera_id obj_id frame x y w h X Y. Every `frame_stride`-th frame is kept to
    limit near-duplicate boxes. Boxes cut by the image border are kept (they're real crops).
    """
    hs = []
    for p in gt_files:
        a = np.loadtxt(p, usecols=(2, 6))
        hs.append(a[a[:, 0] % frame_stride == 0, 1])
    return np.concatenate(hs) if hs else np.zeros(0)


def weighted_by_heights(coco: dict[str, Any], heights_main: np.ndarray, factor: int) -> dict[str, Any]:
    """Expected wrist PCK at a camera's person-size distribution, main vs sub (factor), from
    COCO per-instance hits. Each COCO instance is weighted by how common its height bin is
    in `heights_main` relative to COCO. Proxy: COCO views are not overhead retail views."""
    out: dict[str, Any] = {"factor": factor}
    weights: dict[str, float] = {}
    labels = [_bin(h) for h in heights_main]
    for b in set(labels):
        weights[b] = labels.count(b) / len(labels)
    out["height_bin_share"] = weights
    for cond in ("1", str(factor)):
        num = den = 0.0
        per_bin: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in coco["rows"][cond]:
            for j in WRISTS:
                if r["hits"][j] >= 0:
                    per_bin[_bin(r["h"])][0] += r["hits"][j]
                    per_bin[_bin(r["h"])][1] += 1
        for b, w in weights.items():
            if per_bin[b][1]:
                num += w * per_bin[b][0] / per_bin[b][1]
                den += w
        out["main" if cond == "1" else "sub"] = num / den if den else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=["rtmpose", "yolo"], required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--coco-images", type=Path, required=True)
    ap.add_argument("--coco-ann", type=Path, required=True)
    ap.add_argument("--smartspaces-gt", type=Path, nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from scs.perception.pose import TopDownPoseEstimator

    if args.backend == "rtmpose":
        from scs.perception.pose_rtmpose import RTMPoseTRT

        backend: Any = RTMPoseTRT(args.engine, args.model_id)
    else:
        from scs.perception.pose_yolo import YOLOPoseTRT

        backend = YOLOPoseTRT(args.engine, args.model_id)
    est = TopDownPoseEstimator(backend, stream_check="off", min_box_px=1.0)
    res = run_coco(est, args.coco_images, args.coco_ann, limit=args.limit)
    res["model_id"] = args.model_id
    if args.smartspaces_gt:
        hs = smartspaces_heights(args.smartspaces_gt)
        res["smartspaces"] = {
            "source": "SmartSpaces MTMC_Tracking_2024 GT boxes (synthetic retail/warehouse, 1080p)",
            "n_boxes": int(len(hs)),
            "height_px_percentiles_1080p": {q: float(np.percentile(hs, q)) for q in (10, 25, 50, 75, 90)},
            "1080p_main_vs_360p_sub": weighted_by_heights(res, hs, 3),
            "1440p_main_vs_360p_sub": weighted_by_heights(res, hs * 1440 / 1080, 4),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
