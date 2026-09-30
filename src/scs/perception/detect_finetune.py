"""Fine-tune the person detectors on overhead retail (SmartSpaces) + real indoor CCTV (MEVA).

Why: COCO-pretrained detectors are trained mostly on eye-level photos; ceiling cameras
at 2.4–3 m see foreshortened heads and shoulders, which is the domain gap T03 must close.

Data (all CC-BY-4.0, splits by scene/camera from track_data.py, never by frame):
- SmartSpaces train scenes: every labelled person, 1 frame / `every_s` per camera.
- MEVA train cameras: only activity participants are labelled, and training on that
  would teach the model that bystanders are background. So unlabelled people are
  **pseudo-labelled** by the COCO-pretrained Apache-2.0 teacher (D-FINE-S) at
  score >= `teacher_thresh` where they don't overlap a GT box. Pseudo-labels are used
  for training only, never as test truth (docs/DATA.md `label_source: model`).
Lineage: SmartSpaces + MEVA are `prod` in docs/DATA.md; YOLO11 fine-tunes stay AGPL/R&D
because the base weights are; D-FINE fine-tunes stay Apache-2.0/prod-candidate.

Output is YOLO format under data/t03_det/ (gitignored). Usage:
  python -m scs.perception.detect_finetune build --teacher-engine models/dfine-s/dfine_bf16_b16.engine
  scripts/gpu exclusive -- python -m scs.perception.detect_finetune train-yolo --epochs 30
  scripts/gpu exclusive -- python -m scs.perception.detect_finetune train-dfine --epochs 12
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from scs.contracts import FrameRef
from scs.perception.detect import iou_matrix
from scs.perception.track_data import (
    MEVA_SPLITS,
    SMARTSPACES_SPLITS,
    GtClip,
    data_root,
    load_meva,
    load_smartspaces,
    meva_person_boxes,
    smartspaces_cameras,
)
from scs.perception.track_eval import read_frames


def ds_root() -> Path:
    return data_root() / "t03_det"


def _ss_clips(split: str, minutes: float) -> Iterator[GtClip]:
    for scene in SMARTSPACES_SPLITS[split]:
        d = data_root() / "smartspaces" / "raw" / "MTMC_Tracking_2024" / "test" / scene
        if (d / "ground_truth.txt").exists():
            for cam in smartspaces_cameras(scene):
                yield load_smartspaces(scene, cam, 0.0, minutes * 60)


def _meva_train_clips(split: str, per_camera: int) -> Iterator[GtClip]:
    root = data_root() / "meva" / "raw"
    for cam in MEVA_SPLITS[split]:
        clips = sorted(p.name.replace(".r13.avi", "") for p in root.glob(f"*/*/*.{cam}.r13.avi"))
        scored = sorted(
            ((sum(len(v[0]) for v in meva_person_boxes(c).values()), c) for c in clips), reverse=True
        )
        for n, c in scored[:per_camera]:
            if n > 0:
                yield load_meva(c)


def build(args: argparse.Namespace) -> None:
    import cv2

    teacher = None
    if args.teacher_engine:
        from scs.perception.detect import DetectorConfig
        from scs.perception.detect_dfine import DFineDetector

        teacher = DFineDetector(args.teacher_engine, DetectorConfig(resize_mode="stretch", score_thresh=0.3))
    stats: dict[str, Any] = {"images": {}, "boxes_gt": 0, "boxes_pseudo": 0}
    for split in ("train", "val"):
        clips: list[GtClip] = []
        if "smartspaces" in args.datasets:
            clips += list(_ss_clips(split, args.ss_minutes))
        if "meva" in args.datasets:
            clips += list(_meva_train_clips(split, args.meva_per_camera))
        (ds_root() / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_root() / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for clip in clips:
            step = int(args.every_s * clip.fps)
            idxs = [
                i
                for i in range(clip.start, clip.end, step)
                if clip.dataset != "meva" or i in clip.frames or not args.meva_labelled_only
            ]
            for idx, img in read_frames(clip.video, idxs):
                _, boxes = clip.gt(idx)
                pseudo = np.zeros((0, 4))
                if clip.partial and teacher is not None:
                    ref = FrameRef(
                        camera_id=clip.camera_id,
                        epoch=0,
                        seq=0,
                        frame_idx=idx,
                        ts=0.0,
                        width=clip.width,
                        height=clip.height,
                    )
                    t = np.array(
                        [d.bbox for d in teacher.detect([(ref, img)])[0] if d.score >= args.teacher_thresh]
                    )
                    t = t.reshape(-1, 4)
                    if len(t) and len(boxes):
                        t = t[iou_matrix(t, boxes).max(1) < 0.3]
                    pseudo = t
                elif clip.partial:
                    continue  # never train on partially-labelled frames without pseudo-labels
                allb = np.r_[boxes.reshape(-1, 4), pseudo]
                name = f"{clip.dataset}_{clip.clip_id.replace('/', '_')}_{idx:06d}"
                cv2.imwrite(
                    str(ds_root() / "images" / split / f"{name}.jpg"),
                    img[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                w, h = clip.width, clip.height
                allb[:, [0, 2]] = allb[:, [0, 2]].clip(0, w)
                allb[:, [1, 3]] = allb[:, [1, 3]].clip(0, h)
                allb = allb[(allb[:, 2] - allb[:, 0] > 2) & (allb[:, 3] - allb[:, 1] > 2)]
                lines = [
                    f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} "
                    f"{(b[3] - b[1]) / h:.6f}"
                    for b in allb
                ]
                (ds_root() / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                stats["boxes_gt"] += len(boxes)
                stats["boxes_pseudo"] += len(pseudo)
                n += 1
            print(f"{split} {clip.clip_id}: total {n} images", flush=True)
        stats["images"][split] = n
    yaml = ds_root() / "t03_person.yaml"
    yaml.write_text(f"path: {ds_root()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    stats["args"] = {k: str(v) for k, v in vars(args).items() if k != "func"}
    (ds_root() / f"build_stats_{'_'.join(args.datasets)}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def train_yolo(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(ds_root() / "t03_person.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        single_cls=True,
        project=str(Path("runs/t03/ft").resolve()),
        name=args.name,
        exist_ok=True,
        workers=8,
        seed=0,
        deterministic=False,
        cos_lr=True,
        close_mosaic=5,
        plots=False,
    )


class _YoloDir:
    """YOLO-format images/labels → (3x640x640 float tensor, HF D-FINE label dict). Picklable for workers."""

    def __init__(self, split: str, flip: bool = True) -> None:
        self.imgs = sorted((ds_root() / "images" / split).glob("*.jpg"))
        self.labels = ds_root() / "labels" / split
        self.flip = flip

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Any, dict[str, Any]]:
        import cv2
        import torch

        p = self.imgs[i]
        raw = cv2.imread(str(p))
        if raw is None:
            raise FileNotFoundError(p)
        im = cv2.resize(raw[:, :, ::-1], (640, 640), interpolation=cv2.INTER_LINEAR)
        lab = (self.labels / f"{p.stem}.txt").read_text().split("\n")
        b = np.array([list(map(float, ln.split()[1:])) for ln in lab if ln.strip()]).reshape(-1, 4)
        x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255
        if self.flip and np.random.rand() < 0.5:  # horizontal flip
            x = x.flip(-1)
            b[:, 0] = 1 - b[:, 0]
        return x, {"class_labels": torch.zeros(len(b), dtype=torch.long), "boxes": torch.tensor(b).float()}


def _collate(items: list[tuple[Any, dict[str, Any]]]) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    return torch.stack([x for x, _ in items]), [y for _, y in items]


def train_dfine(args: argparse.Namespace) -> None:
    """Plain PyTorch loop over the YOLO-format set; keeps the COCO head, trains label 0 (person)."""
    import time

    import torch
    from transformers import DFineForObjectDetection

    from scs.perception.detect_dfine import HF_ID

    torch.manual_seed(0)
    dev = torch.device("cuda")
    model = DFineForObjectDetection.from_pretrained(args.hf_id or HF_ID).to(dev).train()  # type: ignore[arg-type]
    data = torch.utils.data.DataLoader(
        _YoloDir("train"),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=_collate,
        persistent_workers=True,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.lr * 0.1},
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    steps = args.epochs * len(data)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * 0.1, args.lr], total_steps=steps, pct_start=0.05
    )
    out = Path("runs/t03/ft") / args.name
    out.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        for xs, ys in data:
            with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16: no loss scaling needed
                o = model(
                    pixel_values=xs.to(dev, non_blocking=True),
                    labels=[{k: v.to(dev) for k, v in y.items()} for y in ys],
                )
            opt.zero_grad(set_to_none=True)
            o.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == args.max_steps:
                print(
                    f"epoch {ep} step {step}/{steps} loss {o.loss.item():.3f} "
                    f"{step / (time.time() - t0):.2f} it/s",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                break
        model.save_pretrained(out / "last")
        if args.max_steps and step >= args.max_steps:
            break
    print(f"saved {out / 'last'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--teacher-engine", default=None)
    b.add_argument("--teacher-thresh", type=float, default=0.5)
    b.add_argument("--every-s", type=float, default=4.0)
    b.add_argument("--ss-minutes", type=float, default=5.0)
    b.add_argument("--meva-per-camera", type=int, default=6)
    b.add_argument("--meva-labelled-only", action="store_true", default=True)
    b.add_argument("--datasets", nargs="+", default=["smartspaces", "meva"], choices=["smartspaces", "meva"])
    b.add_argument("--store-width", type=int, default=960)
    b.set_defaults(func=build)
    y = sub.add_parser("train-yolo")
    y.add_argument("--weights", default="models/yolo11s/yolo11s.pt")
    y.add_argument("--epochs", type=int, default=30)
    y.add_argument("--batch", type=int, default=32)
    y.add_argument("--name", default="yolo11s_retail")
    y.set_defaults(func=train_yolo)
    d = sub.add_parser("train-dfine")
    d.add_argument("--hf-id", default=None)
    d.add_argument("--epochs", type=int, default=12)
    d.add_argument("--batch", type=int, default=16)
    d.add_argument("--lr", type=float, default=1e-4)
    d.add_argument("--name", default="dfine_s_retail")
    d.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    d.add_argument("--workers", type=int, default=12)
    d.set_defaults(func=train_dfine)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
