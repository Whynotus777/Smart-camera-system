"""Build the detector TensorRT engines from a recorded recipe (engines are never committed).

Why a CLI: AGENTS.md requires engines to be rebuilt per target GPU from a recipe, with the
recipe/versions/hash in models/MANIFEST.yaml. This is that recipe; each build also writes
`<engine>.recipe.json` with the exact TRT/torch/CUDA versions and sha256 hashes.

  scripts/gpu exclusive -- python -m scs.perception.detect_export dfine
  scripts/gpu exclusive -- python -m scs.perception.detect_export yolo11      # AGPL, R&D only
  scripts/gpu exclusive -- python -m scs.perception.detect_export dfine \
      --weights runs/t03/ft/dfine_s_retail/last --out models/dfine-s-retail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scs.perception.detect_trt import BatchProfile


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detector", choices=["dfine", "yolo11"])
    ap.add_argument("--weights", default=None, help="HF id / local dir (dfine) or .pt (yolo11)")
    ap.add_argument("--revision", default=None, help="HF revision (dfine)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--opt-batch", type=int, default=8)
    a = ap.parse_args(argv)
    profile = BatchProfile(min_batch=1, opt_batch=a.opt_batch, max_batch=a.max_batch)
    if a.detector == "dfine":
        from scs.perception.detect_dfine import HF_ID, HF_REVISION, export_dfine

        engine = export_dfine(
            a.weights or HF_ID,
            a.out or Path("models/dfine-s"),
            a.revision or (None if a.weights else HF_REVISION),
            profile,
        )
    else:
        from scs.perception.detect_yolo import export_yolo11

        engine = export_yolo11(a.weights or "yolo11s.pt", a.out or Path("models/yolo11s"), profile)
    check_finite(engine)
    check_batch_invariance(a.detector, engine)
    print(json.dumps(json.loads(engine.with_suffix(".recipe.json").read_text()), indent=2))


def check_finite(engine: Path) -> None:
    """16-bit engines can silently produce NaN/inf (it happened with D-FINE); refuse to hand one out."""
    import torch

    from scs.perception.detect_trt import TrtRunner

    out = TrtRunner(engine)(torch.rand(4, 3, 640, 640, device="cuda"))
    bad = [k for k, v in out.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"{engine}: non-finite values in outputs {bad}; don't use this engine")


def check_batch_invariance(detector: str, engine: Path, n: int = 8) -> None:
    """Same frames as one batch vs one at a time must give the same people (it didn't for D-FINE
    before `_traceable_len`: the ONNX trace had frozen the batch size)."""
    import cv2
    import numpy as np

    from scs.contracts import FrameRef
    from scs.perception.detect import make_detector

    det = make_detector(detector, engine=engine)
    cap = cv2.VideoCapture("tests/fixtures/video/demo_1.mp4")
    imgs: list[np.ndarray] = []
    while len(imgs) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        imgs.append(np.ascontiguousarray(bgr[:, :, ::-1]))
        for _ in range(9):
            cap.grab()
    refs = [
        FrameRef(camera_id="c", epoch=0, seq=i, frame_idx=i, ts=0.0, width=640, height=360)
        for i in range(len(imgs))
    ]
    batched = det.detect(list(zip(refs, imgs, strict=True)))
    doubled = det.detect(list(zip(refs + refs, imgs + imgs, strict=True)))  # same images twice in one batch
    n = len(imgs)
    for i, (ref, img) in enumerate(zip(refs, imgs, strict=True)):
        single = det.detect([(ref, img)])[0]
        for what, x, y in (
            ("batch vs single", batched[i], single),
            ("duplicates in one batch", doubled[i], doubled[i + n]),
        ):
            if not (_same_people(x, y) and _same_people(y, x)):
                raise RuntimeError(f"{engine}: {what} disagree on frame {i}; don't use this engine")


def _same_people(a: list, b: list, hi: float = 0.5, lo: float = 0.25, min_iou: float = 0.85) -> bool:
    """Every confident box in `a` has a (possibly less confident) match in `b`. Hysteresis keeps
    16-bit score jitter around a single threshold from failing the check."""
    import numpy as np

    from scs.perception.detect import iou_matrix

    x = np.array([d.bbox for d in a if d.score > hi]).reshape(-1, 4)
    y = np.array([d.bbox for d in b if d.score > lo]).reshape(-1, 4)
    return len(x) == 0 or (len(y) > 0 and iou_matrix(x, y).max(1).min() >= min_iou)


if __name__ == "__main__":
    main()
