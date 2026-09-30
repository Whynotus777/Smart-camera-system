"""D-FINE person detector (Apache-2.0 code and COCO weights), TensorRT FP16. Production candidate.

Why D-FINE: it's the Apache-2.0 option in docs/DATA.md with the best accuracy/latency
among real-time DETRs, it's NMS-free (fixed 300 queries, so latency doesn't grow with
crowd size), and it runs through Hugging Face `transformers` (Apache-2.0), so export
needs no compiled extensions. That matters on sm_120, where mmcv-based RTMDet needs a
source build.

Export wraps the HF model so the ONNX graph already contains sigmoid + cxcywh→xyxy +
person-class selection; the engine returns (B, Q) scores and (B, Q, 4) input-pixel boxes.
Weights: `ustc-community/dfine-small-coco` (COCO-only; the *obj2coco* variants add
Objects365, whose terms we haven't cleared, so we don't use them).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scs.perception.detect import BatchedDetector, DetectorConfig
from scs.perception.detect_trt import BatchProfile, TrtRunner, build_engine

if TYPE_CHECKING:
    from torch import Tensor

MODELS = Path("models")
HF_ID = "ustc-community/dfine-small-coco"
HF_REVISION = "f79e65b5fbb33ceb9d3ebba042955d7410c608f8"  # pinned; bump deliberately
PERSON = 0  # COCO label id 0 in the HF id2label


def _export_module(model: Any, size: int, person: int) -> Any:
    import torch

    class Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = model

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            o = self.m(pixel_values=pixel_values)
            scores = o.logits[..., person].sigmoid()
            cx, cy, w, h = o.pred_boxes.unbind(-1)
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * size
            return scores, boxes

    return Wrapped().eval()


def _fp16_safe_anchor_sentinels(onnx_path: Path, value: float = 100.0) -> int:
    """Replace float32-max scalar constants feeding `Where` with `value`; returns how many.

    HF D-FINE masks invalid anchors with ``where(valid, logit(anchor), float32.max)``. In
    FP16 TensorRT clips that to 65504 and the decoder arithmetic then overflows to NaN (every
    score comes out NaN). Anchors are logits, so 100 is equivalent after the sigmoid
    (sigmoid(100) == 1.0 in fp32) and stays finite in fp16.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    users: dict[str, list[str]] = {}
    for n in model.graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n.op_type)
    fixed = 0
    for n in model.graph.node:
        if n.op_type != "Constant" or "Where" not in users.get(n.output[0], []):
            continue
        v = numpy_helper.to_array(n.attribute[0].t)
        if v.dtype.kind == "f" and v.size == 1 and abs(float(v.ravel()[0])) > 65504:
            new = np.full(v.shape, np.sign(v.ravel()[0]) * value, dtype=v.dtype)
            n.attribute[0].t.CopyFrom(numpy_helper.from_array(new, n.attribute[0].t.name))
            fixed += 1
    onnx.save(model, str(onnx_path))
    return fixed


FP32_RULE = "ELEMENTWISE layers in /m/model/decoder/ and the output head (names not under /m/)"


def _fp32_decoder_elementwise(layer: Any) -> bool:
    """D-FINE's decoder refines unbounded box distributions; its add/sub/div overflow in FP16 → NaN.

    Found by bisection over FP32-pinned layer sets (T03 report). Backbone and encoder stay FP16.
    """
    import tensorrt as trt

    return layer.type == trt.LayerType.ELEMENTWISE and (
        "/m/model/decoder/" in layer.name or not layer.name.startswith("/m/")
    )


@contextmanager
def _traceable_len() -> Iterator[None]:
    """Make HF D-FINE's ``batch_size = len(source_flatten)`` survive ONNX tracing.

    The TorchScript tracer turns ``len(tensor)`` into a Python constant (the dummy batch
    size), so the exported graph silently mixes up images at any other batch size (we saw
    it: correct at batch 2, wrong at 8/16). ``tensor.shape[0]`` is traced symbolically, so
    for the duration of the export `len` in that module returns it for tensors.
    """
    import builtins

    import torch
    from transformers.models.d_fine import modeling_d_fine as mdf

    def _len(x: Any) -> Any:
        return x.shape[0] if isinstance(x, torch.Tensor) else builtins.len(x)

    mdf.len = _len  # type: ignore[attr-defined]  # module global shadows the builtin
    try:
        yield
    finally:
        del mdf.len  # type: ignore[attr-defined]


def export_dfine(
    hf_id: str = HF_ID,
    out_dir: Path = MODELS / "dfine-s",
    revision: str | None = None,
    profile: BatchProfile | None = None,
    person: int = PERSON,
) -> Path:
    """HF checkpoint (or a local fine-tuned dir) → ONNX (dynamic batch) → TRT FP16 engine."""
    import torch
    from transformers import DFineForObjectDetection

    profile = profile or BatchProfile()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DFineForObjectDetection.from_pretrained(hf_id, revision=revision or "main").eval()
    wrapped = _export_module(model, profile.height, person)
    onnx = out_dir / "dfine.onnx"
    dummy = torch.rand(2, 3, profile.height, profile.width)
    with _traceable_len():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(onnx),
            input_names=["pixel_values"],
            output_names=["scores", "boxes"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={"pixel_values": {0: "batch"}, "scores": {0: "batch"}, "boxes": {0: "batch"}},
        )
    fixed = _fp16_safe_anchor_sentinels(onnx)
    # BF16, not FP16: on TRT 10.16 / sm_120 the FP16 (and even FP32/TF32) dynamic-batch engines
    # returned wrong results at batch > 1; BF16 + FP32 decoder arithmetic matched PyTorch at every
    # batch size (T03 report, "D-FINE engine numerics"). Same tensor-core speed as FP16 on Blackwell.
    engine = out_dir / f"dfine_bf16_b{profile.max_batch}.engine"
    build_engine(
        onnx,
        engine,
        profile,
        fp32_layers=_fp32_decoder_elementwise,
        fp32_rule=FP32_RULE,
        precision="bf16",
        extra={
            "model": "d-fine",
            "weights": hf_id,
            "revision": revision,
            "license": "Apache-2.0",
            "use": "prod candidate",
            "fp16_sentinel_fixes": fixed,
        },
    )
    return engine


class DFineDetector(BatchedDetector):
    def __init__(
        self,
        engine: str | Path = MODELS / "dfine-s" / "dfine_bf16_b16.engine",
        cfg: DetectorConfig | None = None,
    ) -> None:
        # D-FINE is trained on stretched 640x640 inputs, not letterboxed ones.
        super().__init__(cfg or DetectorConfig(resize_mode="stretch"))
        self.runner = TrtRunner(Path(engine), self.cfg.device)
        self.model_id = f"dfine:{Path(engine).parent.name}/{Path(engine).stem}"

    def _infer(self, x: Tensor) -> list[np.ndarray]:
        import torch

        out = self.runner(x)
        scores, boxes = out["scores"].float(), out["boxes"].float()
        res = torch.cat([boxes, scores[..., None]], -1).cpu().numpy()
        return [r[r[:, 4] >= self.cfg.score_thresh] for r in res]
