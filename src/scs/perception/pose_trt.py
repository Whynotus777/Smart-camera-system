"""TensorRT engines for pose backends: build from ONNX with a fixed max batch, run on torch tensors.

Engines are built per GPU from a recorded recipe and never committed (AGENTS.md rule 4).
`build_engine` returns that recipe (TRT version, GPU, precision, profile, ONNX sha256,
engine sha256) for `models/MANIFEST.yaml`. The optimization profile is
min=1 / opt=max=`max_batch`: the fixed max batch is where kernels are tuned, and
smaller batches (a quiet store) still run without padding.

I/O stays on the GPU: inputs are the torch crops from `pose.crop_batch`, outputs are
torch tensors on the same stream, so nothing round-trips through host memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

_DTYPES: dict[Any, torch.dtype] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    input_chw: tuple[int, int, int],
    max_batch: int,
    fp16: bool = True,
    workspace_gb: float = 4.0,
) -> dict[str, Any]:
    """ONNX -> serialized TRT engine with a dynamic batch profile [1, max_batch]. Returns the recipe."""
    import tensorrt as trt

    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, (1, *input_chw), (max_batch, *input_chw), (max_batch, *input_chw))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(blob))
    recipe = {
        "onnx": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "engine_sha256": _sha256(engine_path),
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "sm": "sm_{}{}".format(*torch.cuda.get_device_capability()),
        "precision": "fp16" if fp16 else "fp32",
        "profile": {"input": name, "min": [1, *input_chw], "opt_max": [max_batch, *input_chw]},
    }
    engine_path.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=2))
    return recipe


class TRTEngine:
    """Minimal runner: one input, N outputs, torch tensors on the current CUDA stream."""

    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load engine {engine_path} (built for another GPU/TRT version?)")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = self.engine.get_tensor_mode
        self.input = next(n for n in names if mode(n) == trt.TensorIOMode.INPUT)
        self.outputs = [n for n in names if mode(n) == trt.TensorIOMode.OUTPUT]
        self.max_batch = int(self.engine.get_tensor_profile_shape(self.input, 0)[2][0])
        self.input_chw = tuple(int(d) for d in self.engine.get_tensor_profile_shape(self.input, 0)[2][1:])

    def _torch_dtype(self, name: str) -> torch.dtype:
        trt = self._trt
        return {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
        }[self.engine.get_tensor_dtype(name)]

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[0] > self.max_batch:
            raise ValueError(f"batch {x.shape[0]} > engine max {self.max_batch}")
        x = x.contiguous().to(self._torch_dtype(self.input))
        self.ctx.set_input_shape(self.input, tuple(x.shape))
        self.ctx.set_tensor_address(self.input, x.data_ptr())
        outs = {}
        for n in self.outputs:
            t = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=self._torch_dtype(n), device=x.device)
            self.ctx.set_tensor_address(n, t.data_ptr())
            outs[n] = t
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return outs
