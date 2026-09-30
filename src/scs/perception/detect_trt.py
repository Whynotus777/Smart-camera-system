"""TensorRT FP16 engines with a dynamic batch dimension, built from ONNX and run on torch tensors.

Why: both detectors deploy the same way (ARCHITECTURE D8), so the build recipe and the
runtime live in one place. Engines are per-GPU and never committed (AGENTS.md rule 4);
`build_engine` writes a JSON recipe next to the engine (TRT/CUDA versions, profile,
ONNX sha256, engine sha256) which is what goes into `models/MANIFEST.yaml`.

I/O stays on the GPU: input/output buffers are torch tensors whose pointers are handed
to `execute_async_v3` on the current torch stream, so there's no pycuda dependency and
no host round-trip.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class BatchProfile:
    min_batch: int = 1
    opt_batch: int = 8
    max_batch: int = 16
    height: int = 640
    width: int = 640


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    profile: BatchProfile,
    precision: str = "fp16",
    workspace_gb: float = 4.0,
    extra: dict[str, Any] | None = None,
    fp32_layers: Callable[[Any], bool] | None = None,
    fp32_rule: str | None = None,
) -> dict[str, Any]:
    """ONNX (dynamic batch on input 0) → serialized TRT engine + ``<engine>.recipe.json``.

    `precision` is "fp16", "bf16" or "fp32". `fp32_layers(layer) -> bool` pins matching
    layers to FP32 inside a 16-bit engine (for layers that overflow or lose too much
    precision); `fp32_rule` describes the rule in the recipe.
    """
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"precision must be fp16, bf16 or fp32, got {precision!r}")
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    pinned = 0
    if precision != "fp32" and fp32_layers is not None:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if fp32_layers(layer):
                layer.precision = trt.float32
                for k in range(layer.num_outputs):
                    if layer.get_output(k).dtype in (trt.float16, trt.bfloat16, trt.float32):
                        layer.set_output_type(k, trt.float32)
                pinned += 1
    inp = network.get_input(0)
    opt = builder.create_optimization_profile()
    c, h, w = 3, profile.height, profile.width
    opt.set_shape(
        inp.name, (profile.min_batch, c, h, w), (profile.opt_batch, c, h, w), (profile.max_batch, c, h, w)
    )
    config.add_optimization_profile(opt)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(blob))
    recipe = {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(Path(onnx_path)),
        "engine_sha256": sha256(engine_path),
        "precision": precision,
        "fp32_pinned_layers": pinned,
        "fp32_rule": fp32_rule,
        "profile": profile.__dict__,
        "workspace_gb": workspace_gb,
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "sm": "sm_{}{}".format(*torch.cuda.get_device_capability(0)),
        "build_s": round(time.time() - t0, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(extra or {}),
    }
    engine_path.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=2))
    return recipe


class TrtRunner:
    """Runs a dynamic-batch engine. `__call__(x)` with x (B,3,H,W) → dict of output tensors."""

    def __init__(self, engine_path: Path, device: str = "cuda") -> None:
        import tensorrt as trt
        import torch

        self._trt, self._torch = trt, torch
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load engine {engine_path} (built for another GPU/TRT?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(self.inputs) != 1:
            raise ValueError(f"expected one input tensor, got {self.inputs}")
        self.input_dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.inputs[0]))
        self.max_batch = int(self.engine.get_tensor_profile_shape(self.inputs[0], 0)[2][0])
        # TRT inserts extra synchronizations on the legacy default stream; use our own.
        self.stream = torch.cuda.Stream(self.device)

    def _torch_dtype(self, dt: Any) -> Any:
        trt, torch = self._trt, self._torch
        return {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }[dt]

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        torch = self._torch
        if x.shape[0] > self.max_batch:  # split rather than fail; the batcher normally prevents this
            parts = [self(x[i : i + self.max_batch]) for i in range(0, x.shape[0], self.max_batch)]
            return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        x = x.to(self.device, self.input_dtype).contiguous()
        name = self.inputs[0]
        self.context.set_input_shape(name, tuple(x.shape))
        self.context.set_tensor_address(name, x.data_ptr())
        out: dict[str, Tensor] = {}
        for o in self.outputs:
            shape = tuple(self.context.get_tensor_shape(o))
            out[o] = torch.empty(
                shape, dtype=self._torch_dtype(self.engine.get_tensor_dtype(o)), device=self.device
            )
            self.context.set_tensor_address(o, out[o].data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(self.device))  # x was produced there
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return out


def env_record() -> dict[str, Any]:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603
        except OSError:
            return ""

    return {
        "git_sha": sh("git", "rev-parse", "HEAD"),
        "gpu": sh(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
