"""T04 throughput/latency bench: 30 / 100 / 300 crops/s through the full pose path.

Run ONLY under `scripts/gpu exclusive -- ...` (AGENTS.md): numbers from a shared GPU
are invalid. The report records `nvidia-smi` at start.

Scenario (ARCHITECTURE §5): 10 cameras, 2560x1440 main-stream frames already on the GPU
(decode is T02's cost, not ours), pose at 10 Hz, so each 100 ms tick carries rate/10
crops spread over the 10 cameras, micro-batched into one backend call
(`estimate_many`, D9). Measured per tick, synchronized: crop (grid_sample) -> TRT ->
SimCC/YOLO decode -> map back -> `Pose` objects. "Per batch" latency = that tick time.
The loop is paced in real time for `--seconds`; "sustained" means every tick finished
before the next one was due. A separate unpaced run measures peak crops/s at max batch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch

from scs.contracts import FrameRef, Track

W, H, N_CAMS, TICK_S = 2560, 1440, 10, 0.1


def _scene(n_crops: int, rng: np.random.Generator, frames: list[torch.Tensor], seq: int) -> list[tuple]:
    items = []
    for cam in range(N_CAMS):
        k = n_crops // N_CAMS + (1 if cam < n_crops % N_CAMS else 0)
        ref = FrameRef(
            camera_id=f"cam{cam}", epoch=0, seq=seq, frame_idx=seq, ts=time.time(), width=W, height=H
        )
        tracks = []
        for t in range(k):
            h = rng.uniform(250, 700)  # person height in main-stream px at 2.4-3 m mounts
            x, y = rng.uniform(0, W - h * 0.4), rng.uniform(0, H - h)
            tracks.append(Track(frame=ref, track_id=t, bbox=(x, y, x + 0.4 * h, y + h), score=0.9))
        items.append((ref, frames[cam], tracks))
    return items


def bench(est: Any, rate: int, seconds: float, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frames = [torch.randint(0, 255, (H, W, 3), dtype=torch.uint8, device="cuda") for _ in range(N_CAMS)]
    n_crops = max(1, round(rate * TICK_S))
    for i in range(20):  # warm-up (engine tactics, allocator)
        est.estimate_many(_scene(n_crops, rng, frames, i))
    torch.cuda.synchronize()
    lat, late, done = [], 0, 0
    t_start = time.perf_counter()
    i = 0
    while (now := time.perf_counter()) - t_start < seconds:
        due = t_start + i * TICK_S
        if now < due:
            time.sleep(due - now)
        items = _scene(n_crops, rng, frames, 1000 + i)
        t0 = time.perf_counter()
        poses = est.estimate_many(items)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lat.append(dt * 1e3)
        done += sum(len(p) for p in poses)
        late += dt > TICK_S
        i += 1
    wall = time.perf_counter() - t_start
    a = np.array(lat)
    return {
        "target_crops_per_s": rate,
        "crops_per_tick": n_crops,
        "ticks": len(a),
        "achieved_crops_per_s": done / wall,
        "latency_ms": {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        },
        "ticks_over_budget_100ms": int(late),
        "sustained": bool(late == 0),
    }


def peak(est: Any, batch: int, iters: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    frames = [torch.randint(0, 255, (H, W, 3), dtype=torch.uint8, device="cuda") for _ in range(N_CAMS)]
    items = _scene(batch, rng, frames, 0)
    for _ in range(20):
        est.estimate_many(items)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        est.estimate_many(items)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"batch": batch, "ms_per_batch": dt / iters * 1e3, "crops_per_s": batch * iters / dt}


def main() -> None:
    ap = argparse.ArgumentParser(description="T04 pose throughput bench (run under scripts/gpu exclusive)")
    ap.add_argument("--backend", choices=["rtmpose", "yolo"], required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--rates", type=int, nargs="+", default=[30, 100, 300])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scs.perception.pose import TopDownPoseEstimator

    if args.backend == "rtmpose":
        from scs.perception.pose_rtmpose import RTMPoseTRT

        backend: Any = RTMPoseTRT(args.engine, args.model_id)
    else:
        from scs.perception.pose_yolo import YOLOPoseTRT

        backend = YOLOPoseTRT(args.engine, args.model_id)
    est = TopDownPoseEstimator(backend)
    smi_bin = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    query = "--query-gpu=name,driver_version,memory.used,utilization.gpu,clocks.sm"
    smi = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [smi_bin, query, "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    res: dict[str, Any] = {
        "model_id": args.model_id,
        "nvidia_smi_at_start": smi,
        "max_batch": backend.max_batch,
        "torch": torch.__version__,
    }
    res["paced"] = [bench(est, r, args.seconds) for r in args.rates]
    res["peak"] = [peak(est, b) for b in (1, 8, 16, backend.max_batch)]
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
