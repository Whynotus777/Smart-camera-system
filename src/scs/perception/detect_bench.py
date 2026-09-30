"""Detector latency/throughput per batch size, and dead/slow-camera isolation on the real engine.

Why: T03 acceptance needs "batch of 10 × 640 px < 10 ms (TRT FP16)" and the D9 claim
that a dead camera doesn't move other cameras' latency, both measured under
`scripts/gpu exclusive` (numbers taken any other way are invalid; AGENTS.md).

Two latencies per batch size:
- `engine`: TRT execute only, on an already-preprocessed (B,3,640,640) fp32 tensor.
- `e2e`: `Detector.detect()` from main-stream-sized uint8 GPU frames (2560x1440 by
  default, the Reolink 4MP main stream): resize + pad + engine + decode + NMS + mapping.

Usage: scripts/gpu exclusive -- python -m scs.perception.detect_bench \
          --detector dfine --engine models/dfine-s/dfine_bf16_b16.engine --out runs/t03/bench
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from scs.contracts import FrameRef
from scs.perception.detect import make_detector
from scs.perception.detect_batching import MicroBatcher
from scs.perception.detect_trt import env_record


def _stats(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=100)
    return {
        "mean_ms": 1e3 * statistics.fmean(xs),
        "p50_ms": 1e3 * q[49],
        "p95_ms": 1e3 * q[94],
        "p99_ms": 1e3 * q[98],
        "n": len(xs),
    }


def bench_batches(det: Any, sizes: list[int], iters: int, main_wh: tuple[int, int]) -> dict[str, Any]:
    import torch

    dev = torch.device("cuda")
    w, h = main_wh
    frames = [torch.randint(0, 255, (h, w, 3), dtype=torch.uint8, device=dev) for _ in range(max(sizes))]
    refs = [
        FrameRef(camera_id=f"cam{i}", epoch=0, seq=0, frame_idx=0, ts=0.0, width=w, height=h)
        for i in range(max(sizes))
    ]
    out: dict[str, Any] = {}
    for b in sizes:
        x = torch.rand(b, 3, 640, 640, device=dev)
        batch = list(zip(refs[:b], frames[:b], strict=True))
        for _ in range(max(20, iters // 10)):  # warm-up: TRT picks kernels for this shape lazily
            det.runner(x)
            det.detect(batch)
        torch.cuda.synchronize()
        eng, e2e = [], []
        for _ in range(iters):
            t0 = time.perf_counter()
            det.runner(x)  # TrtRunner synchronizes its stream
            eng.append(time.perf_counter() - t0)
        for _ in range(iters):
            t0 = time.perf_counter()
            det.detect(batch)
            e2e.append(time.perf_counter() - t0)
        se, s2 = _stats(eng), _stats(e2e)
        out[str(b)] = {
            "engine": se,
            "e2e": s2,
            "engine_img_per_s": b / (se["mean_ms"] / 1e3),
            "e2e_img_per_s": b / (s2["mean_ms"] / 1e3),
        }
        print(
            f"batch {b:>2}: engine p50 {se['p50_ms']:.2f} ms p95 {se['p95_ms']:.2f} | "
            f"e2e p50 {s2['p50_ms']:.2f} ms p95 {s2['p95_ms']:.2f} | "
            f"{out[str(b)]['e2e_img_per_s']:.0f} img/s",
            flush=True,
        )
    return out


def isolation(
    det: Any,
    n_cams: int = 10,
    fps: float = 10.0,
    seconds: float = 20.0,
    max_batch: int = 16,
    deadline_s: float = 0.020,
    main_wh: tuple[int, int] = (2560, 1440),
) -> dict[str, Any]:
    """p50/p95 latency of healthy cameras: all alive vs. cam3 dead vs. cam5 slow (1 fps)."""
    import torch

    w, h = main_wh
    img = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8, device="cuda")

    def run(dead: str | None, slow: str | None) -> dict[str, float]:
        lat: dict[str, list[float]] = {}
        lock = threading.Lock()

        def on_result(ref: FrameRef, dets: Any, t: Any) -> None:
            with lock:
                lat.setdefault(ref.camera_id, []).append(t.latency)

        b = MicroBatcher(det, on_result, max_batch=max_batch, deadline_s=deadline_s)
        stop = threading.Event()

        def producer(cam: str, period: float, phase: float) -> None:
            time.sleep(phase)
            seq, nxt = 0, time.monotonic()
            while not stop.is_set():
                b.submit(
                    FrameRef(
                        camera_id=cam, epoch=0, seq=seq, frame_idx=seq, ts=time.time(), width=w, height=h
                    ),
                    img,
                )
                seq += 1
                nxt += period
                time.sleep(max(0.0, nxt - time.monotonic()))

        ths = [
            threading.Thread(
                target=producer,
                args=(f"cam{i}", 1.0 / (1.0 if f"cam{i}" == slow else fps), i * 0.1 / n_cams),
                daemon=True,
            )
            for i in range(n_cams)
            if f"cam{i}" != dead
        ]
        with b:
            for t in ths:
                t.start()
            time.sleep(seconds)
            stop.set()
            for t in ths:
                t.join()
            time.sleep(0.2)
        healthy = [x for c, v in lat.items() if c not in (dead, slow) for x in v[5:]]  # skip warm-up
        s = _stats(healthy)
        s["mean_batch"] = statistics.fmean(b.stats.batch_sizes)
        s["dropped"] = sum(b.stats.dropped.values())
        return s

    res = {"all_alive": run(None, None), "cam3_dead": run("cam3", None), "cam5_slow_1fps": run(None, "cam5")}
    for k, v in res.items():
        print(
            f"isolation {k:>15}: healthy-camera p50 {v['p50_ms']:.1f} ms p95 {v['p95_ms']:.1f} ms "
            f"(mean batch {v['mean_batch']:.1f})",
            flush=True,
        )
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--sizes", default="1,4,8,10,16")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--main-wh", default="2560x1440")
    ap.add_argument("--isolation-seconds", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=Path("runs/t03/bench"))
    a = ap.parse_args(argv)
    wh = tuple(int(v) for v in a.main_wh.split("x"))
    det = make_detector(a.detector, engine=a.engine)
    rec = {"detector": det.model_id, "engine": a.engine, "main_wh": wh, "env": env_record()}
    recipe = Path(a.engine).with_suffix(".recipe.json")
    if recipe.exists():
        rec["recipe"] = json.loads(recipe.read_text())
    rec["batches"] = bench_batches(det, [int(s) for s in a.sizes.split(",")], a.iters, wh)  # type: ignore[arg-type]
    if a.isolation_seconds > 0:
        rec["isolation"] = isolation(det, seconds=a.isolation_seconds, main_wh=wh)  # type: ignore[arg-type]
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"{Path(a.engine).parent.name}_{Path(a.engine).stem}.json"
    f.write_text(json.dumps(rec, indent=2, default=float))
    print(f"wrote {f}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
