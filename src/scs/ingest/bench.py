"""T02 acceptance benchmark: N looping 1440p15 files on NVDEC, per-camera stats over 10 min.

    python -m scs.ingest.bench prepare                    # MEVA → 1440p15 H.264/H.265 (NVENC)
    scripts/gpu exclusive -- python -m scs.ingest.bench run --codec h265 --duration 600

`prepare` picks one downloaded MEVA clip per camera (CC-BY-4.0, attribution in the
manifest), and re-encodes it like a Reolink 4MP main stream: 2560×1440, 15 fps, 2 s GOP,
no B-frames, H.264 ≈ 6 Mbit/s and H.265 ≈ 4 Mbit/s. MEVA is 1080p, so frames are
*upscaled*; decode cost depends on resolution and bitrate, not on detail, but reports
must say so.

`run` starts N `FileSource(realtime=True, loop=True, decode="nvdec")` in a
`CameraManager`, drains them like an analytics stage would (micro-batches, conversion to
CUDA RGB tensors), and writes a JSON report: per-camera decoded/dropped/errors, decode
fps, fresh-frame age p50/p95/max (host monotonic clock, sampled at 10 Hz), plus the
decoder element that ran. `nvidia-smi dmon` runs alongside as supporting evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scs.ingest.manager import CameraManager
from scs.ingest.sources import FileSource

NVDEC = {"nvh264dec", "nvh265dec", "nvv4l2decoder", "h264_cuvid", "hevc_cuvid"}
VARIANTS = {
    "h264": ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "6M", "-maxrate", "8M", "-bufsize", "12M"],
    "h265": ["-c:v", "hevc_nvenc", "-preset", "p4", "-b:v", "4M", "-maxrate", "6M", "-bufsize", "8M"],
}


def data_root() -> Path:
    env = os.environ.get("SCS_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    git = shutil.which("git")
    if git:
        r = subprocess.run([git, "rev-parse", "--path-format=absolute", "--git-common-dir"],  # noqa: S603
                           cwd=Path(__file__).parent, capture_output=True, text=True)
        if r.returncode == 0:
            return Path(r.stdout.strip()).parent / "data"
    return Path("data").resolve()


def bench_dir() -> Path:
    return data_root() / "ingest_bench"


# --------------------------------------------------------------------------- prepare


def pick_meva(n: int) -> list[dict[str, Any]]:
    root = data_root() / "meva"
    have = json.loads((root / "MANIFEST.json").read_text())["files"]
    best: dict[str, dict[str, Any]] = {}
    for line in (root / "index" / "clips.jsonl").read_text().splitlines():
        r = json.loads(line)
        rel = r["key"].split("/", 1)[1]
        if rel in have and r["camera"] not in best:
            best[r["camera"]] = {**r, "path": str(root / "raw" / rel)}
    return sorted(best.values(), key=lambda r: r["camera"])[:n]


def prepare(n: int = 10, seconds: int = 300, gpu_wrapper: str | None = None, jobs: int = 4) -> list[Path]:
    """Encode the loop files (idempotent). Each ffmpeg call runs under `gpu_wrapper shared`."""
    from concurrent.futures import ThreadPoolExecutor

    ff = shutil.which("ffmpeg") or sys.exit("ffmpeg not found")
    out = bench_dir()
    out.mkdir(parents=True, exist_ok=True)
    clips = pick_meva(n)
    if len(clips) < n:
        print(f"only {len(clips)} MEVA cameras downloaded; using those", file=sys.stderr)
    wrap = [gpu_wrapper, "shared", "--"] if gpu_wrapper else []
    todo = [(i, c, codec, out / f"cam{i + 1:02d}_{codec}_1440p15.mp4")
            for i, c in enumerate(clips) for codec in VARIANTS]

    def encode(item: tuple[int, dict[str, Any], str, Path]) -> Path:
        _i, c, codec, dest = item
        if not dest.exists():
            tmp = dest.with_suffix(".part.mp4")
            subprocess.run([*wrap, ff, "-v", "error", "-y", "-hwaccel", "cuda",  # noqa: S603
                            "-hwaccel_output_format", "cuda", "-t", str(seconds), "-i", c["path"], "-an",
                            "-vf", "scale_cuda=2560:1440", "-r", "15", *VARIANTS[codec], "-g", "30",
                            "-bf", "0",
                            "-movflags", "+faststart", str(tmp)], check=True)
            tmp.replace(dest)
        print(f"{dest.name} <- {c['clip']}", flush=True)
        return dest

    with ThreadPoolExecutor(jobs) as ex:
        made = list(ex.map(encode, todo))
    rows = [{"file": d.name, "codec": codec, "source_clip": c["clip"], "source_camera": c["camera"],
             "bytes": d.stat().st_size} for (_i, c, codec, d) in todo]
    (out / "MANIFEST.json").write_text(json.dumps({
        "dataset_id": "ingest_bench", "license": "CC-BY-4.0 (derived from MEVA)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Derived from MEVA (Kitware Inc. and IARPA), CC BY 4.0. Upscaled 1080p->1440p, "
                       "re-encoded 15 fps for decode benchmarking (T02).",
        "use": "prod", "files": rows}, indent=1))
    return made


# --------------------------------------------------------------------------- run


def _gpu_state() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found"
    q = "name,driver_version,memory.used,memory.total,utilization.gpu,utilization.decoder,temperature.gpu"
    r = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)  # noqa: S603
    return r.stdout.strip() or r.stderr.strip()


def _git_state() -> str:
    """Commit the benchmark ran on, plus `-dirty` if src/ had uncommitted changes."""
    git = shutil.which("git")
    if not git:
        return "unknown"
    here = Path(__file__).parent
    sha = subprocess.run([git, "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,  # noqa: S603
                         text=True).stdout.strip()
    dirty = subprocess.run([git, "status", "--porcelain", "--", "."], cwd=here, capture_output=True,  # noqa: S603
                           text=True).stdout.strip()
    return f"{sha}{'-dirty' if dirty else ''}"


def run(codec: str, n: int, duration: float, backend: str, output: str, out_json: Path | None,
        deadline_s: float = 0.02, cv2_threads: int = 1) -> dict[str, Any]:
    files = sorted(bench_dir().glob(f"cam*_{codec}_1440p15.mp4"))[:n]
    if len(files) < n:
        raise SystemExit(f"need {n} {codec} files in {bench_dir()}, found {len(files)}; run `prepare` first")
    dmon_log = (out_json.with_suffix(".dmon.txt") if out_json else None)
    dmon = None
    if dmon_log and shutil.which("nvidia-smi"):
        dmon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "5"],  # noqa: S603, S607
                                stdout=dmon_log.open("w"), stderr=subprocess.STDOUT)
    gpu_start = _gpu_state()
    if output == "numpy":
        # OpenCV's default pool busy-spins ~8 cores converting 150 fps of 1440p; 1 thread costs ~0.15.
        import cv2

        cv2.setNumThreads(cv2_threads)
    srcs = [FileSource(f, f"cam{i + 1:02d}", loop=True, realtime=True, backend=backend,  # type: ignore[arg-type]
                       decode="nvdec", queue_size=4) for i, f in enumerate(files)]
    consumed = 0
    batches = 0
    mgr = CameraManager(srcs).start()
    # Warm-up: wait until every camera has decoded a frame, then measure a clean window.
    t_start = time.monotonic()
    first: dict[str, float] = {}
    while len(first) < n and time.monotonic() - t_start < 60:
        for s in srcs:
            if s.camera_id not in first and s.stats.decoded > 0:
                first[s.camera_id] = round(time.monotonic() - t_start, 2)
        mgr.next_batch(deadline_s=deadline_s, output="raw")
    base = {s.camera_id: (s.stats.decoded, s.stats.dropped, s.stats.errors, s.stats.reconnects) for s in srcs}
    for s in srcs:
        s.stats.reset_samples()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.monotonic()
    last_print = t0
    try:
        while (now := time.monotonic()) - t0 < duration:
            batch = mgr.next_batch(deadline_s=deadline_s, output=output)  # type: ignore[arg-type]
            consumed += len(batch)
            batches += bool(batch)
            if output == "torch" and batch:
                import torch

                torch.cuda.synchronize()
            if now - last_print >= 30:
                last_print = now
                snap = mgr.snapshot()
                fps = [v["fps"] for v in snap.values()]
                dropped = sum(v["dropped"] for v in snap.values()) - sum(b[1] for b in base.values())
                print(f"[{now - t0:6.0f}s] fps min/max {min(fps):.2f}/{max(fps):.2f} dropped {dropped} "
                      f"errors {sum(v['errors'] for v in snap.values())}", flush=True)
        elapsed = time.monotonic() - t0
        snap = mgr.snapshot()
    finally:
        mgr.close()
        if dmon:
            dmon.terminate()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    expected = 15.0 * elapsed
    cams = {}
    for cid, st in snap.items():
        d0, dr0, e0, r0 = base[cid]
        dec = st["decoded"] - d0
        cams[cid] = {**st, "decoded": dec, "dropped": st["dropped"] - dr0, "errors": st["errors"] - e0,
                     "reconnects": st["reconnects"] - r0, "decoded_total_incl_warmup": st["decoded"],
                     "time_to_first_frame_s": first.get(cid), "decode_fps_avg": round(dec / elapsed, 3),
                     "decoded_vs_expected": round(dec / expected, 4)}
    worst_p95 = max((c["fresh_frame_age_p95_s"] or 0) for c in cams.values())
    ok = (all(c["decoded_vs_expected"] >= 0.99 for c in cams.values())
          and all(c["errors"] == 0 for c in cams.values())
          and all(c["decoder"] in NVDEC for c in cams.values()))
    report = {
        "suite": "T02-ingest-decode", "codec": codec, "cameras": n, "resolution": "2560x1440", "fps": 15,
        "source": "MEVA upscaled 1080p->1440p (CC-BY-4.0), see ingest_bench/MANIFEST.json",
        "backend": backend, "output": output, "duration_s": round(elapsed, 1),
        "expected_frames_per_camera": round(expected), "gpu_at_start": gpu_start,
        "warmup_s": round(t0 - t_start, 2),
        "host": platform.node(), "python": platform.python_version(), "git": _git_state(),
        "clock_note": "fresh_frame_age_* = host monotonic now - last decode, sampled 10 Hz over the "
                      "measured window; decode_fps = decoded / wall elapsed; the window starts once every "
                      "camera has decoded its first frame (warm-up and time_to_first_frame_s reported)",
        "consumer": {"frames": consumed, "batches": batches, "deadline_s": deadline_s,
                     "cv2_threads": cv2_threads if output == "numpy" else None},
        "process_cpu_cores_avg": round(cpu_s / elapsed, 2),
        "totals": {"decoded": sum(c["decoded"] for c in cams.values()),
                   "dropped": sum(c["dropped"] for c in cams.values()),
                   "errors": sum(c["errors"] for c in cams.values()),
                   "worst_fresh_frame_age_p95_s": worst_p95},
        "pass": ok, "per_camera": cams,
        "dmon_log": str(dmon_log) if dmon_log else None,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scs.ingest.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seconds", type=int, default=300)
    p.add_argument("--gpu-wrapper", default=None, help="path to scripts/gpu; wraps each encode in `shared`")
    p.add_argument("-j", "--jobs", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--codec", choices=list(VARIANTS), required=True)
    r.add_argument("-n", type=int, default=10)
    r.add_argument("--duration", type=float, default=600)
    r.add_argument("--backend", choices=["gstreamer", "pyav", "auto"], default="gstreamer")
    r.add_argument("--output", choices=["torch", "numpy", "raw"], default="torch")
    r.add_argument("--json", type=Path, default=None)
    r.add_argument("--cv2-threads", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        for f in prepare(a.n, a.seconds, a.gpu_wrapper, a.jobs):
            print(f)
        return 0
    rep = run(a.codec, a.n, a.duration, a.backend, a.output, a.json, cv2_threads=a.cv2_threads)
    print(json.dumps({k: rep[k] for k in ("codec", "backend", "duration_s", "totals", "pass")}, indent=1))
    for cid, c in rep["per_camera"].items():
        print(f"{cid} {c['decoder']:<10} decoded {c['decoded']:>6} dropped {c['dropped']:>5} "
              f"errors {c['errors']} fps {c['decode_fps_avg']:.2f} age p95 {c['fresh_frame_age_p95_s']}s")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
