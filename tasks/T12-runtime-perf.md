# T12 — Runtime, packaging, perf/soak, edge decision

**Wave 3 · owned:** `src/scs/runtime/`, `docker/`, `eval/suites/perf*`, `eval/suites/soak*`

## Goal
One service that runs the pipeline headless for a site config, survives a 24 h soak, and produces the data for choosing the edge hardware.

## Deliverables
1. `scs run --site configs/sites/<id>.yaml`: process layout (GPU pipeline process + bus consumers), graceful shutdown, config validation, structured JSON logs, Prometheus metrics (fps, latency, queue depth, reconnects, VRAM, NVDEC).
2. Docker images: `scs-runtime` (TensorRT base) and `scs-review`; `docker compose` for runtime + redis + review UI. Engines built on first start per GPU and cached in a volume.
3. `perf` suite: 1, 4, 10 streams; report fps, p50/p95 latency, GPU/NVDEC utilization.
4. `soak` suite: 24 h, 10 streams, forced RTSP drops every hour; zero crashes, bounded memory.
5. Edge decision memo `docs/reports/T12-edge-target.md`: extrapolate from 5090 measurements to Jetson Orin NX/AGX and to a small x86 + RTX 4000-class box. Where possible, measure on real hardware. Include BOM cost, power, NVDEC headroom for H.265, and remote management.

## Acceptance
- [ ] G1 perf/soak criteria in `docs/EVAL.md` met on the 5090.
- [ ] Cold start to first alert-capable frame < 3 min with cached engines.
- [ ] A new engineer can run the full stack from `README.md` on a clean Ubuntu + NVIDIA box.
