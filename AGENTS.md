# AGENTS.md — operating manual for coding agents

This repo is being rebuilt from a lab PoC into a production retail loss-prevention
system (first customer: 7-Eleven, Reolink cameras). Multiple agents (Claude Code,
Codex, others) work in parallel. Read this file fully before starting any task.

## What we're building (one paragraph)

An edge box ingests 4–10 RTSP cameras, detects and tracks people, estimates pose on
high-resolution crops, and runs a **journey state machine** per person
(shelf interaction → item pickup → conceal candidate → exit without checkout).
Suspicious journeys become `Alert`s with evidence clips that go to a **human review
queue** before anything reaches the store. Everything is measured by the eval
harness in `eval/` against public data, simulation, and our own staged footage.
Gesture-only alerting (the legacy approach) is explicitly retired.

Read next: `docs/ARCHITECTURE.md` → `docs/ROADMAP.md` → your task in `tasks/`.

## Ground rules

1. **One task = one branch = one worktree.** Branch name `agent/<task-id>-<slug>`
   (e.g. `agent/T07-camera-emulator`). Use `git worktree add ../scs-T07 -b agent/T07-camera-emulator`.
2. **Stay inside your task's owned paths** (listed in each brief). Every task also owns
   `tests/<its area>/` and `docs/reports/<task-id>-*.md`. Need a change elsewhere?
   Open an issue or leave a `HANDOFF.md` note in your PR; don't edit it yourself.
   Wave 1 branches from the tag `wave0-baseline` on `main`, never from another agent's branch.
3. **`src/scs/contracts.py` is frozen for Wave 1+.** You may *add optional fields*.
   Anything else (rename, remove, change meaning) needs `docs/adr/NNNN-*.md` and a
   `CONTRACTS_VERSION` bump, in a PR that touches nothing else.
4. **Never commit**: credentials, RTSP URLs, video, images of real people, datasets,
   model weights (`*.pt/*.pth/*.engine/*.onnx`). `data/`, `models/`, `runs/` are gitignored
   (except `models/MANIFEST.yaml`). The only tracked video is the two PoC clips in
   `tests/fixtures/video/` (owner decision).
   Datasets are registered in `docs/DATA.md` with their license; weights in `models/MANIFEST.yaml`.
5. **License gate.** Before adding any dependency, dataset, pretrained weight, or sim asset,
   record its license in `docs/DATA.md` (data/weights/assets) or the PR description (code).
   - AGPL / non-commercial / "research only": R&D and eval only, behind a swappable
     interface, PR title tagged `[license-risk]`.
   - **No license stated = not licensed** (default copyright). Status `pending`: don't
     download into shared paths or train anything on it until approved in `docs/DATA.md`.
   - Track lineage: any model trained on R&D-only data inherits R&D-only status.
6. **Privacy by default.** No face recognition, no identity databases. Persist pose +
   tracks, not pixels, except alert evidence clips. Cross-camera appearance embeddings
   live in memory only and expire with the visit (≤ 30 min).
7. **Numbers or it didn't happen.** Any PR that changes model behavior attaches the
   `eval/` report (JSON + markdown summary) for the relevant suites and compares to
   `main`. Regressions > 2 pts on a gated metric block merge.
8. **Tests.** `pytest -m "not gpu and not data and not slow"` must pass on CPU in CI.
   GPU/data tests are marked and must pass locally on the 5090 box; paste the output.
9. **Small PRs.** Aim for < 600 changed lines. Split otherwise.
10. **When blocked, write it down.** Add a `## Blockers` section to your PR. Stop the
    *affected* work if the blocker is an interface, licensing, security, or correctness
    decision. Keep going on independent parts of your task that don't need a guess.
11. **Causal by default.** Anything on the live path may only use past frames. If an
    offline variant uses future frames (smoothing, gap filling), it must say so, and eval
    must run the causal version.
12. **Zero-spend.** No paid datasets, paid APIs, cloud GPUs, or purchases without Abdul's
    written OK in the PR. Free, commercially licensed sources first (see `docs/DATA.md`).

## Environment (the 5090 workstation, Ubuntu)

- Python 3.11 via `uv` (pinned in `.python-version`): `uv venv && uv pip install -e ".[dev,perception]"`.
  GPU tasks add the shared `torch` extra (cu128 wheels, configured in `pyproject.toml`):
  `uv pip install -e ".[dev,perception,torch]"`. Don't install torch any other way.
- NVIDIA driver: use the version listed as validated on the Isaac Sim requirements
  page for the installed Isaac Sim release; Blackwell (sm_120) + Isaac Sim is
  driver-sensitive. Don't upgrade the driver without checking.
- PyTorch ≥ 2.7 built for CUDA 12.8+ (`cu128` wheels or newer). Older wheels don't
  include sm_120 kernels and fail at runtime. Framework support ≠ whole-stack support:
  every compiled extension/exporter you add (mmcv, TensorRT plugins, etc.) must be tested
  on sm_120, and the working set pinned in a lockfile.
- TensorRT 10.x for deployment engines. Build engines per GPU; never commit them.
- Docker + NVIDIA Container Toolkit. Isaac Sim runs in its official container (T08).
- Redis ≥ 7 (Streams) via `docker run -p 6379:6379 redis:7`.

### Sharing one GPU across many agents

Every GPU command goes through `scripts/gpu` (T00 adds it; a ~20-line `flock` wrapper):
- `scripts/gpu shared -- <cmd>`: short interactive tests (< 2 min, < 6 GB VRAM). Many can run at once.
- `scripts/gpu exclusive -- <cmd>`: benchmarks, training, Isaac Sim rendering. Waits until
  no shared or exclusive holder remains, then blocks others. **Benchmark numbers are only
  valid if taken under `exclusive`**; reports include `nvidia-smi` state at start.
- Long jobs are queued with `tsp` (task-spooler) *and* wrapped in `scripts/gpu exclusive`.

The 5090 is the **development** GPU. The store appliance is chosen in T12 and must be
benchmarked on real target hardware. TensorRT engines are built per target from a recorded
recipe; engines never go in git, but their recipe, versions, and hash go in `models/MANIFEST.yaml`.

## Repo layout (target)

```
src/scs/
  contracts.py        # shared schemas (frozen)
  bus.py              # Redis Streams helper + in-memory fake
  geometry.py         # polygon/zone helpers (T04 and T05 must use these)
  ingest/             # T02  RTSP/file/sim sources → frames
  perception/         # T03  detector + tracker; T04 pose
  behavior/           # T06  behavior models → BehaviorScore
  events/             # T05  zones + journey state machine → Event, Alert
  evidence/           # T10  ring buffer, clip export, review queue API
  verify/             # T11  VLM second-stage verifier
  camera_emu/         # T07  camera emulation / degradation
  runtime/            # T12  process orchestration, config, health
sim/                  # T08  Isaac Sim scenes, actor behaviors, camera rigs
eval/                 # T09  dataset loaders, metrics, suites, reports
configs/              # camera profiles, example site
docs/                 # architecture, roadmap, data registry, eval spec, ADRs
tasks/                # one brief per workstream
legacy/               # the Sept-2025 PoC — reference only, do not extend or import
```

## Definition of done (every task)

- Acceptance criteria in the task brief all checked, with evidence in the PR.
- Unit tests + at least one integration test against recorded fixtures.
- Public functions typed; module docstring explains the *why*.
- `docs/` updated where behavior or interfaces changed.
- PR description: what, why, eval deltas, license notes, known gaps, follow-ups.
