# Task briefs

Each brief is self-contained: an agent should be able to start from the brief +
`AGENTS.md` + `docs/`. Claim a task by opening a draft PR from `agent/<id>-<slug>`.

| id | Title | Wave | Owned paths | Depends on |
|---|---|---|---|---|
| [T00](T00-contracts-skeleton.md) | Contracts, skeleton, CI | 0 | `src/scs/contracts.py`, `src/scs/*/base.py`, `pyproject.toml`, `.github/` | — |
| [T01](T01-hygiene-legacy.md) | Security hygiene & legacy quarantine | 0 | root legacy files, `legacy/`, `README.md` | — |
| [T02](T02-ingest.md) | Ingest (RTSP/file/sim sources) | 1 | `src/scs/ingest/` | T00 |
| [T03](T03-detect-track.md) | Detection + tracking | 1 | `src/scs/perception/detect*`, `track*` | T00 |
| [T04](T04-pose.md) | Pose on high-res crops | 1 | `src/scs/perception/pose*` | T00 |
| [T05](T05-journey-engine.md) | Zones + journey state machine | 2 | `src/scs/events/` | T00 (fixtures from T03/T04) |
| [T06](T06-behavior-models.md) | Behavior models (pose sequences) | 2 | `src/scs/behavior/`, `eval/suites/public_pose*` | T09 |
| [T07](T07-camera-emulator.md) | Camera emulator & profiles | 1 | `src/scs/camera_emu/`, `configs/camera_profiles/` | T00 |
| [T08](T08-isaac-sim.md) | Isaac Sim synthetic store | 1 | `sim/` | T00, T07 (profiles) |
| [T09](T09-eval-harness.md) | Eval harness + dataset converters | 1 | `eval/`, `docs/EVAL.md` | T00 |
| [T10](T10-evidence-review.md) | Evidence clips + review queue | 2 | `src/scs/evidence/` | T00, T02 |
| [T11](T11-vlm-verifier.md) | VLM verifier | 2 | `src/scs/verify/` | T10 (clip format) |
| [T12](T12-runtime-perf.md) | Runtime, packaging, perf/soak, edge decision | 3 | `src/scs/runtime/`, `docker/`, `eval/suites/perf*`, `soak*` | all |

Suggested agent allocation on one 5090: Wave 1 = 6 agents (T08 on its own schedule),
Wave 2 = 4 agents, Wave 3 = 1–2 agents plus reviewers.
