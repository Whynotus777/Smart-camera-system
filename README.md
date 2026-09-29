# Smart Camera System

Retail loss-prevention perception for convenience stores. The first pilot is
7-Eleven stores with Reolink cameras.

An edge box ingests 4–10 RTSP cameras, detects and tracks people, and estimates pose
on high-resolution crops. A **journey state machine** then follows each person:
shelf interaction → item pickup → conceal candidate → exit without checkout.
Suspicious journeys become alerts with evidence clips. Every alert goes to a **human
review queue** before anything reaches the store; the system never confronts or
accuses anyone on its own. Everything is measured by an eval harness against public
data, simulation and staged lab footage.

This repo is mid-rebuild from a lab proof of concept into that system.

**Contributors and coding agents: read [`AGENTS.md`](AGENTS.md) first.** Its rules
are binding. Then read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/ROADMAP.md`](docs/ROADMAP.md), and your brief in [`tasks/`](tasks/).

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv -p 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"            # add ",perception" for OpenCV/PyAV work

pytest -m "not gpu and not data and not slow"
ruff check src tests
mypy src/scs/contracts.py src/scs/bus.py src/scs/*/base.py

uv tool install pre-commit && pre-commit install   # secret, credential and size checks
```

Optional, to run the bus tests against a real Redis:

```bash
docker run -d --rm -p 6379:6379 redis:7
SCS_REDIS_URL=redis://localhost:6379/15 pytest tests/test_bus.py
```

Camera URLs come from environment variables, never from code or configs. Copy
`.env.example` to `.env` (gitignored) and fill it in. See
[`docs/SECURITY.md`](docs/SECURITY.md).

## Layout

| Path | What |
|---|---|
| `src/scs/contracts.py` | Shared data types (pydantic). The only coupling point between stages. |
| `src/scs/bus.py` | Redis Streams helper, plus an in-memory fake for tests |
| `src/scs/<area>/base.py` | Stage interfaces: ingest, perception, behavior, events, verify |
| `configs/` | Camera profiles and an example site config |
| `tests/fixtures/` | Synthetic Track/Pose JSONL for building without cameras |
| `docs/` | Architecture, roadmap, eval spec, data/license registry, security, ADRs |
| `tasks/` | One brief per workstream (T00–T12) |
| `legacy/` | The Sept-2025 PoC, reference only |

## Legacy PoC

The original lab prototype (YOLO + ByteTrack/DeepSORT + pose gestures + LSTM, with
a Redis robot-dispatch demo) is in [`legacy/`](legacy/README.md). It's kept as a
reference: `docs/ARCHITECTURE.md` §6 lists what's worth porting. It isn't part of the
package, nothing imports it, and single-frame gesture alerting is retired.
`legacy/camera_system_with_lstm.py` can't run until four missing modules are
committed (ROADMAP H0b). `demo_1.mp4` and `demo2.mp4` at the root are the PoC's output videos.

## Authors

- Ishan Kharat (ishanmk@umd.edu)
- Abdul Manan (abdul@quantumroboticslab.com)

## License

MIT, see [LICENSE](LICENSE). Some R&D dependencies and weights carry other licenses
(AGPL, non-commercial); see [`docs/DATA.md`](docs/DATA.md) before shipping anything.
