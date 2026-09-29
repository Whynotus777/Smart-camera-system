# 0001 — Shared data contracts v0.1

- Status: accepted
- Date: 2026-09-29
- Deciders: planning kit (agent-kit PR), T00
- Task: T00
- Contracts: none → 0.1.0

## Context

Wave 1 runs up to six agents in parallel on separate stages (ingest, detect/track,
pose, camera emulation, sim, eval). They need to agree on the types that cross
stage boundaries before any stage exists, or they'll collide at integration.

## Decision

1. `src/scs/contracts.py` is the only coupling point between workstreams. It
   defines pydantic v2 models, all `extra="forbid"` and `frozen=True`:
   - config: `StreamSpec`, `CameraProfile`, `Zone`, `CameraInstall`, `SiteConfig`
   - perception: `FrameRef`, `Detection`, `Track`, `Pose` (COCO17), `BehaviorScore`
   - events: `Event` (`EventType`), `Alert` (`AlertStatus`)
   - bus topology: `Streams` stream names
2. Conventions, as documented in the module docstring: epoch-second UTC timestamps
   taken at decode; absolute pixel coords in the frame the stage received; zone
   polygons normalized to [0, 1]; keypoints COCO17 `(x, y, conf)` in pixels.
3. Secrets never appear in configs. `CameraInstall` holds only env var *names*
   (validated as `UPPER_SNAKE`).
4. Stage interfaces are `typing.Protocol`s in `src/scs/<area>/base.py`, copied
   verbatim from `docs/ARCHITECTURE.md` §4. torch appears only under
   `TYPE_CHECKING`, so the core package stays CPU/torch-free.
5. Wire format (`src/scs/bus.py`): one Redis Streams entry per model, with fields
   `v=<CONTRACTS_VERSION>` and `json=<model_dump_json()>`. Every publish trims its
   stream with `MAXLEN ~`. Consumers read through consumer groups and ack
   explicitly (at-least-once delivery).
6. Change policy (AGENTS.md rule 3): adding optional fields is free. Renaming,
   removing, or changing a field's meaning needs a new ADR plus a
   `CONTRACTS_VERSION` bump, in a PR that touches nothing else.

## Consequences

- Every stage can be tested alone against JSONL fixtures (`tests/fixtures/`) or the
  `InMemoryBus`, with no Redis and no GPU.
- `extra="forbid"` means a producer on a newer minor version that adds a field
  breaks older consumers. Deploy consumers before producers when adding fields.
- The `v` field lets consumers detect version skew; nothing enforces it yet (T12).

## Open questions (resolve by ADR before the affected Wave 1 work merges)

- **Coordinate frame of `Detection`/`Track` boxes.** D1 decodes the main stream only,
  but `FrameRef.stream` defaults to `"sub"`. T04's brief also says pose crops use
  "track boxes scaled from detector resolution", which suggests tracks are in
  detector-input pixels. Pick one: boxes in main-stream pixels (with `FrameRef`
  describing the main frame), or detector-input pixels plus a way to recover the
  scale. The T00 fixtures use main-stream pixels (`stream="main"`, 2560×1440).
- **Monotonic timestamps.** T02 wants both wall-clock and monotonic time in
  `FrameRef.ts`, but there is only one float. Adding an optional
  `FrameRef.ts_mono` is allowed without a version bump.

## Alternatives considered

- Protobuf/msgpack on the wire: smaller and faster, but JSON is debuggable with
  `redis-cli`, and payloads are small next to video. Revisit if T12 shows the bus
  is a bottleneck.
- Dataclasses without validation: cheaper, but boundary validation (bbox order,
  17 keypoints, env var names) catches the integration bugs parallel agents cause.
