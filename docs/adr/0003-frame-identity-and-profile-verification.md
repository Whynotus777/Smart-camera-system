# 0003 — Frame identity, capture time, transforms; camera-profile verification levels

- Status: accepted
- Date: 2026-09-29
- Deciders: Abdul (owner, T00 Addendum / review round 1), T00
- Task: T00 (Addendum)
- Contracts: 0.2.0 → 0.3.0

## Context

Review round 1 (ARCHITECTURE D10) found two gaps:

1. **Frame identity.** `frame_idx` resets whenever an RTSP source reconnects, so
   `(camera_id, frame_idx)` can name two different images. Nothing proved that a
   detector box and a high-resolution pose crop came from the same frame. Latency
   was only measured from decode, not from capture.
2. **Camera profile trust.** `verified: bool` couldn't tell a spec-sheet number from a
   measured one, or from an emulator calibrated against real footage. A datasheet says
   nothing about noise, IR or low-light behavior, and those are exactly what T07
   emulates.

## Decision

1. `FrameRef` gains:
   - `epoch: int` (≥ 0, **required**): connection id. The first connection is 0, and it
     increments on every (re)connect of the source.
   - `seq: int` (≥ 0, **required**): frame counter within an epoch. It starts at 0 and
     increases by 1 per decoded frame.
   - `source_ts: float | None`: capture time from the RTP/camera clock, as epoch
     seconds, when the source provides it.
   - `transform: str | None`: preprocessing applied to the image the ref travels with,
     e.g. `"resize640:letterbox"`. `None` means the decoded main-stream frame as-is.
     Output coordinates stay main-stream (ADR 0002).

   **Frame identity = `(camera_id, epoch, seq)`**, exposed as `FrameRef.identity`.
   `frame_idx` (the source-level counter) and `ts` (decode wall clock) are kept.
   `ts_mono` is unchanged (ADR 0002). Capture-to-decode latency is `ts - source_ts`.
2. `CameraProfile.verified: bool` is **replaced** by:
   - `verification`, one of the following, weakest to strongest (default `approximation`):
     - `approximation`
     - `spec_sourced`: datasheet link in `sources`
     - `measured`: `rtsp_probe` or footage from the physical camera
     - `emulator_calibrated`: T07 output matched to that camera's real footage
   - `sources: list[str]`. Any level above `approximation` must cite at least one
     source; the validator enforces this.

   All three shipped profiles are `approximation`. The Reolink profile's sub-stream
   width/height/fps were measured with ffprobe on the PoC clips, recorded in its
   `notes` and `sources`. Its sub-stream codec and bitrate were not measured, because
   those clips are the PoC's MPEG-4 re-encode.
3. `CONTRACTS_VERSION` → **0.3.0**. It's a breaking change: new required fields, and a
   field removed.

## Consequences

- T02 must maintain `epoch`/`seq` per camera: bump `epoch` and reset `seq` on
  reconnect. It fills `source_ts` from RTP when available, and records `transform` on
  any resized copy it hands out.
- T03 sets `transform` on detector-input refs, but emits boxes in main-stream pixels.
  T04 checks that a crop's source frame has the same `identity` as the track's frame.
- Joins across stages (evidence clips, eval alignment, latency) key on `identity`,
  never on `frame_idx` or `ts`.
- Old YAML profiles with `verified:` now fail to load (`extra="forbid"`). The three
  in-repo profiles are migrated.
- The T00 JSONL fixtures carry `epoch=0` and `seq == frame_idx`.

## Alternatives considered

- **Optional `epoch`/`seq` with defaults:** consumers would silently merge frames
  from different connections, the bug this ADR exists to prevent.
- **A single global frame UUID:** unique, but not ordered. It loses "which connection"
  and "how many frames were missed", which T02 health stats and T13 durability need.
- **Keeping `verified` alongside `verification`:** two sources of truth; dropped.
