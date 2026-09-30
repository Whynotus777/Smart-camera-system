# 0002 — Main-stream pixel coordinates; decode vs. monotonic timestamps

- Status: accepted
- Date: 2026-09-29
- Deciders: Abdul (owner), T00
- Task: T00 (Wave 0.5)
- Contracts: 0.1.0 → 0.2.0

## Context

ADR 0001 left two open questions:

1. **Which frame do box and keypoint pixel coordinates refer to?** ARCHITECTURE D1
   decodes the main stream only and resizes on the GPU for detection. But
   `FrameRef.stream` defaulted to `"sub"`, and the T04 brief spoke of "track boxes
   scaled from detector resolution". T03 and T04 would each have picked a different
   answer.
2. **Monotonic time.** T02 must record wall-clock and monotonic time at decode, but
   `FrameRef` had a single `ts` float.

## Decision

1. **All `Detection`/`Track` boxes and `Pose` keypoints are in the camera's
   main-stream, full-resolution pixel frame** (e.g. 2560×1440 for the Reolink 4MP).
   `FrameRef.width/height` are that frame's size.
   - A stage that runs on a resized image (the ~640 px detector input) or on the
     sub-stream maps its outputs back to main-stream pixels before emitting them.
   - Nothing on the bus is in detector-input pixels.
   - `FrameRef.stream` now **defaults to `"main"`** and records which stream was
     actually decoded. The coordinates are main-stream either way.
   - Zones remain normalized, and `scs.geometry` converts between the two.
2. **Timestamps:**
   - `FrameRef.ts` is epoch seconds (UTC), **wall clock at decode**.
   - New optional `FrameRef.ts_mono: float | None` is the decoding host's monotonic
     clock (`time.monotonic()`) at the same instant. Use it for intervals, stall
     detection and drift. It is not comparable across hosts or reboots.
   - Capture time from the camera/RTP clock is `FrameRef.source_ts`, added with the
     rest of the T00 Addendum fields (`epoch`, `seq`, `transform`) in
     [ADR 0003](0003-frame-identity-and-profile-verification.md).
3. `CONTRACTS_VERSION` → **0.2.0**. Changing a default is a change of meaning
   (AGENTS.md rule 3), so this gets a version bump even though the new field is
   optional.

## Consequences

- T03 detectors keep the letterbox/resize parameters needed to invert their
  preprocessing, and test the round trip.
- T04 crops directly from the main-stream frame using track boxes as-is. No
  rescaling is needed.
- T05 zone tests use `scs.geometry` with the `FrameRef` dimensions.
- T02 fills `ts` and `ts_mono` on every frame.
- Producers that relied on the old `"sub"` default: none. No stage existed yet, and
  the T00 fixtures already used `"main"`.

## Alternatives considered

- **Detector-input pixels plus a scale field:** every consumer would have to rescale,
  and sub/main aspect ratios can differ (e.g. 704×480 vs. 1920×1080), which is
  error-prone.
- **Normalized boxes:** loses the pixel scale that pose crops and size heuristics
  need, and differs from every public dataset format we ingest.
