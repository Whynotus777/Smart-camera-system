# PoC demo clips

Two short clips recorded by the Sept-2025 PoC (`legacy/deepsort_poc.py`), kept as small
fixtures for smoke tests. They're the only video tracked in git (AGENTS.md rule 4; the
exception is by owner decision, and `.gitignore` names each file explicitly).

| File | Codec | Size | fps | Length | Source |
|---|---|---|---|---|---|
| `demo_1.mp4` | MPEG-4 Part 2 | 640×360 | 20 | 14.2 s (284 frames) | Reolink sub-stream, lab |
| `demo2.mp4` | MPEG-4 Part 2 | 640×360 | 20 | 9.2 s (184 frames) | Reolink sub-stream, lab |

Limits (read before using them as a baseline):
- They're PoC **output**. Detection boxes, track IDs and the camera's timestamp OSD
  are **burned into the pixels**. A detector sees the drawn boxes too, so treat any
  detector or tracker numbers on these clips as smoke tests, not benchmarks.
- The camera is at desk height, not ceiling-mounted, and the footage is sub-stream only.
- They show team members' faces. Don't publish frames, and don't use them for
  anything identity-related (AGENTS.md rule 6).
