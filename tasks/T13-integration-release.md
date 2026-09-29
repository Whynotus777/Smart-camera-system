# T13 — Integration & release owner

**Wave 1 → ongoing · owned:** `tests/integration/`, `src/scs/app/` (walking-skeleton glue), `docs/RELEASE.md`, `docs/reports/T13-*.md`

## Goal
Own the product path end to end so six good component PRs add up to one working
system. You are the only agent judged on whether the **whole workflow** works.

## Milestones (see docs/ROADMAP.md)
**M1 — walking skeleton (end of Wave 1).** File or RTSP source → a persisted `Event` from a
trivial rule (e.g. "person dwells in zone X > 5 s") or an injected test event → a
playable before/after clip → a review item → a recorded review outcome.
- Start with the thinnest possible stand-ins: a `FileSource` stub, a fake detector, SQLite,
  ffmpeg remux for clips, a minimal FastAPI review page. Swap in the real T02/T03/T10
  pieces as they land, and keep the test green through each swap.
- Durability: kill -9 any process at any point in the flow. After restart, no event, clip,
  or review outcome is lost or duplicated. This is an automated test.

**M2 — installable (by G1).** With T12: a versioned bundle installs on a clean target
machine without manual fixes; explicit degraded states for camera loss, network loss,
process crash, disk full; model/config update with rollback.

**M3 — shadow-ready (before store).** Clip access control, retention/deletion, credential
handling, and review-burden metrics (review minutes/day, useful-alert rate,
evidence-retrieval time).

## Standing duties
- Maintain `tests/integration/`: runs the real pipeline from video files on every merge to
  `main`. Keep a CPU-only variant for CI and a GPU variant for the 5090.
- Enforce merge order in docs/ROADMAP.md; flag contract drift between agents early.
- Keep `docs/RELEASE.md`: the release checklist, current milestone status, open risks.

## Acceptance (M1)
- [ ] One command (`scs demo --video demo_1.mp4`) runs the whole flow headless and prints the review URL.
- [ ] Automated kill/restart test passes 20 runs in a row.
- [ ] Clip plays in a browser and covers ≥ 10 s before and ≥ 5 s after the event.
- [ ] Review outcome is stored and exported in the canonical label format.

## Out of scope
Detection quality. A dumb rule is correct for M1.
