# T10 — Evidence clips & review queue

**Wave 2 · owned:** `src/scs/evidence/`

## Goal
Every alert comes with a short, reviewable clip, and a reviewer can confirm or dismiss it. Reviewer decisions become labels.

## Deliverables
1. GPU-friendly ring buffer per camera holding the last 60 s of **encoded** packets (not decoded frames) so memory stays flat; clip export = remux (no re-encode) from pre-roll 15 s to post-roll 10 s, across all cameras in the alert.
2. Optional privacy overlay render (blur faces of non-subjects) as a separate output; the original stays access-controlled.
3. Review queue service (FastAPI + SQLite to start): list pending alerts, view clips + journey timeline (from T05), confirm/dismiss with reason codes, export labels to canonical format.
4. Retention policy config: dismissed clips deleted after N days, confirmed after M days; pose/track data retained per policy.
5. Notification adapters behind an interface (email/SMS/webhook) that fire **only on confirmed** alerts; the legacy `alerts/notifier.py` is retired.

## Acceptance
- [ ] RSS stays flat over a 24 h soak with 10 streams buffering.
- [ ] Clip export for a 2-camera alert completes in < 3 s after post-roll ends.
- [ ] Review decisions round-trip into `eval` as labels.

## Out of scope
Production auth/SSO and multi-tenant hosting (CTO phase). Basic auth is fine.
