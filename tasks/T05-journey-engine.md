# T05 — Zones + journey state machine

**Wave 2 · owned:** `src/scs/events/`

## Goal
Turn tracks, poses, and behavior scores into interpretable `Event`s and `Alert`s. This replaces single-frame gesture alarms (ARCHITECTURE D2).

## Deliverables
1. Zone engine: normalized polygons from `SiteConfig`, with all point-in-polygon/intersection math via **`scs.geometry`** (shared with T04; extend it through a T00-owned PR if something's missing); enter/exit events with hysteresis (min dwell 0.5 s) using the **foot point** (bottom-center of the box) for floor zones and **wrist keypoints** for shelf zones.
2. `SHELF_INTERACTION`: a wrist inside a SHELF/HIGH_VALUE polygon for ≥ N frames. `ITEM_PICKUP` heuristic v0: interaction followed by a wrist trajectory moving away from the shelf toward the torso. Leave a hook for a learned classifier.
3. Journey state machine per (camera, track) and optionally per global id:
   `BROWSING → INTERACTED(zone, hv?) → [CONCEAL_CANDIDATE] → CHECKOUT_VISIT | STORE_EXIT`.
4. Alert policy (configurable YAML, versioned):
   - `A1`: pickup in HIGH_VALUE + conceal candidate (score ≥ θc) + exit without checkout → score from calibrated combination.
   - `A2`: conceal candidate ≥ θhigh alone → **analytics only** by default (not an alert).
   - Dedupe: at most one open alert per global id per visit.
5. Replay CLI: `python -m scs.events.replay --tracks x.jsonl --poses y.jsonl --site site.yaml` → events/alerts JSONL + an HTML timeline per track.

## Acceptance
- [ ] Deterministic on the T00 fixtures; unit tests for every transition and the dedupe.
- [ ] Track loss/re-acquire within 3 s keeps the journey (merge by proximity + time), tested.
- [ ] On `sim_matrix` clips with ground-truth journeys: checkout-visit and store-exit F1 ≥ 0.9 *(when data exists; substitute: hand-built journey fixtures in the T00 format covering each transition)*.
- [ ] Policy file changes are reflected in eval reports (policy hash recorded).

## Out of scope
Learned pickup detection (follow-up once `lab_mock_aisle` exists).
