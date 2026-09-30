# Roadmap — agent waves

Goal: before we're back in store, have a system that's **measured**: it runs on 10
streams, has behavior models trained on public + synthetic + staged data, and has a
data-backed camera recommendation. Then the CTO hardens it for deployment.

## Operating constraints (current)
- **Zero spend.** Free, commercially licensed data and open models only.
- **No cameras or store access for several weeks.** Everything runs on public data, sim, generated clips, and MEVA replayed as fake cameras. Real-footage items (H1, H1a, store archive) are optional/deferred; tasks must not block on them.

## Release milestones (owned by T13, run alongside the waves)

- **M1 — walking skeleton (target: end of Wave 1).** A replayed video (or a real Reolink)
  → a persisted `Event` from a trivial rule or injected test event → a playable
  before/after clip → a review item → a recorded review outcome. Kill and restart any
  process mid-flow: nothing is lost or duplicated.
- **M2 — installable (target: G1).** Versioned bundle installs on a clean target box with
  no manual fixes; camera/network/process/disk failures show explicit degraded states and
  recover; a model/config update can be deployed and rolled back.
- **M3 — shadow-ready (before store).** Access control on clips, retention/deletion
  working, credentials handled, review-burden metrics captured.

Research results (T03–T09, T11) never on their own authorize deployment. M-milestones and
G-gates both have to pass.

## Baseline and integration

- Wave 0 (agent-kit + T00 + T01) is reviewed, merged to `main`, and tagged **`wave0-baseline`**.
  Every Wave 1 branch starts from that tag.
- Merge order into `main`: T13 walking-skeleton scaffolding → T14 downloads/replay → T09 → T02 → T03 → T04 → T07 → T08.
  Each merge must keep T13's integration test green.
- T13 owns the cross-agent integration test suite (`tests/integration/`).

## Waves and dependencies

```
Wave 0 (serial, 1 agent, ~2 days)
  T00 contracts + skeleton + CI  ──┐   (contracts.py already drafted in this PR)
  T01 hygiene + legacy quarantine ─┤
                                   ▼
Wave 1 (parallel, up to 8 agents)
  T02 ingest ─────────┐
  T03 detect+track ───┤
  T04 pose ───────────┤
  T07 camera emulator ┤
  T08 sim (Isaac) ────┤   (separate env; long-running)
  T09 eval harness + dataset converters
  T13 integration + release owner (walking skeleton M1)
  T14 data factory (downloads, MEVA fake cameras, generation, pre-labels)
                      ▼
Wave 2 (parallel, up to 4 agents)
  T05 journey engine   (needs T03/T04 fixtures only)
  T06 behavior models  (needs T09 loaders)
  T10 evidence + review queue
  T11 VLM verifier
                      ▼
Wave 3
  T12 runtime, packaging, perf + soak, edge-target decision
  → Gate G1
                      ▼
Human-in-the-loop
  H1a quick capture (week 1: 1 cam, 30 min; unblocks T03/T04/T07 criteria)
  H1 lab mock-aisle recording (start week 1, runs in parallel with everything)
  → Gate G2 → CTO hand-off → store shadow → G3
```

## Human tasks (agents can't do these)

| id | Task | Who | When |
|---|---|---|---|
| H0 | **Rotate the Reolink admin password now**; it was public in git history. Then decide on history rewrite (`git filter-repo`) + force push, or make the repo private. | Abdul/Ishan | Today |
| H0b | Commit the four missing legacy modules (`pose_action_detector`, `lstm_action_classifier`, `video_recorder`, `byte_tracker_fixed`) to `legacy/` so agents can port their zone/recorder logic. | Ishan | Week 1 |
| H1a | **Quick capture** (`quick_capture` in `docs/DATA.md`): one Reolink at ~2.7 m, **simultaneous main + sub recording**, 30 min, 2 people. Staged pick/return/conceal takes, each with a **benign matched pair** (same motion, no concealment). Log `take_id, subtype, t_start, t_end` in real time. Feeds T03 baseline, T04 wrist PCK, T07 real-vs-emulated. | Team | Week 1 (before H1) |
| H1 | Build a mock aisle (shelf, candy rack, counter, door) and record `lab_mock_aisle` per `docs/DATA.md` protocol. | Team | Weeks 1–3 |
| H2 | Email RetailS authors re commercial license; check MERL Shopping license. RetailS stays unused until they reply. | Abdul | Week 1 |
| H6 | Decide the store appliance candidates for T12 to benchmark (e.g. Orin AGX vs small x86 + RTX 4000-class). Buy or borrow one. | Abdul + CTO | Before G1 |
| H3 | Decide Ultralytics enterprise license vs. Apache-only stack (T03/T04 produce the comparison). | Abdul + CTO | After G1 |
| H4 | Privacy/biometric counsel for the pilot store's jurisdiction; signage text. | Abdul | Before store |
| H5 | Confirm which Reolink models are installed at the store (lets T07 mark profiles `measured`). | Store visit / franchisee | ASAP |

## Hand-off package for the CTO (end state)

- Green G1 + G2 reports in `runs/eval/` (summaries copied into `docs/reports/`).
- ADRs for every deviation from `docs/ARCHITECTURE.md`.
- Known-gaps list: license swaps outstanding, DeepStream port decision, fleet/OTA,
  cloud review UI hardening, security review (RTSP on store LAN, device hardening).
