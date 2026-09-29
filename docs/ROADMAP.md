# Roadmap — agent waves

Goal: before we're back in store, have a system that's **measured**: it runs on 10
streams, has behavior models trained on public + synthetic + staged data, and has a
data-backed camera recommendation. Then the CTO hardens it for deployment.

## Waves and dependencies

```
Wave 0 (serial, 1 agent, ~2 days)
  T00 contracts + skeleton + CI  ──┐   (contracts.py already drafted in this PR)
  T01 hygiene + legacy quarantine ─┤
                                   ▼
Wave 1 (parallel, up to 6 agents)
  T02 ingest ─────────┐
  T03 detect+track ───┤
  T04 pose ───────────┤
  T07 camera emulator ┤
  T08 sim (Isaac) ────┤   (separate env; long-running)
  T09 eval harness + dataset converters
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
  H1 lab mock-aisle recording (start week 1, runs in parallel with everything)
  → Gate G2 → CTO hand-off → store shadow → G3
```

## Human tasks (agents can't do these)

| id | Task | Who | When |
|---|---|---|---|
| H0 | **Rotate the Reolink admin password now**; it was public in git history. Then decide on history rewrite (`git filter-repo`) + force push, or make the repo private. | Abdul/Ishan | Today |
| H0b | Commit the four missing legacy modules (`pose_action_detector`, `lstm_action_classifier`, `video_recorder`, `byte_tracker_fixed`) to `legacy/` so agents can port their zone/recorder logic. | Ishan | Week 1 |
| H1 | Build a mock aisle (shelf, candy rack, counter, door) and record `lab_mock_aisle` per `docs/DATA.md` protocol. | Team | Weeks 1–3 |
| H2 | Email RetailS authors re commercial license; check MERL Shopping license. | Abdul | Week 1 |
| H3 | Decide Ultralytics enterprise license vs. Apache-only stack (T03/T04 produce the comparison). | Abdul + CTO | After G1 |
| H4 | Privacy/biometric counsel for the pilot store's jurisdiction; signage text. | Abdul | Before store |
| H5 | Confirm which Reolink models are installed at the store (fills `verified: true` profiles). | Store visit / franchisee | ASAP |

## Hand-off package for the CTO (end state)

- Green G1 + G2 reports in `runs/eval/` (summaries copied into `docs/reports/`).
- ADRs for every deviation from `docs/ARCHITECTURE.md`.
- Known-gaps list: license swaps outstanding, DeepStream port decision, fleet/OTA,
  cloud review UI hardening, security review (RTSP on store LAN, device hardening).
