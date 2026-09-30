# Evaluation spec

The eval harness (T09) is the product's source of truth. Anything that changes
detection behavior must report these numbers. Run: `python -m eval.run --suite <name>`.
Output: `runs/eval/<git-sha>/<suite>.json` + `summary.md`.

## Suites

| Suite | Data | Purpose |
|---|---|---|
| `public_pose` | PoseLift test (RetailS only once licensed) | Pose-sequence model comparability with published baselines. Not a system-level validation. |
| `sim_matrix` | `sim_store_v*` × all camera profiles × mount heights | Camera selection and robustness |
| `emu_matrix` | `lab_mock_aisle` re-rendered through T07 per profile | Real-footage robustness to camera/codec/lighting |
| `lab_e2e` | `lab_mock_aisle` held-out actors, full pipeline from video file | End-to-end system metric (the one that gates releases) |
| `soak` | 24 h replay of normal-only footage at real time, 10 streams | Stability, false alerts per camera-hour, memory growth |
| `sim_transfer` | Train on {real/mock only, sim only, real+sim}; test on the same untouched real test set | Decides whether sim data earns more investment |
| `perf` | Synthetic 10-stream load | Throughput, latency, GPU/NVDEC utilization |
| `meva_interaction` | MEVA indoor, held-out cameras/sites: `picks_up`, `puts_down`, `transfers`, `steals_object` events | **Free real-footage proxy** for event recall at the FA budget, through the full streaming pipeline |
| `meva_fa` | ≥ 100 camera-hours of continuous MEVA indoor video | False alerts per camera-hour on real continuous footage (proxy until store shadow) |
| `smartspaces_track` | SmartSpaces retail scenes (held-out scenes) | Detection mAP, IDF1, ID switches on overhead retail views |

## Metrics

**Primary (release gate):**
- **Event recall @ false-alert budget:** fraction of labeled theft events that produce
  an `Alert` whose window overlaps the event, while false alerts ≤ **1 per camera per
  10 open-hours**. That's about 10/day for a 10-cam store before review. Report the
  full recall-vs-FA curve too.
- **Precision after verifier** (T11 on): fraction of alerts reaching review that are true.

**Secondary:**
- Frame-level AUC-ROC / AUC-PR on `public_pose` (compare to STG-NF 67.5 PoseLift, 63.2 RetailS-real).
- Per-subtype recall (pocket, bag, waistband, jacket, grab_run).
- Tracking: IDF1, ID switches per person-minute (the legacy demo shows ~2 per 10 s).
- Journey accuracy: checkout-visit and store-exit event F1 (these feed the strongest rule).
- Robustness: worst-profile recall on `emu_matrix` / `sim_matrix` ÷ best-profile recall.
- Perf: p95 event→alert latency, sustained fps per stream, VRAM, NVDEC %.
- Soak: crashes (must be 0), RSS growth < 5%/24 h, reconnect success after forced RTSP drop.

## Gates

| Gate | When | Must show |
|---|---|---|
| **G1 — free data** | End of Wave 2 | `meva_interaction` recall + CI at the FA budget measured on `meva_fa`; `smartspaces_track` IDF1 reported; behavior model ≥ STG-NF on `public_pose`; `sim_transfer` result for sim v0; 10 MEVA replay streams for 24 h with 0 crashes. All labeled "proxy, not retail". |
| **G2 — lab** | After `lab_mock_aisle` recorded | `lab_e2e` recall ≥ 0.6 at the FA budget on held-out actors; soak 24 h clean; camera recommendation written from `emu_matrix` + `sim_matrix`. |
| **G3 — store shadow** | In store, 2+ weeks | Alerts reviewed but NOT sent. Reviewers also check a random sample of **non-alerted** footage to estimate misses. Measured FA/day, recall on known incidents, review minutes/day. Acceptable review burden agreed with the operator **before** enabling live notifications. |

Numbers in G2 are starting targets. Adjust by ADR with evidence, not by vibe.

## Rules

- Splits are by **actor and clip**, never by frame. Held-out actors never appear in training.
  All derivatives of a clip (crops, overlapping windows, emulated variants, augmentations)
  share a `group_id` and stay in one split.
- No threshold tuning on the test split. Calibrate on val and freeze. Site-specific
  calibration is reported separately from generalization results.
- **False alerts per hour are measured on continuous footage only**, never on curated clip sets.
- **End-to-end suites run the deployed streaming path** (sampling, causal smoothing, track
  resets, dedupe, alert suppression) from video files, not saved model scores.
- Report sample counts and 95% confidence intervals (bootstrap over clips/actors) for
  every headline metric. With few events, say so.
- Only report metrics the labels support; otherwise "unavailable".
- Positives for detection metrics are events labeled `visible: observed` for that camera.
- Every report records: git sha, model ids, dataset versions, camera profiles, GPU, driver.
- Sim/emulator results never count as production validation on their own. Only the real
  held-out set and store shadow mode do.
