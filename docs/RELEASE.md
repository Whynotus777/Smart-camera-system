# Release status

Owner: T13 (integration & release). Updated 2026-09-30. Evidence for M1 is in
[`docs/reports/T13-m1.md`](reports/T13-m1.md).

## Milestones

| Milestone | Status | Blocking |
|---|---|---|
| **M1 walking skeleton** | **Acceptance met on the T13 branch; in review.** Stand-ins behind the §4 interfaces; real components swap in from Wave 1/2. | Merge, then B1–B3 below (they don't block M1 itself) |
| **M2 installable** | Not started. | T12 (Wave 3), appliance choice (H6), T02 health events, see "Blocking M2" |
| **M3 shadow-ready** | Not started. | Access control, retention, credentials, review-burden metrics (all listed as risks below) |

## M1 acceptance (tasks/T13-integration-release.md)

- [x] **One command runs the whole flow headless and prints the review URL.**
  `scs demo --video demo_1.mp4` (today: `python -m scs.app demo --video demo_1.mp4` until the
  `scs` entry point lands, see Handoffs). File sources loop like a camera; `--rtsp-env VAR`
  runs the same flow from an RTSP camera or T14's MEVA farm.
- [x] **kill -9 / restart test passes 20 runs in a row.** `tests/integration/m1_chaos.py --runs 20`:
  random SIGKILLs of every role and of the ingest's ffmpeg decoder, plus self-kills at every
  durable boundary (crash points). Every run is compared with an uninterrupted reference run.
- [x] **Clip plays in a browser and covers ≥ 10 s before / ≥ 5 s after.** H.264 yuv420p MP4 with
  faststart, served with HTTP Range. Played and seeked in Firefox 155 (headless, WebDriver).
  Coverage is asserted for every clip in every test run.
- [x] **Review outcome stored and exported in the canonical label format.** One outcome per
  alert (a retry is idempotent, a contradiction is 409). Labels validate against T09's
  `eval.canonical.ClipLabels` as well as DATA.md's minimum.

Also verified: RTSP camera drop, RTSP server death, and ingest kill -9 (reconnect in 2–6.5 s);
kill -9 of the `demo` supervisor (no orphans, resumes); a GPU variant (NVDEC + NVENC); and M1 on
real MEVA footage served by T14's replay farm.

## How to run

```bash
uv pip install -e ".[dev]"                     # plus the ffmpeg/ffprobe binaries
python -m scs.app demo --video demo_1.mp4      # prints "Review queue: http://127.0.0.1:8765/"
SCS_DEMO_RTSP=rtsp://127.0.0.1:8554/meva/h264/cam01 python -m scs.app demo --rtsp-env SCS_DEMO_RTSP
python -m scs.app inject-event --workdir runs/demo/<ts>   # an event without the detector
python -m scs.app export-labels --workdir runs/demo/<ts>

pytest -m "not gpu and not data and not slow" tests/integration        # CI (≈ 5 min)
pytest -m slow tests/integration                                        # 20x chaos, RTSP (≈ 1 h)
scripts/gpu shared -- pytest -m gpu tests/integration                   # 5090 only
python tests/integration/m1_chaos.py --runs 20 --out runs/chaos20.json # acceptance evidence
```

## Release checklist (every merge to `main`)

1. Merge order holds (below). The PR is rebased on current `main`.
2. CI green: `pytest -m "not gpu and not data and not slow"`, which includes `tests/integration`.
3. A PR touching `src/scs/app`, `ingest`, `perception`, `events`, `evidence`, `bus` or `contracts`:
   run 20x chaos on the PR head and paste the summary line.
4. A PR touching GPU paths: `scripts/gpu shared -- pytest -m gpu tests/integration`, output pasted.
5. `tests/integration` itself is unchanged, or changed in the same PR with a reason. The M1
   invariants (`m1_harness.check_invariants`) are the contract. They're never loosened to make a
   swap pass.
6. `CONTRACTS_VERSION` is unchanged, or there's an ADR in a contracts-only PR (AGENTS.md rule 3).

## Merge order and Wave 1 integration status

ROADMAP order: **T13 → T14 → T09 → T02 → T03 → T04 → T07 → T08**. Checked by merging each open
stack on top of this branch and running every non-slow test (2026-09-30):

| PR | Stack | Merges cleanly on T13 | Tests | Integration notes |
|---|---|---|---|---|
| #4–#7 | T14 data factory, fetch, replay farm, dedup | yes | pass | The farm serves M1 over RTSP. It's a shared global resource (risk R6). |
| #8–#10 | T09 metrics, canonical format, suites + e2e driver | yes | pass (183 passed together with T13 + T14) | Labels aligned to `ClipLabels` (R2). The e2e driver runs `scs.app.pipeline:factory`, but its source timestamps break every time-based rule (B1). |
| — | T02, T03, T04, T07, T08 | no PR yet | — | Swap points: `scs.app.pipeline.M1Pipeline` (T03/T05), `FfmpegSource` → T02 sources, `EvidenceStore` → T10 |

## Blockers (decisions needed; the affected work is paused)

- **B1: which clock do time-based rules use in file replay? (correctness, cross-agent)**
  `FrameRef.ts` is decode wall clock. In unpaced replay (T09's e2e default), 20 s of video
  decodes in about 1 s, so a 5 s dwell, a journey timeout or a behavior window never elapses.
  Recall comes out wrong with no error: measured 0 events vs 1 on the same clip.
  **Proposal:** every file/sim source sets `source_ts` = origin + container PTS (already a valid
  contract meaning: capture time), and time logic measures durations on `source_ts ?? ts`.
  M1's engine already does this. Needs sign-off from T02 (FileSource), T09 (VideoFileSource,
  e2e) and T05 (journey engine), and possibly an ADR clarifying the convention.
- **B2: `scs` entry point and CI ffmpeg (AGENTS.md rule 2).** Both live in T00-owned files; see
  Handoffs. Until then the command is `python -m scs.app`, and CI *skips* the M1 tests with a
  visible reason when ffmpeg is missing.
- **B3: how does a *dismissed* review appear in labels? (interface, T09)** Today it's
  `events: []` + `supports.events: true` + `extra.review.decision = "dismissed"`, i.e. a reviewed
  hard negative. T09 to confirm that metrics treat it that way, or define a field.

## Open risks (ranked)

| # | Risk | Owner | Mitigation / next step |
|---|---|---|---|
| R1 | Timestamp convention in replay (B1): silently wrong eval numbers | T02/T05/T09 | Decide B1. Add a T09 e2e smoke test that asserts a known dwell fires. |
| R2 | Canonical label drift: DATA.md says `visible` is "per camera"; T09's model takes a scalar | T09 | M1 emits the scalar; T09 to reword DATA.md. `tests/integration` validates with `ClipLabels` once T09 merges. |
| R3 | **Storage latency**: SQLite `synchronous=FULL` on this box's HDD pushed review POSTs to > 10 s under IO saturation (MEVA download + chaos). Correct, but slow. | T12 | Edge box on SSD/NVMe; T12 to set a review-latency budget and measure it in the soak. |
| R4 | **H.265 cameras**: remuxed HEVC clips don't play in most browsers. Reolink main streams often default to H.265. | T10 | Transcode evidence clips to H.264 at export when the source is HEVC (NVENC), or record the H.264 sub-stream for review. |
| R5 | `FrameSource` image contract: M1's source yields small gray frames; the contract says HxWx3. | T02 | `M1Pipeline` normalizes any image (numpy or tensor). T02's sources plug in unchanged. |
| R6 | T14's replay farm is one global resource (port 8554, fixed container name, `fault server`, `down`): one agent's fault test breaks everyone's streams. | T14 | Asked for port/name overrides. T13's automated RTSP tests use a private mediamtx on a free port. |
| R7 | Credentials: the RTSP URL is in ffmpeg's argv (visible in `ps`) and may appear in ffmpeg error output. Config stores only `env:NAME`. | T02 / M3 | Pass credentials via a URL-less mechanism or redact logs; M3 item. |
| R8 | The review page has no auth (bound to 127.0.0.1); clips are never deleted; live segments are pruned after 120 s. | T10 / M3 | Access control and retention policy are M3 deliverables. |
| R9 | No `dwell` EventType: M1 emits `zone_enter` + `data.rule = "dwell"`. | T05 | T05 decides. Adding an enum value needs an ADR. |
| R10 | Live clip boundaries map stream time to wall clock at first frame (± decode latency, < 0.5 s). The 15 s / 10 s windows absorb it. | T10 | The real ring buffer uses per-packet timestamps from T02's packet tap. |
| R11 | Reconnect after a camera drop takes 5–6.5 s (backoff cap 5 s). | T02 | T02 owns reconnect and HEALTH events. |
| R12 | One SQLite writer shared by all roles: fine for 1 camera, unmeasured at 10. | T12 | Include in the 10-stream soak. |
| R13 | **The dev box's disk is a shared, unscheduled resource.** On 2026-09-30, parallel torch/TensorRT installs into per-worktree venvs plus the MEVA fetch drove load average to 57, with processes blocked on I/O. `scripts/gpu` serializes the GPU, not the disk, so timing, soak and latency numbers taken meanwhile are contaminated. | owner / T12 | Share one venv/wheel cache across worktrees, or pin `UV_CACHE_DIR`; take perf/soak numbers only when the disk is quiet, and record `iostat` next to `nvidia-smi`. |

## Blocking M2

M2 needs T12 (Wave 3): packaging, a supervisor that replaces `scs demo`, config/model rollback,
and explicit degraded states for camera loss, network loss, process crash and disk full. The
signals for those states come from T02's HEALTH events (camera/network), this branch's
supervisor restarts (process), and disk checks that don't exist yet (T12). H6 (appliance
choice) blocks the "clean target machine" test. Nothing in M1 blocks M2.

## Handoffs (changes needed outside T13's paths)

- **T00 files / owner:** add `[project.scripts] scs = "scs.app.cli:main"` to `pyproject.toml`.
  Add `sudo apt-get install -y ffmpeg` to `.github/workflows/ci.yml` before pytest (the M1 CI
  tests skip without it).
- **T02:** `FileSource` sets `source_ts` from PTS (B1). The packet tap should carry per-packet
  timestamps for T10. HEALTH events on reconnect.
- **T09:** `VideoFileSource` sets `source_ts` (B1); confirm B3; reword `visible` in DATA.md (R2).
- **T10:** keep the review API paths in `scs.app.web` (the tests drive them), implement
  `scs.app.evidence.EvidenceStore`, and handle HEVC (R4).
- **T14:** port/container-name overrides for the farm, and a note in its README that
  `fault server` / `down` affect every consumer (R6).
