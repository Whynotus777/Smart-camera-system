# T14 data factory: status report

Owner: T14 agent · Started 2026-09-29 · Branch `agent/T14-data-factory` (from `wave0-baseline`)

## 1. Disk budget

Measured with `python -m data_ops.budget` on 2026-09-29 before any T14 download:

| | |
|---|---|
| Filesystem | `/dev/sda7` (ext4), holds `/home` and the shared data root |
| Total | 5,896 GB |
| Free | 4,826 GB (81.8%) |
| Floor (15%, never go below) | 884 GB |
| **Spendable** | **3,941 GB** |

What the T14 plan needs:

| Item | Size | Notes |
|---|---:|---|
| `meva` indoor | 150.0 GB | cap from the brief; ≈ 116 h of video |
| `smartspaces` retail | 34.3 GB | scenes 071–080, no depth maps |
| `coco_kp` val | 1.1 GB | |
| `poselift` | < 1 GB | pose files only |
| Wan 2.2 TI2V-5B weights | 34.2 GB | `models/`, Apache-2.0 |
| Replay farm transcodes (H.264 + H.265, 10 clips) | ~10 GB | estimate |
| Generated clips (100 × 5 s 720p) + lineage | ~2 GB | estimate |
| Pre-labels | ~5 GB | estimate |
| **Total** | **≈ 240 GB** | **6% of spendable** |

Headroom is large. The next big candidates, if the plan grows, would be the rest of MEVA
indoor (+8.5 GB), outdoor MEVA (~317 GB), or SmartSpaces warehouse scenes (TBs); none is
planned. Every fetcher calls `data_ops.budget.require()` before starting and again per file.

## 2. Shared data root

Every agent works in its own git worktree, and `data/` is gitignored per checkout. So
`data_ops.paths.data_root()` resolves to **the main checkout's `data/`** via
`git rev-parse --git-common-dir`, from any worktree; `$SCS_DATA_ROOT` overrides.
On this box: `/home/quantumc1/Smart-camera-system/data/`. Other agents should read
datasets from there, not from their own worktree's `data/`.

## 3. Findings that affect other tasks

- **MEVA has almost no theft.** Across all official Kitware annotations there are 5,668
  `person_picks_up_object`, 5,146 `person_puts_down_object`, 1,199 `person_transfers_object`,
  and only **5 `person_steals_object`** events (all on indoor cameras: G421 ×2, G331 ×3). In
  the downloaded indoor set: 4,252 picks-ups, 3,927 put-downs, 821 transfers, 5 steals.
  `meva_interaction` (T09) can measure pick/put/transfer recall with real statistical power,
  but **not theft recall**; 5 events can't give a useful confidence interval. (T06, T09.)
- **MEVA has no indoor/outdoor flag.** Our indoor list comes from MEVA's camera-model
  README (G299/G330 gym) plus a one-frame visual check per camera: 10 of 28 RGB/IR
  cameras are indoor. The 4 IR cameras are all outdoor.
- **SmartSpaces "retail" is 10 scenes, all in the 2024 *test* split.** The 2025/2026
  releases are warehouse/hospital/lab only. The split is ours to redefine (T09), since the
  2024 test ground truth is published. 16 cameras per scene; `scene_071/camera_0649` is
  corrupt per the README.
- **Simuletic free sample is CC BY-NC-SA 4.0**, not just "Kaggle terms". Blocked (below).
- **PoseLift data lives on Google Drive**, separate from the Apache-2.0 GitHub repo.
  Treat it as R&D until the authors confirm the license covers the data.

## 4. Download status

Updated as downloads finish; exact numbers come from each `MANIFEST.json`.

| id | Status | Size | Hours | License | Verified by |
|---|---|---:|---:|---|---|
| `meva` | downloading | 150.0 GB target | ≈ 116 | CC-BY-4.0 | S3 ETag (MD5) + sha256 recorded |
| `smartspaces` | downloading | 34.3 GB target | see §5 | CC-BY-4.0 | HF LFS sha256 |
| `coco_kp` | downloading | 1.07 GB | n/a | annotations CC-BY-4.0; images per-Flickr | size only (no published checksum) |
| `poselift` | downloading (Drive throttled) | small | ~1.06 | Apache-2.0 repo; data coverage unconfirmed | size only + sha256 |
| `simuletic_sample` | **blocked** | 0.53 GB | | CC BY-NC-SA 4.0 | |

## 5. Blockers

1. **Simuletic sample:** (a) CC BY-NC-SA 4.0 means non-commercial, share-alike: R&D eval
   only under rule 5, and needs the owner's OK plus a `[license-risk]` tag. (b) Kaggle
   downloads need an account API token (`~/.kaggle/kaggle.json`), which this box doesn't
   have. T14 won't create accounts or handle credentials. Owner: OK the license for eval
   use and place a token, or drop the dataset.
2. **`tsp` (task-spooler) isn't installed** and installing it needs `sudo apt install
   task-spooler`. The overnight Wan generation needs it (AGENTS.md: long GPU jobs go
   through `tsp` + `scripts/gpu exclusive`).
3. **PoseLift data license:** needs an email to the authors (could fold into ROADMAP H2,
   the RetailS email to the same group).
