# Data & model registry

Every dataset, pretrained weight, and generated corpus used anywhere in this repo
gets a row here **before** it's used. `Use` column:
**prod** = may ship in customer builds · **R&D** = training experiments / eval only,
must be swappable · **blocked** = don't use until the open question is resolved.

Local layout (gitignored): `data/<dataset_id>/{raw,converted}/`, `models/<model_id>/`.
Converters output the canonical format below (T09 owns).

## Canonical format

- `data/<id>/converted/tracks/<clip_id>.parquet`: one row per (frame, track), columns
  `camera_id, frame_idx, ts, track_id, x1, y1, x2, y2, score` + `kp_0_x … kp_16_c` (COCO17).
- `data/<id>/converted/labels/<clip_id>.json`: `{clip_id, camera_profile, fps, label_source, events:[...]}`. Each event has:
  - `type`: `EventType` name from contracts, or an **interaction label**:
    `item_pickup`, `item_returned`, `item_to_basket`, `item_to_bag`, `item_to_clothing`
    (subtype: pocket/waistband/jacket), `obscured_interaction`, `exit_no_checkout`, `grab_run`.
  - `track_id`, `t_start`, `t_end`, `actor_id` (if known).
  - `visible`: `observed` | `partially_observed` | `not_observed` **per camera**. The
    scripted truth (what happened) and the camera evidence (what this camera could see)
    are separate. Only `observed` events count as positives for that camera's
    detection metrics; the rest are reported separately.
  - `label_source`: `human` | `script` (sim/staged take log) | `model` (pre-label, never used as test truth).
- Labels only claim what the dataset supports. If a dataset has no actor IDs, subtypes,
  or journeys, those fields are `null` and metrics needing them report "unavailable".
- `data/<id>/converted/splits.json`: train/val/test by **actor, clip, and camera**,
  never by frame. **Derivative groups**: an original clip and every crop, overlap window,
  emulated variant, or augmentation of it share a `group_id` and always land in the same split.

## Public datasets

| id | What | Size | Modality | License | Use | Why we want it |
|---|---|---|---|---|---|---|
| `poselift` | Real US store, 6 ceiling cams, 1080p 15 fps | 155 clips, ~1.06 h, 43 theft events | Pose only (COCO17 via HRNet, interpolated + 8-frame smoothed), boxes, IDs, frame-level labels | Repo is Apache-2.0, but the data is hosted separately on Google Drive (linked from the README) with no license file of its own seen so far (T14, 2026-09-29); confirm with the authors that Apache-2.0 covers it | R&D → prod once confirmed | One benchmark for pose-sequence models. It does **not** validate our decoding, detector, or pose accuracy. Match its normalization, fps, confidence handling, and smoothing before comparing numbers. |
| `retails` | Live US store, 10 days, 6 cams | ~20M normal frames, 898 staged + 53 real thefts | Pose only | **None stated = not licensed**; emailing authors (nrashvan@charlotte.edu) | **pending**: don't use until authors grant permission | Scale; real-vs-staged split shows the domain gap (STG-NF 87.2 staged → 63.2 real AUC) |
| `meva` | Real multi-camera surveillance, 38 RGB+IR cams, indoor/outdoor, scripted actors | **328 h / 516 GB public on AWS** (`aws s3 ls --no-sign-request s3://mevadata-public-01/`); activity annotations from the MEVA data repo (Kitware); 37 activities incl. `person_picks_up_object`, `person_puts_down_object`, `person_transfers_object`, **`person_steals_object`** | Video + boxes + activity spans (no keypoints) | CC-BY-4.0 ([license](http://mevadata.org/resources/MEVA-data-license.txt)), attribution required | prod | **Free real test proxy** (object-interaction events), continuous indoor footage for false-alerts-per-hour, self-supervised domain adaptation, fake-camera replay |
| `smartspaces` | NVIDIA PhysicalAI-SmartSpaces (AI City Challenge 2024/2025 MTMC), synthetic Isaac Sim scenes incl. **retail**, warehouse, hospital | 250 h, ~1,500 cams, 1080p30 H.264, 6.7 TB total: **download retail scenes only**. Retail = `MTMC_Tracking_2024/test/scene_071`–`080` only (README + visual check by T14: 001–070 warehouse, 081–090 hospital; 072 is the storage room). 160 camera videos, 34.3 GB without depth maps; ground truth published | Video + time-synced 2D/3D boxes, global IDs | CC-BY-4.0 | prod | Overhead retail person detection, tracking, cross-camera association; reference for T08 scene/pipeline |
| `simuletic_sample` | Synthetic overhead retail shoplifting (free Kaggle sample of a paid set) | 8 videos, 400 images | Video + boxes + 17-kp pose + captions | **CC BY-NC-SA 4.0** (verified via Kaggle API, 2026-09-29): non-commercial + share-alike. Paid set **not** approved (zero-spend). Download needs a Kaggle account token | **blocked** until the owner OKs NC-SA use for eval (then R&D eval only, `[license-risk]`) | Sanity check for the behavior model on theft-like synthetic clips |
| `merl_shopping` | Overhead cam, mock grocery | 106 × ~2 min | Video; reach/retract/hand-in-shelf/inspect labels | **Check MERL license page** | R&D until checked | Shelf-interaction detector (hand-in-shelf) from an overhead view |
| `ucf_crime` | Surveillance clips incl. "Shoplifting" | ~50 shoplifting videos | Video, weak labels | Research use | R&D (eval only) | Hard negatives/positives for VLM verifier eval |
| `coco_kp` | COCO keypoints | 250k people | Images | Annotations CC-BY-4.0; images various Flickr licenses | prod (weights) | Pose pretraining standard |
| `ntu_rgbd` | 120 actions, lab | 114k clips | Video/skeleton | **Non-commercial** | R&D only; never ship weights trained on it | Skeleton action pretraining experiments |

### Local copies (T14 data factory)

Fetched with `python -m data_ops.fetch <id>` into the shared data root (`data_ops/paths.py`:
`$SCS_DATA_ROOT`, else the main checkout's `data/`, which every worktree shares). Each
`data/<id>/MANIFEST.json` lists every file with bytes, sha256, source URL, source checksum and
how it was verified. Attribution text for CC-BY sources is in each manifest.

| id | Command | What's local |
|---|---|---|
| `meva` | `fetch meva --indoor --max-gb 150` | 10 indoor cameras (G326, G329, G331, G508, G299, G330, G419, G420, G421, G423): all 919 clips with Kitware activity annotations + 477 unannotated continuous clips; 1,396 × 5 min ≈ 116 h, 150 GB; S3 ETag (MD5) verified. Annotations: meva-data-repo @ `421841a` (sets `kitware`, `kitware-meva-training`). Only **5** `person_steals_object` events exist in all of MEVA (all indoor). |
| `smartspaces` | `fetch smartspaces` | Retail scenes 071–080, 350 files, 34.3 GB, sha256 verified against HF LFS. |
| `coco_kp` | `fetch coco_kp` | val2017 images + `person_keypoints_val2017.json` (1.07 GB); size-verified (COCO publishes no checksums). |
| `poselift` | `fetch poselift` | Google Drive folder via `gdown`; sha256 recorded (Drive publishes none). |

Approval status per item lives in this table and covers code, data, weights, and assets
separately. Only `prod` and approved `R&D` items may feed models whose lineage could
reach a customer build.

## Don't use (any purpose)
- YouTube/TikTok CCTV shoplifting compilations and Kaggle/Roboflow sets scraped from them: no license, identifiable real people.
- Any dataset withdrawn for privacy reasons (e.g. DukeMTMC).

## Free-data plan while there are no cameras (zero-spend)
| Need | Free source now | Later (real) |
|---|---|---|
| Real test set for interaction events | `meva` object-interaction events (held-out cameras/sites) | `quick_capture`, store archive |
| False alerts per hour on continuous footage | `meva` indoor continuous video (hundreds of camera-hours) | store shadow mode |
| Retail overhead detection/tracking | `smartspaces` retail scenes | store footage |
| Theft-specific positives | `poselift` (pose), T08 sim v0, T14 generated clips | staged/real |
| Fake cameras for ingest/soak | `meva` replayed as RTSP (T14) | real Reolinks |

`meva` is not retail and its "steal" is scripted, so its numbers are a proxy. Reports must say so.

## Our own data (when available; optional, $0)

| id | What | Status | Owner |
|---|---|---|---|
| `quick_capture` | 30 min, one Reolink at ~2.7 m, simultaneous main + sub streams, 2 actors, staged pick/return/conceal + benign matched pairs, take log (ROADMAP H1a) | **To record, week 1** | Abdul / Ishan |
| `lab_mock_aisle` | Staged footage in a mock c-store aisle (shelf, candy rack, counter, door) with 2–4 Reolinks at 2.4–3.0 m, main + sub streams recorded | **To record, week 1–3** (human task H1) | Abdul / Ishan |
| `sim_store_v*` | Isaac Sim synthetic clips across camera profiles | T08 | agent |
| `emu_*` | Camera-emulated variants of the above | T07 | agent |
| `store_shadow` | Real store, shadow mode, later | After signage/consent | human |

Staging protocol for `lab_mock_aisle` (so labels are cheap and useful):
- ≥ 8 different actors, varied clothing (include dark hoodies, bags, jackets).
- Per session: 70% normal (browse, pick-inspect-return, pick-to-counter, phone use,
  crouch to low shelf, restock as "staff"), 30% theft across subtypes above.
- **Matched pairs**: for each concealment take, record the benign twin with the same
  actor, clothing, item, and position (item to basket, own phone out of pocket, item
  returned). Otherwise models learn the actor or the outfit instead of the action.
- Record 2+ hours of **continuous normal activity** (no script) for false-alert-per-hour
  measurement.
- Script each take and log `take_id, subtype, t_start, t_end` on a tablet in real time.
  That log *is* the label file.
- Record day, dusk, and IR-night lighting. Record at least one session with the
  camera at 2.4 m and one at 3.0 m.
- Consent forms from every actor; footage stays on the workstation.

## Pretrained models

| id | License | Use | Notes |
|---|---|---|---|
| Ultralytics YOLO (v8/11) detect & pose | **AGPL-3.0** (enterprise license available) | R&D | Great velocity; must be licensed or swapped before customer deployment. |
| boxmot trackers | **Verify** (believed AGPL-3.0) | R&D | Prefer MIT ByteTrack/OC-SORT reference impls for prod. |
| RTMDet / RTMPose (OpenMMLab) | Apache-2.0 | prod candidate | Top-down pose on crops; TRT export supported. |
| RT-DETR / D-FINE family | Apache-2.0 (check each repo) | prod candidate | Detector alternative. |
| MediaPipe Pose | Apache-2.0 | retire | Single-person, CPU-bound, frontal bias. |
| STG-NF | check repo | R&D baseline | Best baseline on PoseLift/RetailS. |
| NVIDIA TAO models (PeopleNet, ActionRecognitionNet, PoseClassificationNet) | per NGC model card (check each) | R&D → prod after review | Starting weights; trained partly on synthetic people. |
| Self-supervised video backbones (VideoMAE-family, V-JEPA-family) | per repo (check each) | R&D → prod after review | T06 continues pretraining on `meva` without labels. |
| Wan 2.2 TI2V-5B (image→video) | Apache-2.0 | R&D; generated clips are `label_source: script` | ~9 min per 5 s 720p clip on a 24 GB GPU. Hard negatives + variety only (T14). |
| NVIDIA Cosmos-Transfer (sim→photoreal with control maps) | NVIDIA Open Model License (check) | R&D | Keeps sim labels while changing appearance; may exceed 32 GB → test with offload, no cloud without OK. |
| VLMs for T11 (e.g. Qwen-VL family, NVIDIA Cosmos Reason) | per model (Apache-2.0 / NVIDIA Open Model License) | R&D → prod after review | Must fit in 32 GB alongside the pipeline or run on a separate box. |
| SMPL body model (if used for mocap retargeting in T08) | **Non-commercial** unless licensed | R&D only | Relevant if we convert staged video to 3D motion for sim. |
