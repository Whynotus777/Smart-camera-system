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
| `poselift` | Real US store, 6 ceiling cams, 1080p 15 fps | 155 clips, ~1.06 h, 43 theft events | Pose only (COCO17 via HRNet, interpolated + 8-frame smoothed), boxes, IDs, frame-level labels | Repo is Apache-2.0; confirm it covers the data | R&D → prod once confirmed | One benchmark for pose-sequence models. It does **not** validate our decoding, detector, or pose accuracy. Match its normalization, fps, confidence handling, and smoothing before comparing numbers. |
| `retails` | Live US store, 10 days, 6 cams | ~20M normal frames, 898 staged + 53 real thefts | Pose only | **None stated = not licensed**; emailing authors (nrashvan@charlotte.edu) | **pending**: don't use until authors grant permission | Scale; real-vs-staged split shows the domain gap (STG-NF 87.2 staged → 63.2 real AUC) |
| `meva` | 38 RGB+IR cams, indoor/outdoor | 9,300 h collected, 144 h annotated, 37 activities incl. picks_up, puts_down, transfers, **steals_object** | Video + boxes (no keypoint labels) | CC-BY-4.0 | prod | Detector/tracker domain adaptation to CCTV; object-interaction pretraining |
| `merl_shopping` | Overhead cam, mock grocery | 106 × ~2 min | Video; reach/retract/hand-in-shelf/inspect labels | **Check MERL license page** | R&D until checked | Shelf-interaction detector (hand-in-shelf) from an overhead view |
| `ucf_crime` | Surveillance clips incl. "Shoplifting" | ~50 shoplifting videos | Video, weak labels | Research use | R&D (eval only) | Hard negatives/positives for VLM verifier eval |
| `coco_kp` | COCO keypoints | 250k people | Images | Annotations CC-BY-4.0; images various Flickr licenses | prod (weights) | Pose pretraining standard |
| `ntu_rgbd` | 120 actions, lab | 114k clips | Video/skeleton | **Non-commercial** | R&D only; never ship weights trained on it | Skeleton action pretraining experiments |

Approval status per item lives in this table and covers code, data, weights, and assets
separately. Only `prod` and approved `R&D` items may feed models whose lineage could
reach a customer build.

## Our own data (highest value, create now)

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
| VLMs for T11 (e.g. Qwen-VL family, NVIDIA Cosmos Reason) | per model (Apache-2.0 / NVIDIA Open Model License) | R&D → prod after review | Must fit in 32 GB alongside the pipeline or run on a separate box. |
| SMPL body model (if used for mocap retargeting in T08) | **Non-commercial** unless licensed | R&D only | Relevant if we convert staged video to 3D motion for sim. |
