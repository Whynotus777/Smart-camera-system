# T04 — Pose on high-resolution crops: report

Branch `agent/T04-pose` (from `wave0-baseline`). GPU: RTX 5090 (sm_120), driver 580.173.02,
torch 2.11.0+cu128, TensorRT 10.16.1.11 (pip `tensorrt-cu12`). Raw JSON: `runs/T04/`
(gitignored; regenerate with the commands at the end).

## Metric (stated once, used everywhere)

**Wrist PCK@0.2, normalized by torso diameter.** A predicted wrist is correct when
`||pred − gt|| ≤ 0.2 × torso diameter`, where the torso diameter is the **ground-truth**
distance left shoulder (COCO kp 5) ↔ right hip (kp 12). Instances without both torso
points labeled are excluded and counted. Left and right wrists (kp 9, 10) are pooled. A
wrist counts if it's labeled (v ≥ 1), and results are also split visible (v = 2) vs
occluded (v = 1). 95% CIs come from 1000 bootstrap resamples over **images**. Ground
truth is human annotation only. No model output is used as truth anywhere.

## Models

| Model | Role | License | Engine |
|---|---|---|---|
| RTMPose-m SimCC, body7, 256×192 (OpenMMLab SDK ONNX `e48f03d0`) | **primary** | code Apache-2.0; **weights R&D pending lineage review** (body7 includes research-only datasets, see HANDOFF.md §2) | FP16, profile batch 1..32, ONNX sha256 `5c0a4bf6…`, engine sha256 `f745eb93…` |
| YOLO11m-pose (Ultralytics 8.3, `yolo11m-pose.pt` sha256 `29b17eaf…`) | comparison, **[license-risk]** | AGPL-3.0, R&D only | FP16, batch 1..32, same 256×192 crops |

Both run through the same `TopDownPoseEstimator` path: the same crop (track box ×1.25,
aspect-fixed, from the untransformed main-stream frame), the same batching, and the same
mapping back to frame pixels. The comparison measures the models, not the cropping.
YOLO11-pose is a multi-person model, so per crop we keep the candidate that maximizes
`conf × IoU(candidate, track box in crop)`.

## (a) COCO-keypoints val2017, person-crop (GT box) protocol

2,346 images, 4,472 person instances with a torso (1,880 excluded for no torso GT),
6,633 labeled wrists.

| Model | Crop source | Wrist PCK@0.2 | visible (v=2) | occluded (v=1) | all 17 kp |
|---|---|---|---|---|---|
| **RTMPose-m** | full res | **0.940** [0.933, 0.946] | 0.957 | 0.797 | 0.943 |
| RTMPose-m | ×1/3 (1080p → 360p) | 0.869 [0.858, 0.880] | 0.893 | 0.674 | 0.899 |
| RTMPose-m | ×1/4 (1440p → 360p) | 0.796 [0.783, 0.809] | 0.822 | 0.582 | 0.846 |
| YOLO11m-pose | full res | 0.864 [0.853, 0.874] | 0.890 | 0.652 | 0.853 |
| YOLO11m-pose | ×1/3 | 0.781 [0.767, 0.795] | 0.813 | 0.524 | 0.800 |
| YOLO11m-pose | ×1/4 | 0.720 [0.704, 0.734] | 0.750 | 0.469 | 0.758 |

**RTMPose-m beats YOLO11m-pose by 7.6 pts** on wrists at full resolution (CIs don't
overlap) and by 7.6 pts at ×1/4. It stays the primary model.

### Main stream vs sub stream (ARCHITECTURE D1), by person height

"Sub" = the whole image downscaled (INTER_AREA), the scaled box cropped from the small
image, and keypoints scaled back up. Same model, same box, fewer pixels. Binned by the
person's box height at full resolution. RTMPose-m wrist PCK@0.2:

| Person height (full-res px) | n wrists | full | ×1/3 | ×1/4 |
|---|---|---|---|---|
| < 64 | 272 | 0.879 | 0.618 | **0.335** |
| 64–128 | 1,901 | 0.918 | 0.774 | **0.620** |
| 128–256 | 2,264 | 0.939 | 0.895 | 0.853 |
| ≥ 256 | 2,196 | 0.965 | 0.956 | 0.947 |

The sub-stream penalty is almost entirely on people under ~128 px (sub-stream ≤ 32 px
tall), which is where D1 predicted it.

### Weighted to an overhead-retail person-size distribution (SmartSpaces GT boxes)

SmartSpaces MTMC_Tracking_2024 retail scenes 071–074 (synthetic Isaac Sim, CC-BY-4.0):
526,497 **ground-truth** person boxes (every 30th frame, 64 cameras). The heights at
1080p are p10 76, p25 130, p50 204, p75 286, p90 356 px. The per-bin COCO results above
are re-weighted to that height distribution (heights ×1440/1080 for a 2560×1440 main
stream at the same FOV):

| Model | 1080p main | 360p sub (÷3) | Δ | 1440p main | 360p sub (÷4) | Δ |
|---|---|---|---|---|---|---|
| **RTMPose-m** | **0.940** | 0.875 | −6.5 pts | **0.949** | 0.857 | **−9.2 pts** |
| YOLO11m-pose | 0.866 | 0.787 | −7.9 pts | 0.880 | 0.782 | −9.8 pts |

**Acceptance: full-res crops beat 360p sub-stream crops.** At a Reolink 4MP geometry,
wrist PCK is 0.949 on the main stream vs 0.857 on the sub stream: about 1 in 11 wrists
lost. Caveat: this is a **proxy**. COCO people are mostly frontal, eye-level views, not
overhead. Only the person-size distribution comes from overhead retail. The final number
needs `quick_capture` simultaneous main+sub footage with human-annotated wrists.

## (b) Sim ground truth (by camera profile and mount height): **unavailable**

No simulation keypoint ground truth exists yet. T08 hasn't landed, and SmartSpaces ships
only 2D/3D boxes (checked `ground_truth.txt` and `ground_truth_2025_format.json`), no
keypoints. Per the brief, another model's output doesn't count as ground truth, so this
section stays empty. `pose_eval.pck_hits` / `summarize` take any (pred, GT) pairs, so
when T08 exports 2D keypoints per camera profile and mount height, the numbers can be
filled without code changes.

## (c) Human-annotated `quick_capture` wrists: **unavailable** (data not recorded, H1a).

## Throughput / latency (`scripts/gpu exclusive`)

`nvidia-smi` at start: `RTX 5090, 580.173.02, 940 MiB used, 1 % util` (exclusive lock
acquired 2026-09-30T01:46). Scenario: 10 cameras × 2560×1440 frames resident on the GPU
and pose at 10 Hz, so each 100 ms tick carries rate/10 crops across the 10 cameras,
micro-batched into **one** backend call (`estimate_many`, D9). Tick latency covers GPU
crop → TRT FP16 → decode → map back → `Pose` objects, synchronized. Paced in real time
for 60 s (601 ticks) per rate.

| Model | Rate | Crops / batch | Achieved | p50 | p95 | p99 | max | Ticks over 100 ms |
|---|---|---|---|---|---|---|---|---|
| **RTMPose-m** | 30/s | 3 | 30.0/s | 5.10 ms | 5.46 | 6.14 | 12.91 | 0 |
| **RTMPose-m** | 100/s | 10 | 100.1/s | 7.68 ms | 10.13 | 11.32 | 14.47 | 0 |
| **RTMPose-m** | **300/s** | 30 | **300.4/s** | **8.45 ms** | **12.15** | **12.70** | 13.79 | **0** |
| YOLO11m-pose | 30/s | 3 | 30.0/s | 6.33 ms | 6.89 | 7.69 | 9.13 | 0 |
| YOLO11m-pose | 100/s | 10 | 100.2/s | 8.98 ms | 11.66 | 12.70 | 13.71 | 0 |
| YOLO11m-pose | 300/s | 30 | 300.4/s | 9.78 ms | 13.40 | 13.74 | 14.05 | 0 |

Back-to-back peak (no pacing, one sync per 200 batches): RTMPose-m 3.33 ms per batch of
32 = **9,596 crops/s** (b1 0.95 ms, b8 2.05 ms, b16 2.77 ms); YOLO11m-pose 4.16 ms per
32 = 7,687 crops/s.

**Acceptance: 300 crops/s sustained with < 15 ms per batch. Met.** RTMPose-m p99 is
12.7 ms and the worst tick 13.8 ms. The margin is thin, and it isn't the model: the
paced tick (8.5 ms) is 2.5× the back-to-back batch (3.3 ms). The gap comes from the
per-tick sync, GPU clocks dropping between ticks, one `grid_sample` per camera frame,
and building 30 pydantic `Pose` objects in Python. If T12 needs headroom, move `Pose`
construction off the timed path or build poses lazily. This is the dev GPU. The edge
target must be re-benchmarked (T12).

## Live path: causal smoothing and gap filling

- `LivePoseSmoother` (One-Euro, `min_cutoff=1.0 Hz`, `beta=0.7` per box-height/s,
  `d_cutoff=1.0 Hz`). The speed term is normalized by the person's pixel height, so one
  `beta` works near and far.
- Gap filling is bounded by **elapsed time** (≤ 0.5 s on the host monotonic clock,
  falling back to the decode wall clock), never by frame count. Tested at 5, 10 and 30 fps.
  Imputed keypoints hold the last smoothed position and their confidence decays linearly
  to 0 over the window. Every keypoint has an `OBSERVED | IMPUTED` mask. Raw and smoothed
  poses are both kept (`SmoothedPose`).
- Resets: new `epoch` (reconnect), `seq` going backwards, or a gap > 0.5 s.
- **Proof of causality** (`test_live_path_never_reads_future_frames`), run on the full
  live pipeline (gate → estimate → smooth → export): (1) outputs for frames ≤ k are
  identical when every later frame and track is replaced; (2) running on the stream
  truncated at k gives the same outputs; (3) each `step` returns before the next frame is
  pulled from the generator.
- Offline variant: `offline_interpolate_smooth` is **NON-CAUSAL** (linear interpolation
  across gaps + centered window), named and documented as labeling-only. The
  `to_poselift(mode="offline")` window is also non-causal; `mode="live"` is trailing
  (causal, test included).

## `to_poselift()` adapter: choices

Details are in the `pose_poselift.py` module docstring. Summary:

| Property | PoseLift | Adapter choice |
|---|---|---|
| Keypoints | COCO17 from HRNet on YOLOv8+ByteTrack boxes | same order; different model (RTMPose) |
| Coordinates | pixels of 1920×1080 | rescale from main-stream (16:9 enforced; others raise) |
| fps | 15 | linear-in-time resample onto a 15 fps grid from RAW poses; only times ≤ newest input |
| Missing poses | linear interpolation (limit unstated) | linear, only across gaps ≤ 0.5 s; longer gaps split the track |
| Smoothing | "8-frame window" (method unstated; assumed moving average) | 8-frame mean at 15 fps: centered for offline benchmark parity, trailing (causal, +0.23 s lag) for live |
| Confidence | HRNet heatmap max | passed through (RTMPose SimCC max, clipped to [0,1]). **Distributions differ**: re-tune STG-NF `seg_conf_th` on our val split |
| Model normalization | STG-NF `normalize_pose`: center on segment mean, divide by std(y) | `stgnf_normalize`, tested against the reference formula |

Open: PoseLift's pickle layout and whether its 8-frame smoothing is centered can't be
verified until the PoseLift files are downloaded (not on disk). The adapter output shape
is the README's per-frame `{person_id: {bbox XYWH, keypoints XYC}}`.

## Zone gating and crop export

- `gate_tracks`: only tracks whose box touches SHELF / HIGH_VALUE / CHECKOUT / ENTRY_EXIT
  zones expanded by 10%, via `scs.geometry.bbox_touches_zone(..., expand=0.10)`. No
  polygon math in T04. `lost` tracks are skipped by default.
- `CropExporter` (D11): person crop plus one square hand crop per confident wrist (side =
  0.5 × torso diameter, ≥ 32 px). **Off by default**, requires `source_kind` lab/sim, and
  store needs explicit `store_consent`. Writes go through a background thread with a
  bounded queue that drops when full, so export never blocks the live path. An
  `index.jsonl` row carries frame identity, keypoints, model id and `group_id` for T09
  splits.

## Frame identity (D10)

`TopDownPoseEstimator.check_inputs` raises `FrameIdentityError` if any track's
`(camera_id, epoch, seq)` differs from the frame being cropped, if the frame isn't the
untransformed main stream, or if the image size ≠ `FrameRef.width/height`. All three are
tested.

## Findings / known gaps

1. **Arms reaching outside the track box get cut off.** With the model-native 1.25 padding,
   a wrist more than ~0.6 box-widths beyond the box edge falls outside the crop. The T00
   fixture's shelf reach (1.4 box-widths from center) is one case. This matters most for
   the key event, a hand in a shelf. Next step: measure on T03 boxes and, if needed, widen
   the crop for SHELF-gated tracks (costs pixels per wrist). Tracked in HANDOFF.md §5.
2. Weights lineage: body7 is R&D until reviewed. A COCO-only RTMPose checkpoint would
   need an mmpose export environment (mmcv on sm_120 untested). Not done.
3. Hand keypoints (21-pt): **not worth it yet**. At the median SmartSpaces person height
   (~270 px at 1440p), a hand is ~25–30 px, so 21 points get ~1 px each and aren't stable.
   The D11 hand-region crops give T06 the same information at lower cost. Revisit if
   `quick_capture` shows hand crops at ≥ 64 px.
4. T03 isn't merged, so tests use T00 fixtures and synthetic scenes. The first integration
   against real T03 tracks will happen in T13.

## Reproduce

```bash
M=~/Smart-camera-system/models
python -m scs.perception.pose_trt $M/rtmpose/rtmpose-m_body7.onnx $M/rtmpose/rtmpose-m_body7_b32_fp16.engine --max-batch 32
python -m scs.perception.pose_trt $M/yolo11/yolo11m-pose.onnx   $M/yolo11/yolo11m-pose_b32_fp16.engine --max-batch 32
scripts/gpu shared -- python -m scs.perception.pose_eval --backend rtmpose --engine ... --model-id ... \
  --coco-images <val2017> --coco-ann <person_keypoints_val2017.json> \
  --smartspaces-gt data/smartspaces/raw/MTMC_Tracking_2024/test/scene_*/ground_truth.txt --out runs/T04/eval_rtmpose.json
scripts/gpu exclusive -- python -m scs.perception.pose_bench --backend rtmpose --engine ... --model-id ... --out runs/T04/bench_rtmpose.json
```
YOLO ONNX export (isolated venv, AGPL code kept out of the project env):
`YOLO('yolo11m-pose.pt').export(format='onnx', imgsz=(256,192), dynamic=True, simplify=True, opset=17)`.
