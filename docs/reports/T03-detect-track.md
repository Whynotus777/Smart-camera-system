# T03 — Person detection + tracking: comparison report

Status: acceptance **partially met** (see §4) · branch `agent/T03-detect-track` · RTX 5090 (sm_120), driver 580.173.02, TensorRT 10.16.1.11,
torch 2.11.0+cu128 · all GPU numbers taken under `scripts/gpu exclusive` · 2026-09-30

## TL;DR

- **Recommended production candidate: D-FINE-S fine-tuned on SmartSpaces + MEVA (`dfine-s-retail`,
  Apache-2.0, CC-BY training data) + the appearance-aware tracker (`AppearanceTracker`, config frozen
  from val tuning).** YOLO11s stays as a comparison only (AGPL).
- Latency, batch of 10 × 640 px from 2560×1440 frames (TRT, RTX 5090, exclusive GPU):
  **D-FINE 4.11 ms engine / 5.45 ms end-to-end** (BF16), YOLO11s 1.95 / 3.16 ms (FP16). Both < 10 ms.
  The fine-tuned D-FINE engine has the same graph and build recipe (benchmarked on the COCO engine).
- Dead or slow camera: healthy cameras' latency does not rise (D-FINE p95 27.4 ms all alive →
  22.7 ms one dead / 23.0 ms one at 1 fps; bounded by the 20 ms deadline + one batch).
- Detection: COCO weights: YOLO11s is ahead on synthetic SmartSpaces (AP50 0.89 vs 0.86), and **D-FINE
  finds more real people on MEVA CCTV** (recall 0.63 vs 0.52). **Fine-tuning D-FINE** lifts
  SmartSpaces AP50 to **0.94** (AP50–95 0.64 → 0.76) and MEVA AP50 to 0.45 (recall flat at 0.63).
  The first-pass YOLO11 fine-tune regressed on SmartSpaces (§9). The synthetic→real gap stays large
  (AP50 drop ≈ 0.4–0.5; MEVA AP is a lower bound, so part of the gap is label incompleteness).
- ID switches vs default ByteTrack (same detections, frozen val-tuned tracker):
  COCO D-FINE **MEVA −53%** (−46% per *tracked* person-minute: track recall −7 pts), SmartSpaces −36%;
  **recommended fine-tuned D-FINE −33% / −32%** (tracker tuned on COCO-D-FINE detections, not yet
  re-tuned); YOLO11s −39% / −39%. Short-gap switches (crossings, shelf occlusions) drop 52–62%,
  except YOLO11s on MEVA (−33%) and fine-tuned D-FINE on MEVA (−29%). What's left is mostly **re-entries** (people leaving the camera's view and
  returning ≥ 3 s later), which a weight-free colour embedder can't fix.
  **The −50% criterion is not met** → decision needed (Blockers in the PR).
- D-FINE's TensorRT engine had three numerics bugs (NaN in FP16, batch size frozen in the trace,
  wrong results at batch > 1). All fixed and now guarded at export time (§5).

## 1. What was built

| Piece | File | Notes |
|---|---|---|
| Input transform + shared detector base | `perception/detect.py` | Letterbox/stretch to 640, exact inverse to **main-stream pixels** (ADR 0002), `FrameRef.transform` tag, round-trip unit tests. |
| TensorRT build/run | `perception/detect_trt.py`, `detect_export.py` | ONNX → engine, dynamic batch 1–16, recipe JSON with versions + hashes. Export refuses engines that emit NaN/inf or change results with batch size. |
| YOLO11s detector (AGPL, R&D) | `perception/detect_yolo.py` | FP16, torchvision NMS. `[license-risk]`. |
| D-FINE-S detector (Apache-2.0) | `perception/detect_dfine.py` | BF16 + FP32 decoder arithmetic (see §5), NMS-free. |
| Deadline micro-batcher (D9) | `perception/detect_batching.py` | Batch closes when full or when the oldest frame waited 20 ms; ≤ 1 frame per camera per batch; drop-oldest per camera. |
| ByteTrack (MIT port) | `perception/track.py`, `track_bytetrack.py` | Faithful to ifzhang/ByteTrack incl. its list-update order; numpy-only fallback Hungarian so CPU CI runs it. Default config = the acceptance **baseline**. |
| Appearance tracker (BoT-SORT-style) | `perception/track_botsort.py` | MIT BoT-SORT association (IoU-gated appearance), + lost-track re-ID stage for shelf occlusions, + optional long-term re-ID memory for re-entries. Weight-free colour embedder (§6). |
| Cross-camera associator | `perception/track_xcam.py` | In-memory, TTL ≤ 30 min (enforced), **off by default**. |
| Eval / tuning / fine-tune | `track_eval.py`, `track_tune.py`, `track_data.py`, `detect_finetune.py`, `detect_bench.py` | Readers mirror T09's splits; tuning on val only; bootstrap CIs over clips. |

## 2. Data and protocol

- **SmartSpaces** (synthetic Isaac Sim retail, CC-BY-4.0, fully labelled): test = `scene_073`
  (16 cameras × first 120 s), val = `scene_072`, train = 071, 074–080. Same split as T09's
  `smartspaces_track` (PR #9). All scenes share one store and character set, so appearance
  overlaps across splits; tracking numbers are optimistic for re-ID.
- **MEVA indoor** (real CCTV, CC-BY-4.0): test = site `bus` (G331, G508) + `school.G421`,
  val = `school.G423` (T09's split). Site `bus` is held out entirely (cameras at one site film the
  same actors at once); `school.G421` is an unseen camera at a training site, so its actors may
  appear in training footage from other school cameras (affects the fine-tuned rows). The 2 clips per camera with the most labelled person boxes (5 min each).
  **Only activity participants are labelled** → precision, AP and IDF1 are lower bounds;
  recall and ID switches per labelled person-minute are unaffected. Proxy, not retail.
- Detection at **10 fps** (every 3rd frame of 30 fps sources; ARCHITECTURE §5), trackers
  online/causal, all trackers scored on identical cached detections.
- ID switches: CLEAR-MOT (py-motmetrics, IoU ≥ 0.5) per GT person-minute. Split into
  **short-gap** (< 3 s since the person was last matched: crossings, shelf occlusions) and
  **re-entry** (≥ 3 s: mostly the person left the view and came back).
- 95% CIs: bootstrap over clips. SmartSpaces clips are simultaneous views of one scene, so
  those CIs are narrower than the true uncertainty.
- Demo clips (`tests/fixtures/video`): smoke level only (PoC boxes burned into pixels);
  used by the integration test, not for numbers.

## 3. Results

### 3.1 Detection (person, held-out test)

| Detector | SmartSpaces AP50 [95% CI] | AP50–95 | recall@0.3 | MEVA AP50* [95% CI] | AP50–95* | recall@0.3 | Gap AP50 (SS − MEVA) |
|---|---|---|---|---|---|---|---|
| D-FINE-S (COCO) | 0.862 [0.81, 0.90] | 0.636 | 0.828 | 0.397 [0.35, 0.45] | 0.151 | 0.633 | 0.465 |
| YOLO11s (COCO) | 0.893 [0.86, 0.93] | 0.754 | 0.854 | 0.359 [0.34, 0.40] | 0.144 | 0.518 | 0.534 |
| D-FINE-S fine-tuned | 0.942 [0.92, 0.96] | 0.755 | 0.918 | 0.445 [0.40, 0.50] | 0.172 | 0.626 | 0.497 |
| YOLO11s fine-tuned | 0.803 [0.64, 0.92] | 0.716 | 0.753 | 0.415 [0.34, 0.49] | 0.173 | 0.476 | 0.388 |

\* MEVA labels only activity participants: AP is a lower bound; recall is the reliable column.

### 3.2 Tracking — SmartSpaces `scene_073` (synthetic retail, 16 cams × 120 s)

ID switches per GT person-minute (lower is better), Δ vs **default ByteTrack on the same detections**.

| Detector | Tracker | IDSW/p-min [95% CI] | Δ vs baseline | short-gap | re-entry | IDF1† | MOTA† | track recall |
|---|---|---|---|---|---|---|---|---|
| D-FINE-S (COCO) | bytetrack_default | 5.19 [4.24, 6.16] | +0% | 2.51 | 2.68 | 0.505 | 0.732 | 0.791 |
| D-FINE-S (COCO) | bytetrack_tuned | 5.22 [4.26, 6.19] | +1% | 2.48 | 2.74 | 0.511 | 0.714 | 0.806 |
| D-FINE-S (COCO) | appearance_noreid | 4.37 [3.59, 5.14] | -16% | 1.80 | 2.57 | 0.513 | 0.747 | 0.782 |
| D-FINE-S (COCO) | appearance | 3.75 [3.16, 4.46] | -28% | 1.23 | 2.52 | 0.531 | 0.750 | 0.785 |
| D-FINE-S (COCO) | **appearance_tuned** | 3.30 [2.70, 3.87] | **-36%** | 1.19 | 2.10 | 0.572 | 0.757 | 0.786 |
| YOLO11s (COCO) | bytetrack_default | 5.05 [3.97, 6.21] | +0% | 2.39 | 2.67 | 0.530 | 0.777 | 0.813 |
| YOLO11s (COCO) | bytetrack_tuned | 4.72 [3.79, 5.70] | -7% | 2.06 | 2.66 | 0.539 | 0.786 | 0.823 |
| YOLO11s (COCO) | appearance_noreid | 4.16 [3.36, 5.20] | -18% | 1.66 | 2.51 | 0.545 | 0.779 | 0.800 |
| YOLO11s (COCO) | appearance | 3.52 [2.78, 4.48] | -30% | 1.03 | 2.50 | 0.560 | 0.782 | 0.802 |
| YOLO11s (COCO) | **appearance_tuned** | 3.08 [2.52, 3.81] | **-39%** | 0.91 | 2.17 | 0.594 | 0.784 | 0.804 |
| D-FINE-S fine-tuned | bytetrack_default | 5.06 [3.60, 6.57] | +0% | 2.25 | 2.81 | 0.570 | 0.843 | 0.891 |
| D-FINE-S fine-tuned | bytetrack_tuned | 4.63 [3.22, 6.07] | -8% | 1.84 | 2.79 | 0.576 | 0.836 | 0.896 |
| D-FINE-S fine-tuned | appearance_noreid | 4.63 [3.28, 5.97] | -9% | 1.83 | 2.79 | 0.582 | 0.863 | 0.895 |
| D-FINE-S fine-tuned | appearance | 3.76 [2.74, 4.90] | -26% | 0.98 | 2.78 | 0.597 | 0.866 | 0.897 |
| D-FINE-S fine-tuned | **appearance_tuned** | 3.46 [2.51, 4.45] | **-32%** | 1.04 | 2.41 | 0.635 | 0.866 | 0.898 |
| YOLO11s fine-tuned | bytetrack_default | 4.79 [3.31, 6.54] | +0% | 2.15 | 2.64 | 0.464 | 0.663 | 0.706 |
| YOLO11s fine-tuned | bytetrack_tuned | 4.33 [2.89, 6.02] | -10% | 1.73 | 2.60 | 0.480 | 0.667 | 0.715 |
| YOLO11s fine-tuned | appearance_noreid | 4.12 [2.84, 5.58] | -14% | 1.49 | 2.63 | 0.473 | 0.674 | 0.700 |
| YOLO11s fine-tuned | appearance | 3.57 [2.53, 4.79] | -25% | 0.97 | 2.61 | 0.485 | 0.676 | 0.701 |
| YOLO11s fine-tuned | **appearance_tuned** | 3.03 [2.10, 4.07] | **-37%** | 0.82 | 2.21 | 0.531 | 0.679 | 0.703 |

### 3.3 Tracking — MEVA test (real CCTV, 6 clips × 5 min, partial labels)

ID switches per GT person-minute (lower is better), Δ vs **default ByteTrack on the same detections**.

| Detector | Tracker | IDSW/p-min [95% CI] | Δ vs baseline | short-gap | re-entry | IDF1† | MOTA† | track recall |
|---|---|---|---|---|---|---|---|---|
| D-FINE-S (COCO) | bytetrack_default | 2.08 [1.54, 2.46] | +0% | 1.56 | 0.53 | 0.359 | -0.011 | 0.560 |
| D-FINE-S (COCO) | bytetrack_tuned | 3.38 [2.35, 4.27] | +62% | 2.85 | 0.53 | 0.330 | -0.099 | 0.599 |
| D-FINE-S (COCO) | appearance_noreid | 1.08 [0.69, 1.46] | -48% | 0.70 | 0.38 | 0.374 | 0.080 | 0.506 |
| D-FINE-S (COCO) | appearance | 1.02 [0.63, 1.37] | -51% | 0.65 | 0.37 | 0.370 | 0.080 | 0.506 |
| D-FINE-S (COCO) | **appearance_tuned** | 0.97 [0.63, 1.33] | **-53%** | 0.62 | 0.35 | 0.361 | 0.074 | 0.488 |
| YOLO11s (COCO) | bytetrack_default | 0.94 [0.63, 1.25] | +0% | 0.55 | 0.39 | 0.380 | 0.125 | 0.477 |
| YOLO11s (COCO) | bytetrack_tuned | 0.96 [0.66, 1.28] | +3% | 0.58 | 0.38 | 0.366 | 0.099 | 0.498 |
| YOLO11s (COCO) | appearance_noreid | 0.69 [0.46, 1.08] | -26% | 0.39 | 0.29 | 0.383 | 0.142 | 0.444 |
| YOLO11s (COCO) | appearance | 0.65 [0.44, 0.99] | -31% | 0.37 | 0.28 | 0.363 | 0.142 | 0.444 |
| YOLO11s (COCO) | **appearance_tuned** | 0.57 [0.35, 0.94] | **-39%** | 0.37 | 0.20 | 0.364 | 0.143 | 0.446 |
| D-FINE-S fine-tuned | bytetrack_default | 2.27 [1.90, 2.54] | +0% | 1.71 | 0.56 | 0.357 | 0.090 | 0.609 |
| D-FINE-S fine-tuned | bytetrack_tuned | 3.26 [2.56, 3.84] | +44% | 2.69 | 0.57 | 0.343 | 0.064 | 0.624 |
| D-FINE-S fine-tuned | appearance_noreid | 1.78 [1.37, 2.15] | -21% | 1.33 | 0.45 | 0.373 | 0.125 | 0.595 |
| D-FINE-S fine-tuned | appearance | 1.84 [1.40, 2.23] | -19% | 1.39 | 0.45 | 0.371 | 0.123 | 0.595 |
| D-FINE-S fine-tuned | **appearance_tuned** | 1.51 [1.03, 1.93] | **-33%** | 1.21 | 0.30 | 0.372 | 0.134 | 0.578 |
| YOLO11s fine-tuned | bytetrack_default | 0.88 [0.74, 1.11] | +0% | 0.52 | 0.35 | 0.347 | 0.151 | 0.391 |
| YOLO11s fine-tuned | bytetrack_tuned | 0.98 [0.71, 1.36] | +12% | 0.63 | 0.36 | 0.351 | 0.165 | 0.430 |
| YOLO11s fine-tuned | appearance_noreid | 0.56 [0.44, 0.71] | -37% | 0.33 | 0.23 | 0.294 | 0.109 | 0.307 |
| YOLO11s fine-tuned | appearance | 0.55 [0.43, 0.72] | -37% | 0.33 | 0.23 | 0.289 | 0.111 | 0.309 |
| YOLO11s fine-tuned | **appearance_tuned** | 0.51 [0.38, 0.70] | **-42%** | 0.32 | 0.19 | 0.289 | 0.111 | 0.311 |

† On MEVA, IDF1/MOTA are lower bounds (unlabelled bystanders count as false positives).


### 3.4 Latency (TensorRT, RTX 5090, `scripts/gpu exclusive`)

`engine` = TRT execute on a ready (B,3,640,640) tensor; `e2e` = `Detector.detect()` from 2560×1440 uint8 GPU frames (resize + pad + engine + decode/NMS + mapping to main-stream pixels). 300 iterations each.

| Detector | Precision | Batch | engine p50 / p95 ms | e2e p50 / p95 ms | e2e img/s |
|---|---|---|---|---|---|
| dfine:dfine-s/dfine_bf16_b16 | bf16 | 1 | 1.55 / 1.55 | 1.73 / 1.76 | 575 |
| dfine:dfine-s/dfine_bf16_b16 | bf16 | 4 | 2.30 / 2.31 | 2.88 / 2.90 | 1392 |
| dfine:dfine-s/dfine_bf16_b16 | bf16 | 8 | 3.36 / 3.37 | 4.43 / 4.46 | 1806 |
| dfine:dfine-s/dfine_bf16_b16 | bf16 | **10** | 4.11 / 4.13 | 5.45 / 5.49 | 1834 |
| dfine:dfine-s/dfine_bf16_b16 | bf16 | 16 | 6.32 / 6.34 | 8.50 / 8.55 | 1882 |
| yolo11:yolo11s_fp16_b16 | fp16 | 1 | 0.96 / 0.97 | 1.14 / 1.17 | 870 |
| yolo11:yolo11s_fp16_b16 | fp16 | 4 | 1.29 / 1.30 | 1.78 / 1.81 | 2242 |
| yolo11:yolo11s_fp16_b16 | fp16 | 8 | 1.67 / 1.69 | 2.64 / 2.66 | 3027 |
| yolo11:yolo11s_fp16_b16 | fp16 | **10** | 1.95 / 1.96 | 3.16 / 3.18 | 3166 |
| yolo11:yolo11s_fp16_b16 | fp16 | 16 | 3.41 / 3.42 | 5.37 / 5.40 | 2978 |

**Dead/slow camera isolation** (10 cameras × 10 fps of 2560×1440 frames, deadline 20 ms, max batch 16, 20 s per scenario; latency of the *healthy* cameras, submit → detections):

| Detector | Scenario | p50 ms | p95 ms | p99 ms | mean batch |
|---|---|---|---|---|---|
| dfine:dfine-s/dfine_bf16_b16 | all_alive | 14.0 | 27.4 | 28.5 | 2.4 |
| dfine:dfine-s/dfine_bf16_b16 | cam3_dead | 12.8 | 22.7 | 23.5 | 2.2 |
| dfine:dfine-s/dfine_bf16_b16 | cam5_slow_1fps | 13.2 | 23.0 | 23.6 | 2.4 |
| yolo11:yolo11s_fp16_b16 | all_alive | 12.0 | 22.0 | 22.1 | 2.4 |
| yolo11:yolo11s_fp16_b16 | cam3_dead | 12.0 | 22.0 | 22.2 | 2.3 |
| yolo11:yolo11s_fp16_b16 | cam5_slow_1fps | 13.2 | 24.4 | 24.7 | 2.4 |

Notes on the numbers:
- Tracker rows share one detection run per detector, so tracker differences are pure tracker
  effects. `appearance_tuned` was chosen on **val only** (84 + 36 + 8 configs; `track_tune.py`,
  every row in `T03-tune-grid.jsonl`) and evaluated once on test.
- Selection guard: switch rate alone is gameable (`track_thresh` 0.85 gave ~0 switches at IDF1 0.02).
  Chosen configs must keep IDF1 ≥ baseline and MOTA within 2 pts on both val sets.
- The tuned config trades some coverage on real CCTV: D-FINE on MEVA test loses 7.2 pts of
  track recall on labelled people (MOTA +8.4, IDF1 +0.2). Normalized per *tracked* person-minute
  the MEVA reduction is −46% (SmartSpaces −36%, unchanged).
- `bytetrack_tuned` (longer buffer, lower thresholds) is **worse** than the defaults with D-FINE
  on MEVA (+62%): D-FINE's NMS-free low-score duplicates feed ByteTrack's second association.
  The tuned appearance tracker suppresses duplicates (`dedup_iou` 0.6) before association.
- Isolation: mean batch ≈ 2.4 because the synthetic cameras are phase-staggered; batch size
  follows arrival alignment, not camera count. That's the intended D9 behaviour.

## 4. Acceptance

- [x] **Batch of 10 × 640 px < 10 ms** on the 5090 under `scripts/gpu exclusive`: D-FINE 4.11 ms
  (engine) / 5.45 ms (e2e), YOLO11s 1.95 / 3.16 ms (§3.4).
- [x] **One camera stops → others' latency unchanged**: `test_dead_camera_does_not_change_other_latency`
  and `test_slow_camera_…` (CPU, fake detector) + `test_dead_camera_isolation_real_engine` (GPU) +
  the bench table in §3.4.
- [x] **One recommended production candidate** (§7).
- [ ] **ID switches per person-minute ≥ 50% lower than default ByteTrack**: COCO D-FINE MEVA −53%
  (−46% per tracked person-minute), SmartSpaces −36%; recommended fine-tuned D-FINE −33% / −32%.
  Short-gap switches (COCO D-FINE) are −52% (SS) and −60% (MEVA). The demo
  clips have no ground truth (PoC boxes burned in), so they're a smoke test only
  (`test_integration_detect_track_fixture`). `quick_capture` doesn't exist yet. **Not met.**
- [x] **Both detectors pass the same interface suite**: `tests/perception/test_detect_gpu.py`,
  8 tests × 2 detectors, 16/16 passed on the 5090 (incl. the real-engine isolation test).
- [x] **Report committed**: this file, `T03-eval-results.json` (all test/bench numbers, generated),
  `T03-tune-grid.jsonl` (val tuning rows).


## 5. D-FINE engine numerics (a finding worth knowing before T12)

Three separate defects appeared between PyTorch and the TensorRT engine; each is fixed and
guarded so it cannot silently come back:

1. **NaN in FP16.** HF D-FINE masks invalid anchors with `float32.max`; FP16 clips it to
   65504 and the decoder overflows. Fix: replace the sentinel with 100 in the ONNX (identical
   after the sigmoid) and pin the decoder/head elementwise layers (1318 of 6054, found by
   bisection) to FP32. Backbone and encoder stay 16-bit.
2. **Batch size baked into the graph.** `batch_size = len(source_flatten)` in HF's code is
   frozen by the TorchScript tracer to the dummy batch (2). Fix: a scoped shim during export
   so `len(tensor)` traces as `tensor.shape[0]`.
3. **Wrong results at batch > 1 in dynamic-shape FP16** (and FP32/TF32 at batch 16) on
   TRT 10.16 / sm_120: identical images in one batch got different detections, and some
   frames lost a 0.92-confidence person. Static-shape FP16 engines and **BF16** dynamic
   engines were correct. Fix: D-FINE engines are **BF16** (same tensor-core speed as FP16 on
   Blackwell). This deviates from "FP16" in the brief on purpose; YOLO11 is FP16.
   Worth re-testing on the next TensorRT release and on the T12 target GPU.

Guard: `detect_export` runs the fresh engine on the fixture clip and fails if outputs are
non-finite, if batch-N and batch-1 results disagree, or if two copies of the same image in
one batch disagree. The interface test `test_batch_equals_single` checks the same thing.
Before these fixes the full SmartSpaces eval reported AP50 0.58 for D-FINE; after, 0.86.

## 6. Licenses

| Component | License | Use | Notes |
|---|---|---|---|
| D-FINE code + `ustc-community/dfine-small-coco` weights | Apache-2.0 | **prod candidate** | COCO-only weights, revision pinned in `models/MANIFEST.yaml`. |
| HF transformers (export only) | Apache-2.0 | prod | Not needed at runtime (engine only). |
| Ultralytics YOLO11 + weights | **AGPL-3.0** | R&D only `[license-risk]` | Needs an enterprise license before any customer build. |
| ByteTrack reference (ported) | MIT | prod | Attribution in `track.py`. |
| BoT-SORT reference (ported) | MIT | prod | Attribution in `track.py`. |
| boxmot | **AGPL-3.0** (verified via GitHub API) | not used | Recorded in docs/DATA.md. |
| Learned ReID weights (OSNet etc.) | weights trained on Market-1501/MSMT17/DukeMTMC | **blocked** | Dataset terms not cleared / withdrawn → colour-histogram embedder instead (no weights, no lineage). |
| TensorRT, py-motmetrics, scipy, OpenCV | Apache-2.0 / MIT / BSD / Apache-2.0 | prod | |
| SmartSpaces, MEVA | CC-BY-4.0 | prod | Attribution required in any published material. |

## 7. Recommendation

**Production candidate: `dfine-s-retail` (D-FINE-S fine-tuned on SmartSpaces + MEVA; Apache-2.0
base, CC-BY-4.0 data, pseudo-labels from the Apache COCO teacher, so prod lineage) →
`AppearanceTracker` (frozen config in `T03-eval-results.json` → `tune.chosen`), cross-camera
association off.** Build recipe and hashes: `models/MANIFEST.yaml`.

Why D-FINE over YOLO11s:
1. **License.** YOLO11 is AGPL-3.0; shipping it needs an enterprise license. D-FINE code and
   weights are Apache-2.0 with COCO-only lineage.
2. **Accuracy.** On MEVA (real CCTV, the closer proxy for a store camera) D-FINE finds
   more people: recall@0.3 0.63 vs 0.52 (COCO weights), 0.63 vs 0.48 (fine-tuned). Fine-tuned,
   it is also best on SmartSpaces (AP50 0.94 vs 0.80 for the YOLO fine-tune, 0.89 for COCO YOLO).
3. **Latency fits.** 5.45 ms end-to-end for 10 cameras, well within the 100 ms/10 fps budget.
   YOLO's 2× speed advantage doesn't buy anything at this camera count.
4. **NMS-free** outputs mean latency doesn't grow with crowd size.

Costs: D-FINE needed three engine fixes (§5), emits low-score near-duplicates (handled by the
tracker's `dedup_iou`), and its boxes jitter by a few pixels between batch sizes. On MEVA its
absolute switch rate is higher than YOLO's (1.51 vs 0.51 per person-minute, fine-tuned, tuned
tracker) because it tracks more of the hard, partly occluded people YOLO misses (track recall
0.58 vs 0.31); IDF1 is on par (0.37 vs 0.29) and MOTA higher (0.13 vs 0.11). Re-tuning the tracker
on val with the fine-tuned detector is the first follow-up. YOLO11s stays in the repo as the
comparison detector only.

## 8. Cross-camera associator (off by default)

`CrossCameraAssociator(enabled=False)` passes tracks through untouched. When enabled it keeps one
appearance vector per global identity in process memory, never persists or publishes it, and
expires identities ≤ 30 min after last sighting (constructor rejects longer TTLs). It
**helps** when cameras hand people off through a door or aisle with little overlap and T05
uses `global_id` (e.g. "picked up at the shelf cam, exited through the door cam"). It **hurts**
with look-alike clothing (uniformed staff, dark hoodies), where a wrong merge is worse for T05
than no merge. With the colour embedder, turn it on only after measuring the merge rate on that
site. SmartSpaces has global IDs, so a proper MTMC evaluation is a cheap follow-up.

## 9. Fine-tuning (first pass)

`detect_finetune.py` builds a YOLO-format set from SmartSpaces train scenes (9,525 frames, 44k
boxes) and MEVA train cameras (1,409 frames; 37k GT boxes + 5.5k **pseudo-labels** from the
COCO D-FINE teacher for unlabelled bystanders, used for training only). Frames are stored at 960 px.

| Model | Recipe | SmartSpaces AP50 / AP50–95 | MEVA AP50* / recall@0.3 |
|---|---|---|---|
| D-FINE-S COCO → fine-tuned | 10 epochs, AdamW 1e-4 (backbone 1e-5), one-cycle, BF16 autocast, h-flip; ~15 min | 0.862 / 0.636 → **0.942 / 0.755** | 0.397 / 0.633 → **0.445 / 0.626** |
| YOLO11s COCO → fine-tuned | ultralytics, 20 epochs, batch 32, `single_cls`, best.pt by val fitness | 0.893 / 0.754 → 0.803 / 0.716 | 0.359 / 0.518 → 0.415 / 0.476 |

- D-FINE fine-tuning closes most of the synthetic gap and helps on real CCTV AP (lower bound)
  without losing recall. The first-pass YOLO fine-tune got *worse* on SmartSpaces: it selects
  `best.pt` on a val set that is the storage room plus one MEVA camera, not the retail floor,
  and its scores shift down under the 1-class head (recall@0.3 drops). Not tuned further: it's
  the AGPL comparison model.
- The tracker config was tuned on the COCO D-FINE's val detections. With the fine-tuned detector
  the same frozen config gives −32% (SmartSpaces) / −33% (MEVA) vs default ByteTrack on its own
  detections (rows in §3). Re-tuning on val with the fine-tuned detector is the next step.
- \* MEVA AP is a lower bound (partial labels). Pseudo-labels are training-only; test truth is
  never model-labelled.

## 10. Known gaps and follow-ups

- **Re-entry ID switches** dominate what's left on SmartSpaces. Options, in order of expected value:
  (a) train a small ReID model on SmartSpaces global IDs (CC-BY-4.0, so licence-clean), in memory only;
  (b) evaluate `CrossCameraAssociator` as MTMC on SmartSpaces;
  (c) score re-entries separately and give them to T05's journey logic rather than the per-camera tracker.
- **MEVA partial labels** make precision/AP/IDF1 lower bounds. A small, fully-labelled MEVA
  subset (a few minutes per camera) would make MEVA AP meaningful.
- **Fine-tuning**: see §9. Checkpoint selection should use a retail-floor val scene, and the
  score calibration shift needs a closer look before more epochs.
- **TensorRT**: re-test FP16 dynamic D-FINE on the next TRT release and on the T12 target GPU.
  The export guard will catch a regression.
- **Micro-batch tuning**: deadline 20 ms and max batch 16 are defaults. With 10 aligned cameras
  the batch fills before the deadline; T12 should tune both on the real ingest timing.
- Hand/item detection (out of scope): SmartSpaces has no hand labels. MERL Shopping
  (licence to check) is the candidate for a hand-in-shelf detector.

## 11. Reproduce

```bash
uv pip install -e ".[dev,perception,torch]" -c docs/reports/T03-env.lock.txt   # + deps in HANDOFF.md
scripts/gpu exclusive -- python -m scs.perception.detect_export dfine            # and: yolo11
scripts/gpu exclusive -- python -m scs.perception.detect_bench --detector dfine --engine models/dfine-s/dfine_bf16_b16.engine
for s in test val; do for ds in smartspaces meva; do
  scripts/gpu exclusive -- python -m scs.perception.track_eval --dataset $ds --split $s --detector dfine \
      --engine models/dfine-s/dfine_bf16_b16.engine --out runs/t03/eval_$s; done; done
python -m scs.perception.track_tune --round 1|2|3 --dets-smartspaces runs/t03/dets/smartspaces_val_… --dets-meva …
python -m scs.perception.track_eval … --tracker-config runs/t03/tune2/appearance_tuned.json   # CPU, cached dets
```

