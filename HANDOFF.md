# T04 handoff: changes needed outside T04's owned paths

T04 didn't edit these files itself (AGENTS.md rule 2). Each item names its owner.

1. **`pyproject.toml` (T00/T12): add a `trt` extra.** T04 runs TensorRT engines through
   the `tensorrt-cu12` pip wheel. The pinned working set on the 5090 (sm_120) is listed
   in `docs/reports/T04-pose.md` under "Environment". Suggested:
   `trt = ["tensorrt-cu12==<pinned>"]`. License: NVIDIA TensorRT SDK license (proprietary,
   free to use and redistribute with applications; see the TensorRT SLA). Record it in the
   release notes.
2. **`docs/DATA.md` (registry owner): weights lineage for RTMPose body7.** The row
   "RTMDet / RTMPose (OpenMMLab) | Apache-2.0 | prod candidate" covers the code. The
   `*-body7` checkpoints T04 uses are trained on COCO, AI Challenger, CrowdPose, Halpe,
   MPII, PoseTrack18 and sub-JHMDB, and several of those have research-only or
   non-commercial terms. Per rule 5 (lineage), please add a row
   `rtmpose_body7_weights | R&D (lineage: body7 datasets) | pending review` and decide
   whether a COCO-only (CC-BY-4.0 annotations) RTMPose checkpoint is required for prod.
   The body7 ONNX lives at `models/rtmpose/` (gitignored).
3. **`models/MANIFEST.yaml` (T12): engine recipes.** `pose_trt.build_engine` writes
   `<engine>.recipe.json` (TRT/torch versions, GPU, sm, precision, profile, ONNX and
   engine sha256). The recipes for the engines benchmarked here are in the T04 report.
   Please copy them into the manifest when it exists (it isn't on `main` yet).
4. **`contracts.Pose` (ADR, T00): optional fields proposal, not blocking.** The live
   smoother keeps raw and smoothed poses and an `observed | imputed` mask per keypoint
   (`pose_smooth.SmoothedPose`). On the bus only the smoothed `Pose` fits, and its
   `model_id` carries the smoother tag. Imputed keypoints have confidence decayed
   linearly to 0 over 0.5 s, so consumers that only read `Pose` still see them as
   low-confidence. If T05/T06 need the mask on the bus, the proposal is the optional
   `Pose.kp_imputed: list[bool] | None = None` and `Pose.raw_keypoints: list[Keypoint] | None = None`.
5. **T05 / T06: reaching arms outside the track box.** Top-down crops use the track box
   padded by 1.25 (model native). A wrist extended more than ~0.6 box-widths past the box
   edge falls outside the crop and gets hallucinated inside it. The T00 fixture's shelf
   reach (wrist 1.4 box-widths from center) is one example. Detector boxes normally
   include the arm, but T03 should confirm its boxes cover outstretched arms, or T04
   should widen the crop for shelf-gated tracks. That trade-off is tracked in the report.
