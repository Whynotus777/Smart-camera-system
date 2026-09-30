"""`meva_interaction`: interaction-event recall on held-out MEVA cameras/sites (proxy, not retail).

What it measures (T14 update): pick-up / put-down / transfer detection by the full
streaming pipeline, NOT theft (MEVA has only 5 scripted steals; they are counted as
pickups here). Theft recall comes from `public_pose` (real, pose-only) and synthetic
suites flagged synthetic.

- Data: `meva` split `test` (site `bus` + camera `school.G421`), annotated clips only.
  Threshold fit on split `val` (camera `school.G423`) at the budget, then frozen.
- Predictions: `eval.e2e.interaction_preds` from a pipeline run on the videos.
- Positives: item_pickup / item_put_down / item_transfer, `visible: observed`,
  annotation `src_status: good`; not_good annotations are ignore regions.
- Budget: false interaction detections per camera-hour (`--opt budget=`, default 0.1 =
  the product alert budget, stricter than needed for pre-alert events; recall is also
  reported at 1 and 10 per camera-hour from the test curve, marked oracle).
- MEVA clips are continuous 5-min recordings with exhaustive labels for these types,
  so false detections per hour are defined here.
"""

from __future__ import annotations

from eval.datasets import load_dataset
from eval.e2e import interaction_preds
from eval.metrics.events import (
    curve,
    latencies,
    match,
    operating_point,
    per_subtype_recall,
    threshold_at_budget,
)
from eval.metrics.latency import latency_summary
from eval.suites._common import crash_summary, gt_for, hours, run_e2e
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite

POSITIVE = frozenset({"item_pickup", "item_put_down", "item_transfer"})


@register_suite
class MevaInteraction(Suite):
    name = "meva_interaction"
    description = (
        "Interaction-event recall at an FA budget, MEVA held-out cameras/sites, full streaming pipeline"
    )
    datasets = ("meva",)
    labels = ("proxy, not retail", "pick-up/put-down/transfer detection only; not theft recall")
    continuous = True
    end_to_end = True
    gated = ("recall_at_budget",)

    def run(self, ctx: RunContext) -> SuiteResult:
        budget = ctx.opt("budget", 0.1, float)
        tol = ctx.opt("tol", 2.0, float)
        dup = ctx.opt("dup_as_false", "1") not in ("0", "false")
        try:
            ds = load_dataset("meva", ctx.data_root, ctx.split_dir)
        except FileNotFoundError as e:
            return self.unavailable(str(e))

        def annotated(split: str) -> list[str]:
            ids = [c for c in ds.clip_ids(split) if ds.clip(c).labels.supports.events]
            return ids[: ctx.limit] if ctx.limit else ids

        val_ids, test_ids = annotated("val"), annotated(ctx.split)
        if not test_ids:
            return self.unavailable(f"no annotated converted clips in split {ctx.split!r}")
        outs = run_e2e(ctx, ds, val_ids + test_ids, max_frames=ctx.opt("max_frames", None, int))
        crashes = crash_summary(outs)

        def matched(ids: list[str]):
            ok = [c for c in ids if outs[c].error is None]
            gts = [g for c in ok for g in gt_for(ds.clip(c).labels, POSITIVE)]
            preds = [p for c in ok for p in interaction_preds(outs[c])]
            gts = [g.__class__(**{**g.__dict__, "subtype": g.type}) for g in gts]  # per-type breakdown
            return match(gts, preds, tol, {c: hours(ds.clip(c).labels) for c in ok}), ok

        val, val_ok = matched(val_ids)
        test, test_ok = matched(test_ids)
        notes = []
        if val_ok:
            thr = threshold_at_budget(val, budget, dup)
            thr_src = "val (school.G423), frozen"
        else:
            thr = threshold_at_budget(test, budget, dup)
            thr_src = "ORACLE on test (no val clips available)"
            notes.append("No val clips: threshold fit on test (oracle). Not a valid generalization number.")
        op = operating_point(test, thr, dup, ctx.bootstrap)
        val_h = sum(val.unit_hours.values()) if val.unit_hours else 0.0
        if val_ok and budget * val_h < 1:
            notes.append(f"val has {val_h:.1f} camera-hours: one false detection = {1 / val_h:.2f}/h "
                         f"> budget {budget:g}/h, so the fit tolerates zero val false positives (coarse)")
        oracle = {}
        for b in (0.1, 1.0, 10.0):
            t = threshold_at_budget(test, b, dup)
            oracle[f"recall_at_{b:g}_per_h_oracle"] = operating_point(test, t, dup, ctx.bootstrap).recall
        if crashes["crashed_clips"]:
            notes.append(f"{crashes['crashed_clips']} clip(s) crashed and are excluded from metrics")
        if ctx.limit:
            notes.append(f"--limit {ctx.limit}: smoke run, not a reportable number")
        return self.result(
            "partial" if crashes["crashed_clips"] or ctx.limit else "ok",
            {"recall_at_budget": op.recall, "fa_per_hour": op.fa_per_hour},
            metrics={
                "threshold": thr,
                "threshold_source": thr_src,
                "counts": op.counts,
                "per_type_recall": per_subtype_recall(test, thr, ctx.bootstrap),
                **oracle,
                "latency_event_end_to_emit": latency_summary(
                    latencies(test, thr), "event end -> emit, media time", ctx.bootstrap
                ),
                "val_clips": len(val_ok),
                "test_clips": len(test_ok),
                "pipeline": crashes,
            },
            curves={"test_recall_vs_fa": curve(test, dup)},
            datasets=[ds.version()],
            notes=notes,
            params={
                "budget_per_camera_hour": budget,
                "tol_s": tol,
                "duplicates_count_as_false": dup,
                "split": ctx.split,
            },
        )
