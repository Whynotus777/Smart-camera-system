"""`lab_e2e`: the release-gating end-to-end metric on staged `lab_mock_aisle` footage.

Theft event recall at the false-alert budget (1 per camera per 10 open-hours = 0.1 per
camera-hour) on held-out actors, through the full streaming pipeline from video files.

- Positives: THEFT_TYPES (item_to_clothing, item_to_bag, grab_run, exit_no_checkout)
  `visible: observed` for that camera; partially/not observed -> ignore regions.
  Benign twins (item_returned, item_to_basket, own phone, ...) are negatives: an alert on
  them is a false alert.
- Threshold fit on split `val` at the budget and frozen. FA/h only from clips marked
  `continuous` (the 2 h unscripted sessions); recall from all test clips.
- Per-subtype recall, journey F1 (checkout_visit/store_exit, if labeled), latency.
Status today: `lab_mock_aisle` isn't recorded (H1), so this reports "unavailable".
"""

from __future__ import annotations

from eval.datasets import load_dataset
from eval.e2e import alert_preds
from eval.metrics.events import (
    curve,
    latencies,
    match,
    operating_point,
    per_subtype_recall,
    threshold_at_budget,
)
from eval.metrics.journey import JourneyEvent, journey_f1
from eval.metrics.latency import latency_summary
from eval.metrics.stats import MetricValue
from eval.suites._common import THEFT_TYPES, crash_summary, gt_for, hours, run_e2e
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite

BUDGET = 0.1  # false alerts per camera-hour (docs/EVAL.md primary)


@register_suite
class LabE2E(Suite):
    name = "lab_e2e"
    description = (
        "Theft recall @ 0.1 FA/camera-hour on held-out actors, full streaming pipeline (release gate)"
    )
    datasets = ("lab_mock_aisle",)
    labels = ("staged footage (lab), held-out actors",)
    continuous = True
    end_to_end = True
    gated = ("recall_at_budget",)

    def run(self, ctx: RunContext) -> SuiteResult:
        dataset = ctx.opt("dataset", "lab_mock_aisle")
        budget = ctx.opt("budget", BUDGET, float)
        tol = ctx.opt("tol", 2.0, float)
        try:
            ds = load_dataset(dataset, ctx.data_root, ctx.split_dir)
        except (FileNotFoundError, KeyError) as e:
            return self.unavailable(f"{dataset} not available (not recorded/converted yet): {e}")
        val_ids, test_ids = ds.clip_ids("val"), ds.clip_ids(ctx.split)
        if ctx.limit:
            val_ids, test_ids = val_ids[: ctx.limit], test_ids[: ctx.limit]
        outs = run_e2e(ctx, ds, val_ids + test_ids)
        crashes = crash_summary(outs)

        def matched(ids: list[str], continuous_only: bool):
            ok = [
                c
                for c in ids
                if outs[c].error is None and (ds.clip(c).labels.continuous or not continuous_only)
            ]
            gts = [g for c in ok for g in gt_for(ds.clip(c).labels, THEFT_TYPES)]
            preds = [p for c in ok for p in alert_preds(outs[c])]
            uh = {c: hours(ds.clip(c).labels) for c in ok} if continuous_only else None
            return match(gts, preds, tol, uh)

        fa_val = matched(val_ids, True)
        if not fa_val.units:
            return self.unavailable("no continuous val footage to fit the FA-budget threshold on")
        thr = threshold_at_budget(fa_val, budget)
        rec_test, fa_test = matched(test_ids, False), matched(test_ids, True)
        rec = operating_point(rec_test, thr, b=ctx.bootstrap).recall
        fa = (
            operating_point(fa_test, thr, b=ctx.bootstrap).fa_per_hour
            if fa_test.units
            else MetricValue.unavailable("no continuous test footage")
        )
        jgt = [
            JourneyEvent(c, e.type, e.t_start, e.actor_id)
            for c in test_ids
            for e in ds.clip(c).labels.events
            if e.type in ("checkout_visit", "store_exit")
        ]
        jpred = [
            JourneyEvent(c, it.payload["type"], it.t)
            for c in test_ids
            if outs[c].error is None
            for it in outs[c].items
            if it.kind == "event" and it.payload["type"] in ("checkout_visit", "store_exit")
        ]
        return self.result(
            "partial" if crashes["crashed_clips"] or ctx.limit else "ok",
            {"recall_at_budget": rec, "fa_per_hour": fa},
            metrics={
                "threshold": thr,
                "threshold_source": "val, frozen",
                "per_subtype_recall": per_subtype_recall(rec_test, thr, ctx.bootstrap),
                "journey_f1": journey_f1(jgt, jpred, b=ctx.bootstrap),
                "latency_event_end_to_alert": latency_summary(
                    latencies(rec_test, thr), "event end -> alert emit, media time", ctx.bootstrap
                ),
                "pipeline": crashes,
            },
            curves={"test_recall_vs_fa_continuous": curve(fa_test)},
            datasets=[ds.version()],
            params={"budget_per_camera_hour": budget, "tol_s": tol, "dataset": dataset},
        )
