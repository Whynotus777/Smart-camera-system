"""`meva_fa`: false ALERTS per camera-hour on >= 100 h of continuous MEVA indoor video.

Proxy, not retail, until store shadow mode. MEVA has no retail theft, so every Alert is
a false alert, except alerts overlapping one of the 5 scripted `person_steals_object`
events (ignore regions). Alerts on unannotated clips are all counted false (an
unlabeled scripted steal there would be miscounted; there are ~5 in all of MEVA).

Footage pool — only footage no model trained on:
- always: every converted clip in split `test` (held-out site `bus` + camera `school.G421`);
- plus train/val clips when every model's lineage says it never trained on MEVA
  (`lineage={"train": [...]}` without `meva`). Unknown lineage -> test only (conservative).
The report states the pool, its camera-hours, and whether it reaches `--opt min_hours`
(default 100). Below that the headline is still computed but status is `partial`.
"""

from __future__ import annotations

from collections import defaultdict

from eval.datasets import load_dataset
from eval.e2e import alert_preds
from eval.metrics.events import curve, match, operating_point
from eval.suites._common import crash_summary, gt_for, hours, lineage_trained_on, run_e2e
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite

STEAL = frozenset({"person_steals_object"})


@register_suite
class MevaFA(Suite):
    name = "meva_fa"
    description = "False alerts per camera-hour on continuous MEVA indoor footage (full streaming pipeline)"
    datasets = ("meva",)
    labels = ("proxy, not retail",)
    continuous = True
    end_to_end = True
    gated = ("false_alerts_per_camera_hour",)

    def run(self, ctx: RunContext) -> SuiteResult:
        min_hours = ctx.opt("min_hours", 100.0, float)
        tol = ctx.opt("tol", 2.0, float)
        try:
            ds = load_dataset("meva", ctx.data_root, ctx.split_dir)
        except FileNotFoundError as e:
            return self.unavailable(str(e))
        trained = lineage_trained_on(ctx.models, "meva")
        splits = ["test"] if trained in (True, None) else ["train", "val", "test"]
        pool = [c for s in splits for c in ds.clip_ids(s)]
        if ctx.limit:
            pool = pool[: ctx.limit]
        if not pool:
            return self.unavailable("no converted MEVA clips in the FA pool")
        outs = run_e2e(ctx, ds, pool, max_frames=ctx.opt("max_frames", None, int))
        crashes = crash_summary(outs)
        ok = [c for c in pool if outs[c].error is None]
        unit_h = {c: hours(ds.clip(c).labels) for c in ok}
        gts = [g for c in ok for g in gt_for(ds.clip(c).labels, frozenset(), ignore_types=STEAL)]
        preds = [p for c in ok for p in alert_preds(outs[c])]
        r = match(gts, preds, tol, unit_h)
        thr = ctx.opt("alert_threshold", 0.0, float)  # 0 = every alert the policy emits reaches review
        op = operating_point(r, thr, True, ctx.bootstrap)
        total_h = sum(unit_h.values())
        per_cam: dict[str, dict[str, float]] = defaultdict(lambda: {"hours": 0.0, "false_alerts": 0})
        for c in ok:
            per_cam[ds.clip(c).labels.camera_id or c]["hours"] += unit_h[c]
        k = r.kept(thr)
        for p, o in zip(r.preds[:k], r.outcome[:k], strict=True):
            if o in ("false", "duplicate"):
                per_cam[ds.clip(p.unit).labels.camera_id or p.unit]["false_alerts"] += 1
        why = {
            None: "lineage unknown -> held-out only",
            True: "lineage: trained on MEVA",
            False: "no model trained on MEVA",
        }[trained]
        met = "met" if total_h >= min_hours else "NOT MET"
        notes = [
            f"Footage pool: splits {splits} ({why})",
            f"{total_h:.1f} camera-hours; requirement >= {min_hours:g}: {met}",
        ]
        n_unann = sum(1 for c in ok if not ds.clip(c).labels.supports.events)
        if n_unann:
            notes.append(f"{n_unann} unannotated clips: every alert there counts as false (steals unlabeled)")
        status = (
            "ok" if total_h >= min_hours and not crashes["crashed_clips"] and not ctx.limit else "partial"
        )
        return self.result(
            status,
            {"false_alerts_per_camera_hour": op.fa_per_hour},
            metrics={
                "camera_hours": round(total_h, 3),
                "meets_min_hours": total_h >= min_hours,
                "alerts": op.counts,
                "per_camera": dict(per_cam),
                "pipeline": crashes,
                "alerts_overlapping_steals_ignored": sum(1 for o in r.outcome[:k] if o == "ignored"),
            },
            curves={"fa_vs_alert_threshold": curve(r)},
            datasets=[ds.version()],
            notes=notes,
            params={"alert_threshold": thr, "tol_s": tol, "min_hours": min_hours, "pool_splits": splits},
        )
