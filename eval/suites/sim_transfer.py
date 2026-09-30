"""`sim_transfer`: does sim data earn more investment? Same untouched real test set for all.

Variants are the `--models` roles (e.g. `real=`, `sim=`, `real_sim=`), each a model
trained on a different train set and declaring `lineage = {"train": [{"dataset", "split"}]}`.
All variants are scored on the SAME real test clips (the report records the clip-list
digest); deltas vs the `real` variant use a *paired* bootstrap (same clip resample for
both variants), which is what can show a real improvement with few clips.

Guards: a variant whose lineage lists the test dataset's `test` split fails the suite
(leak); a variant with no lineage is reported as "lineage unknown". Base suite today:
frame AUC on `public_pose` data (`--opt dataset=poselift`); lab_e2e recall once
`lab_mock_aisle` exists. Decision rule (docs/EVAL.md): sim gets more investment only if
`real_sim - real` has a CI above 0.
"""

from __future__ import annotations

import hashlib

import numpy as np

from eval.metrics.frame import _WeightedBlocks, auc_roc, bootstrap_counts, frame_auc
from eval.metrics.stats import MetricValue, percentile_ci
from eval.pose_bench import pose_clip_outputs
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite


def paired_delta(
    a: list[tuple[np.ndarray, np.ndarray]],
    b: list[tuple[np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int = 20260929,
) -> MetricValue:
    """AUC(b) - AUC(a), both resampled with the SAME clip multiplicities (paired bootstrap)."""
    if len(a) != len(b) or any(len(x[0]) != len(y[0]) for x, y in zip(a, b, strict=True)):
        raise ValueError("paired bootstrap needs the same clips for both variants")
    clip_of = np.repeat(np.arange(len(a)), [len(c[0]) for c in a])

    def blocks(v: list[tuple[np.ndarray, np.ndarray]]) -> _WeightedBlocks:
        return _WeightedBlocks(
            np.concatenate([c[0] for c in v]).astype(bool), np.concatenate([c[1] for c in v]), clip_of
        )

    wa, wb = blocks(a), blocks(b)
    d = auc_roc(np.concatenate([c[0] for c in b]), np.concatenate([c[1] for c in b])) - auc_roc(
        np.concatenate([c[0] for c in a]), np.concatenate([c[1] for c in a])
    )
    reps = np.array([wb.auc(c) - wa.auc(c) for c in bootstrap_counts(len(a), n_boot, seed)])
    return MetricValue(d, percentile_ci(reps), {"clips": len(a)})


@register_suite
class SimTransfer(Suite):
    name = "sim_transfer"
    description = "Train-set variants {real, sim, real+sim} on one fixed real test set, paired deltas"
    datasets = ("poselift",)
    labels = ("sim/emulator results never count as production validation on their own",)
    gated = ()

    def run(self, ctx: RunContext) -> SuiteResult:
        dataset = ctx.opt("dataset", "poselift")
        variants = {r: m for r, m in ctx.models.items() if r != "pipeline"}
        if len(variants) < 2:
            return self.unavailable("needs >= 2 variants: --models real=... sim=... real_sim=...")
        for r, m in variants.items():
            for t in (m.lineage or {}).get("train", []):
                if t.get("dataset") == dataset and t.get("split") == ctx.split:
                    raise ValueError(f"variant {r!r} trained on {dataset}/{ctx.split}: test-set leak")
        per: dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]] = {}
        info: dict = {}
        for r, m in variants.items():
            try:
                per[r], info = pose_clip_outputs(dataset, ctx.split, m.obj, ctx, 0.0, 1)
            except (FileNotFoundError, KeyError) as e:
                return self.unavailable(f"{dataset} not available: {e}")
        ids = {r: [c for c, _, _, _ in v] for r, v in per.items()}
        first = next(iter(ids.values()))
        if any(v != first for v in ids.values()):
            raise ValueError("variants were scored on different clips")
        if not first:
            return self.unavailable(f"no {ctx.split} clips with frame labels")
        digest = hashlib.sha256("\n".join(first).encode()).hexdigest()[:16]
        data = {r: [(y, s) for _, _, y, s in v] for r, v in per.items()}
        headline = {f"auc_roc[{r}]": frame_auc(d, b=ctx.bootstrap)["auc_roc"] for r, d in data.items()}
        deltas = {}
        if "real" in data:
            for r in data:
                if r != "real":
                    deltas[f"delta_auc[{r} - real]"] = paired_delta(data["real"], data[r], ctx.bootstrap)
        lineage = {r: (m.lineage or "unknown") for r, m in variants.items()}
        return self.result(
            "ok",
            headline,
            metrics={
                "paired_deltas": deltas,
                "lineage": lineage,
                "test_clip_digest": digest,
                "test_clips": len(first),
            },
            datasets=[info["dataset"]],
            notes=["Decision rule: more sim investment only if delta_auc[real_sim - real] CI lies above 0."]
            + ([] if "real" in data else ["No 'real' variant: paired deltas unavailable"]),
            params={"dataset": dataset, "split": ctx.split, "variants": sorted(variants)},
        )
