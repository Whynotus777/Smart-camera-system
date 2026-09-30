"""Metrics for docs/EVAL.md. Pure numpy (scipy optional) so they run in CPU-only CI.

Every function returns `MetricValue`s: value + bootstrap 95% CI + sample counts, or
an explicit "unavailable" with a reason. Hand-computed toy cases for each metric
live in tests/eval/.
"""

from eval.metrics.events import (
    GTEvent,
    MatchResult,
    OperatingPoint,
    Pred,
    curve,
    latencies,
    match,
    operating_point,
    per_subtype_recall,
    threshold_at_budget,
)
from eval.metrics.frame import auc_roc, average_precision, frame_auc, frame_scores_from_tracks
from eval.metrics.journey import JourneyEvent, journey_f1
from eval.metrics.latency import latency_summary
from eval.metrics.stats import MetricValue, bootstrap_ratio, bootstrap_stat
from eval.metrics.tracking import SeqStats, TrackFrames, detection_ap, sequence_stats, tracking_metrics

__all__ = [
    "GTEvent",
    "JourneyEvent",
    "MatchResult",
    "MetricValue",
    "OperatingPoint",
    "Pred",
    "SeqStats",
    "TrackFrames",
    "auc_roc",
    "average_precision",
    "bootstrap_ratio",
    "bootstrap_stat",
    "curve",
    "detection_ap",
    "frame_auc",
    "frame_scores_from_tracks",
    "journey_f1",
    "latencies",
    "latency_summary",
    "match",
    "operating_point",
    "per_subtype_recall",
    "sequence_stats",
    "threshold_at_budget",
    "tracking_metrics",
]
