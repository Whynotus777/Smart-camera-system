"""Converter registry and the license gate every converter goes through.

A converter reads `<data_root>/<id>/raw/` (downloaded by T14, described by
`<data_root>/<id>/MANIFEST.json`) and writes `<data_root>/<id>/converted/`
(eval.canonical). It never downloads anything. Before touching raw data it checks
the dataset's approval: `pending`/`blocked` refuse to run (AGENTS.md rule 5).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.canonical import data_root

# Mirrors the `Use` column of docs/DATA.md. A converter may only run on approved data.
APPROVAL: dict[str, str] = {
    "poselift": "R&D",
    "retails": "pending",
    "meva": "prod",
    "smartspaces": "prod",
    "quick_capture": "own",
    "lab_mock_aisle": "own",
    "sim_store": "own",
}
LICENSES: dict[str, str] = {
    "poselift": "Apache-2.0 (GitHub repo); coverage of the Drive-hosted data unconfirmed (docs/DATA.md)",
    "retails": "none stated = not licensed; permission requested from the authors",
    "meva": "CC-BY-4.0 (Kitware/IARPA), attribution required",
    "smartspaces": "CC-BY-4.0 (NVIDIA PhysicalAI-SmartSpaces)",
    "quick_capture": "own footage, actor consent forms (docs/DATA.md)",
    "lab_mock_aisle": "own footage, actor consent forms (docs/DATA.md)",
    "sim_store": "own synthetic (T08); asset licenses per sim/ lineage",
}


class LicenseError(RuntimeError):
    pass


def check_license(dataset_id: str, root: Path | None = None) -> str:
    status = APPROVAL.get(dataset_id)
    if status is None:
        raise LicenseError(f"{dataset_id}: not registered in docs/DATA.md / eval.converters.base.APPROVAL")
    if status in ("pending", "blocked"):
        raise LicenseError(
            f"{dataset_id}: status {status!r} in docs/DATA.md; conversion refused until approved"
        )
    man = (root or data_root()) / dataset_id / "MANIFEST.json"
    if man.exists():
        use = json.loads(man.read_text()).get("use")
        if use in ("pending", "blocked"):
            raise LicenseError(f"{dataset_id}: MANIFEST.json says use={use!r}")
    return status


@dataclass(frozen=True)
class ConverterInfo:
    dataset_id: str
    version: str
    fn: Callable[..., dict[str, Any]]
    stub: bool = False


CONVERTERS: dict[str, ConverterInfo] = {}


def register(dataset_id: str, version: str, stub: bool = False):
    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        CONVERTERS[dataset_id] = ConverterInfo(dataset_id, version, fn, stub)
        return fn

    return deco
