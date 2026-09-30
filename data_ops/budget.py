"""Disk budget guard: refuse any download that would leave < 15% of the disk free.

Eight agents share one workstation disk, and several datasets here are hundreds of
GB (MEVA is 516 GB public, SmartSpaces is 6.7 TB). A full disk kills every agent's
jobs, Redis and Docker at once. So every fetcher calls `require(nbytes)` before it
starts, and again per file for long pulls.

CLI:  python -m data_ops.budget [--need-gb N] [--path P]
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from data_ops.paths import data_root

MIN_FREE_FRACTION = 0.15
GB = 1_000_000_000


class BudgetError(RuntimeError):
    """Raised when a download would push free space below the floor."""


@dataclass(frozen=True)
class DiskState:
    path: Path
    total: int
    free: int

    @property
    def floor(self) -> int:
        """Bytes that must stay free."""
        return int(self.total * MIN_FREE_FRACTION)

    @property
    def spendable(self) -> int:
        """Bytes we may still write before hitting the floor (never negative)."""
        return max(0, self.free - self.floor)

    def summary(self) -> str:
        return (f"{self.path}: total {self.total / GB:,.0f} GB, free {self.free / GB:,.0f} GB "
                f"({self.free / self.total:.1%}), floor {self.floor / GB:,.0f} GB "
                f"({MIN_FREE_FRACTION:.0%}), spendable {self.spendable / GB:,.0f} GB")


def disk_state(path: Path | None = None) -> DiskState:
    p = Path(path) if path else data_root()
    probe = p
    while not probe.exists():  # measure the filesystem the path will live on
        probe = probe.parent
    u = shutil.disk_usage(probe)
    return DiskState(path=p, total=u.total, free=u.free)


def require(nbytes: int, path: Path | None = None, what: str = "download") -> DiskState:
    """Raise BudgetError unless writing `nbytes` still leaves >= 15% free."""
    st = disk_state(path)
    if nbytes > st.spendable:
        raise BudgetError(
            f"refusing {what} of {nbytes / GB:,.1f} GB: would leave "
            f"{(st.free - nbytes) / GB:,.1f} GB free (< {MIN_FREE_FRACTION:.0%} floor). {st.summary()}"
        )
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--need-gb", type=float, default=0.0)
    ap.add_argument("--path", type=Path, default=None)
    a = ap.parse_args(argv)
    st = disk_state(a.path)
    print(st.summary())
    if a.need_gb:
        try:
            require(int(a.need_gb * GB), a.path)
            print(f"OK: {a.need_gb:,.1f} GB fits")
        except BudgetError as e:
            print(f"REFUSED: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
