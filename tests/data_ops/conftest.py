"""`data_ops/` lives at the repo root (not in the installed `scs` package); make it importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
