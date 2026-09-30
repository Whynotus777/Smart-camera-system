"""Where shared data lives, so eight agents in eight worktrees see one copy.

`data/` is gitignored, and each git worktree has its own. If every agent downloaded
into its own worktree we'd fetch MEVA eight times. The shared root is, in order:
1. `$SCS_DATA_ROOT` if set;
2. `<main checkout>/data`, found via `git rev-parse --git-common-dir` (works from any
   worktree of the repo);
3. `./data` as a last resort (not in a git checkout).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def data_root(start: Path | None = None) -> Path:
    env = os.environ.get("SCS_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    here = here if here.is_dir() else here.parent
    git = shutil.which("git")
    if git:
        try:
            common = subprocess.run(  # noqa: S603
                [git, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=here, capture_output=True, text=True, check=True,
            ).stdout.strip()
            return Path(common).parent / "data"
        except subprocess.CalledProcessError:
            pass
    return Path.cwd() / "data"


def dataset_dir(dataset_id: str, *parts: str) -> Path:
    """`<data_root>/<dataset_id>/<parts...>`, e.g. dataset_dir("meva", "raw")."""
    return data_root().joinpath(dataset_id, *parts)
