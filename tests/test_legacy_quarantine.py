"""Legacy PoC code stays quarantined in legacy/ and nothing else depends on it."""

import ast
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
# Top-level names importable from the legacy tree (modules and packages), plus the dir itself.
LEGACY_NAMES = {p.stem for p in LEGACY.glob("*.py")} | {p.name for p in LEGACY.iterdir() if p.is_dir()} | {
    "legacy"
}
LEGACY_NAMES.discard("__pycache__")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_legacy_files_at_root():
    assert not [p.name for p in ROOT.glob("*.py")]


def test_nothing_outside_legacy_imports_legacy():
    offenders = {}
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), *ROOT.glob("eval/**/*.py"),
                 *ROOT.glob("sim/**/*.py"), *ROOT.glob("scripts/**/*.py")]:
        hits = _imported_roots(path) & LEGACY_NAMES
        if hits:
            offenders[str(path.relative_to(ROOT))] = sorted(hits)
    assert not offenders


def test_legacy_compiles(tmp_path):
    for i, path in enumerate(sorted(LEGACY.rglob("*.py"))):
        py_compile.compile(str(path), cfile=str(tmp_path / f"{i}.pyc"), doraise=True)
