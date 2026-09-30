# data_ops (T14)

Data factory for a zero-spend, no-camera phase: downloads with manifests, a fake-camera
farm, generated clips, pre-labels, and dedup. Status and numbers: `docs/reports/T14-data.md`.

```bash
uv pip install -e ".[dev,perception]" -r data_ops/requirements.txt
python -m data_ops.budget                              # disk report; --need-gb N to test a size
python -m data_ops.fetch meva --indoor --max-gb 150    # MEVA indoor + Kitware annotations
python -m data_ops.fetch smartspaces                   # retail scenes 071-080
python -m data_ops.fetch coco_kp                       # COCO val2017 keypoints
python -m data_ops.fetch poselift                      # PoseLift (Google Drive, via uvx gdown)
```

**Where data goes:** `data_ops.paths.data_root()` = `$SCS_DATA_ROOT`, else the *main
checkout's* `data/` (shared by every worktree). Each dataset gets
`<root>/<id>/{raw,annotations,index}/` and a `MANIFEST.json`, which lists license,
attribution and per-file bytes, sha256, source and verification. Fetchers resume, and skip
files already verified in the manifest.

**Rules:** zero spend (AGENTS.md rule 12); every source needs a docs/DATA.md row first
(rule 5); `budget.require()` before any download; nothing under `data/` is committed.
