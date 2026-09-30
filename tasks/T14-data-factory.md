# T14 — Data factory (zero-spend)

**Wave 1 · owned:** `data_ops/`, `tests/data_ops/`, `docs/reports/T14-*.md`, rows in `docs/DATA.md` for sources you add

## Goal
With no cameras and no budget, make sure every other agent has data: downloaded,
license-tracked, deduplicated, and servable as fake cameras. Plus targeted generated clips
and machine pre-labels for later human correction.

## Deliverables
1. **Disk budget first.** `data_ops/budget.py` reads free disk and refuses downloads that would leave < 15% free. Report current free space in your first PR.
2. **Downloaders with manifests** (`data_ops/fetch/<id>.py` + `data/<id>/MANIFEST.json` with URLs, checksums, license, date):
   - `meva`: indoor cameras first, from the public S3 bucket (no account). Pull the activity annotations from the MEVA data repo.
   - `smartspaces`: **retail scenes only** from the Hugging Face dataset.
   - `poselift`, `simuletic_sample`, COCO keypoints val.
3. **Fake-camera farm:** `data_ops/replay/` runs mediamtx in Docker and serves N MEVA clips as looping RTSP streams (H.264 and H.265 variants). It supports scripted faults: drop for X s, stall, bitrate spike. T02, T12, and T13 use it.
4. **Generated hard negatives and positives** (`data_ops/generate/`): Wan 2.2 TI2V-5B (Apache-2.0) image→video from overhead retail keyframes (SmartSpaces/sim frames).
   - Prompts are templated per interaction: own phone out of pocket, item to basket, item returned, item to bag.
   - Labels come from the prompt template (`label_source: script`) plus per-clip QA flags.
   - Throughput is about 9 min per 5 s clip, so run it overnight under `scripts/gpu exclusive`; target ~100 clips for the first `sim_transfer`-style check.
   - Every clip keeps its generator, prompt, seed, and source frame for lineage.
5. **Pre-labeler** (`data_ops/prelabel/`): runs detector + tracker + pose + a local VLM over unlabeled clips and writes `label_source: model` suggestions in canonical format, plus a simple review CLI/HTML so a human can accept or fix labels quickly later. Pre-labels never count as test truth.
6. **Dedup and grouping:** perceptual-hash dedup across sources, with `group_id` assignment handed to T09's split generator.

## Acceptance
- [ ] `python -m data_ops.fetch meva --indoor --max-gb 150` works from a clean machine; checksums verified; license row present.
- [ ] Replay farm serves 10 streams for 1 h; fault injection works (T02's reconnect test passes against it).
- [ ] ≥ 50 generated clips across 4 interaction templates, with a contact sheet in the report and an honest QA note on hand/object fidelity.
- [ ] Pre-labeler runs on 1 h of MEVA and outputs valid canonical label files.

## Rules
Zero spend: no paid APIs, datasets, or cloud GPUs. No scraped YouTube/TikTok footage. Downloads go to `data/` (gitignored); only manifests and code are committed.
