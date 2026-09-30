# T11 — VLM verifier (second opinion)

**Wave 2 · owned:** `src/scs/verify/`

## Goal
Cut false alerts before human review by asking a video-capable VLM, running locally on the 5090, a structured question about each alert clip. Also use it offline for pre-labeling.

## Deliverables
1. `Verifier` implementations for at least two open-weight VLMs (e.g. a Qwen-VL-family model and NVIDIA Cosmos Reason), served via vLLM or the model's reference runtime. Record licenses.
2. Structured prompt → JSON: `{concealment_observed: bool, item_visible: bool, where: enum, confidence: 0-1, rationale: str}`, with the subject's box drawn on frames. Validate with pydantic; retry once on malformed output.
3. Offline modes: (a) score all T05 candidate alerts on `lab_e2e`/sim/UCF-Crime to measure FA reduction vs recall loss; (b) pre-label unlabeled footage for human correction.
4. Budget: runs async on alerts only; p95 < 20 s per alert on the 5090 while the pipeline runs.

## Acceptance
- [ ] Report: precision/recall of alerts with and without the verifier at 2–3 thresholds on `lab_e2e` (once available) and sim.
- [ ] VRAM measured alongside the full pipeline; document whether production needs a second GPU or a cloud endpoint.
- [ ] Verifier failure never blocks an alert; it degrades to "unverified".

## Caution
VLM rationales are not evidence. Reviewers see them as hints only; the UI must say so.
