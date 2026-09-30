"""Eval harness (T09): the one place where every model and pipeline change gets its numbers.

Every other workstream is judged by what this package reports, so it favors
explicit, hand-checkable definitions over convenience:

- `eval.metrics`: pure-numpy metrics with bootstrap CIs and sample counts.
- `eval.canonical` / `eval.datasets`: the canonical on-disk format and its loader.
- `eval.converters`: public/own datasets -> canonical format.
- `eval.splits`: group-aware split generator; frozen split IDs live in `eval/splits/`.
- `eval.suites`: suite base class + registry; `python -m eval.run --suite <name>`.

See `docs/EVAL.md` for the spec and `eval/README.md` for how to add a suite.
"""
