# T01 — Security hygiene & legacy quarantine

**Wave 0 · owned:** root legacy `*.py`, `alerts/`, `stream/`, `legacy/`, `README.md`

## Goal
Make the repo safe and unambiguous for agents.

## Deliverables
1. Move legacy PoC files into `legacy/` (keep git history with `git mv`). Update imports only enough that `python -m py_compile legacy/*.py` passes.
2. Rewrite `README.md`: what the product is now, quickstart for the new package, pointer to `AGENTS.md`, and a short "Legacy PoC" section.
3. Add a `pre-commit` config: ruff, gitleaks, a hook that blocks `rtsp://.*:.*@` and files > 5 MB.
4. `docs/SECURITY.md`: RTSP creds via env only; camera VLAN guidance; device hardening checklist stub for the CTO.

## Acceptance
- [ ] `gitleaks detect` on the working tree is clean (history is a human decision, see ROADMAP H0).
- [ ] No file outside `legacy/` imports from legacy code.
- [ ] Pre-commit blocks a test commit containing `rtsp://a:b@1.2.3.4`.

## Notes
Credentials were already replaced with env vars in the agent-kit PR, but the old password is still in git history. Flag it in the PR; don't rewrite history yourself.
