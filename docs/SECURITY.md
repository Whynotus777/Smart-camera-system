# Security

Scope: the lab workstation, the pilot edge box, and the camera network at a store.
The CTO owns hardening before any store deployment; this file is the starting point.

## 1. Camera credentials

- **RTSP credentials live in environment variables only.** Never in code, configs,
  notebooks, logs, or commit messages.
  - Site configs (`configs/sites/*.yaml`, gitignored) name the env var
    (`rtsp_main_env: SCS_CAM1_RTSP_MAIN`). `CameraInstall` rejects anything that
    isn't an `UPPER_SNAKE` name, so a URL pasted there fails validation.
  - Locally: copy `.env.example` to `.env` (gitignored). On the edge box: a
    root-owned env file or a secrets manager, readable only by the service user.
- **Never log a full RTSP URL.** Log the camera id; if a URL must appear, redact
  the userinfo (`rtsp://***@host/...`).
- One non-admin, view-only camera account per site for the pipeline. The admin
  account isn't used by software.
- Rotate on staff turnover and after any suspected exposure.
- **Known incident:** a Reolink admin password was committed in early history of this
  repo (ROADMAP H0). It has been removed from the working tree but remains in git
  history. Treat it as compromised: rotate it on every camera where it was used.
  Rewriting history or making the repo private is an owner decision.

### Guardrails in this repo

- `pre-commit` (`.pre-commit-config.yaml`): gitleaks, a hook that blocks
  RTSP URLs with a literal `user:password@`, a 5 MB limit on added files, and ruff.
  Install with `uv tool install pre-commit && pre-commit install`.
- CI runs gitleaks on every PR.
- `.gitignore` excludes `.env`, `data/`, `models/`, `runs/`, video and weight files.

## 2. Camera network (VLAN guidance)

- Put cameras on a **dedicated VLAN** with no internet egress and no route to the
  store's POS or guest Wi-Fi networks.
- Only the edge box may reach cameras, and only on RTSP (554/tcp, plus RTP ports if
  UDP transport is used) and ONVIF if needed. Deny camera-to-camera and
  camera-to-anything-else traffic.
- Disable camera cloud/P2P features (e.g. Reolink UID/P2P), UPnP, and unused
  services (HTTP, FTP, Telnet, email push) on each camera.
- The edge box sits on two networks: the camera VLAN (inbound RTSP) and a management
  network with outbound-only access to the review service. It must not forward
  between them.
- Reserve static DHCP leases for cameras and record MAC → camera_id in the site
  config notes, not in git.

## 3. Device hardening checklist (stub, for the CTO)

Edge box:
- [ ] Full-disk encryption (evidence clips are sensitive), TPM-backed if available
- [ ] Minimal OS image, unattended security updates, pinned NVIDIA driver
- [ ] SSH keys only, no password login, no root login; access via VPN / bastion
- [ ] Host firewall default-deny; only required ports open per interface
- [ ] Service runs as an unprivileged user; containers rootless or with dropped caps
- [ ] Secrets in a root-owned env file or secrets manager, never in the image
- [ ] Remote management / OTA with signed updates (T12 memo)
- [ ] Log shipping without video or credentials; tamper-evident audit log of reviewer actions
- [ ] Evidence retention enforced (T10 policy); secure delete of expired clips
- [ ] Physical: locked enclosure, BIOS password, USB boot disabled

Cameras:
- [ ] Firmware updated; unique strong admin password per camera, stored in a vault
- [ ] Separate least-privilege viewer account for RTSP
- [ ] Cloud/P2P, UPnP and unused services disabled; NTP set to the edge box
- [ ] On the camera VLAN only (section 2)

Review service:
- [ ] TLS everywhere; auth (basic auth OK for the lab, SSO before a customer)
- [ ] Access to clips logged; per-site authorization
- [ ] Privacy: no face recognition, no identity DB (AGENTS.md rule 6); signage/consent per H4

## 4. Reporting

Suspected secret exposure: rotate first, then tell the repo owners. Don't paste the
secret into an issue, a PR, or a chat.
