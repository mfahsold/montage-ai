# Registry troubleshooting

This document contains quick diagnostics and checks for the internal Docker registry used by the cluster.

Quick checks

- Ping the host: `ping -c 2 <registry-host>`

- Preferred: Use the `REGISTRY_URL` value from `deploy/config.env` and check it directly: `curl -v ${REGISTRY_URL}/v2/` or `nc -zv <registry-host> <registry-port>` (canonical port: `30500`, legacy: `5000`).

- Example (override): `nc -zv ${REGISTRY_HOST:-192.168.1.12} ${REGISTRY_PORT:-30500}` or `curl -v http://${REGISTRY_HOST:-192.168.1.12}:${REGISTRY_PORT:-30500}/v2/`

Useful tools

- `./scripts/registry_check.sh` — quick shell based test (TCP + HTTP/HTTPS)
- `python3 scripts/check-registry.py <host>` — structured output (TCP/HTTP/HTTPS) using `requests` and `socket` (preferred)
- `./scripts/check-hardcoded-registries.sh` — scan the repo for hardcoded registry strings

Common fixes

- Ensure the registry process (docker container or systemd service) is running and listening on the expected port.

- Confirm whether the registry should be served via HTTP or HTTPS; if HTTPS, make sure TLS certs are installed and valid.

- If using HTTPS with private CA, ensure the CA certs are installed on builder and cluster nodes.

- Check firewall rules and host-based ACLs allowing the CI/build hosts to reach the registry port.

Commands to run on the registry host

- `sudo ss -ltnp | grep 30500`

- `docker ps | grep registry`

- `curl -v http://127.0.0.1:5000/v2/` (when run on the host)

Notes for deployment

- The canonical registry host is documented in `deploy/config-global.yaml` and in `fluxibri_core` docs. If a different host/port is required, update `CLUSTER_REGISTRY` in the Makefile or the cluster overlay configuration.

- Consider adding a liveness/readiness Probe to the registry deployment for quicker CX feedback.

Automated fallback behavior

- The `scripts/build-and-deploy.sh` script attempts to push to the configured registry; if the registry is unreachable or the push fails it will attempt fallbacks in order:
  1) GHCR (if `GHCR_PAT` is set in the environment or CI secrets)
  2) Node import using `./scripts/load-image-to-cluster.sh` when `NODE_IMPORT_NODES` (comma-separated list of IPs) is provided.

- Environment variables used by the fallback logic:
  - `GHCR_PAT`: Personal access token for GHCR (required for GHCR fallback)
  - `GHCR_OWNER`: Owner/org to push GHCR images into (defaults to repo owner)
  - `GHCR_USER`: Username used for GHCR login (defaults to the action actor or `mfahsold`)
  - `NODE_IMPORT_NODES`: Comma-separated list of node IPs for node-import fallback (optional)

- If you prefer to always push to GHCR as a mirror, set `GHCR_PAT` and `GHCR_OWNER` in your CI secrets and optionally add a GHCR fallback workflow (we've added an optional, manual `workflow_dispatch` workflow in `fluxibri_core` to support this).

If you prefer, run `make registry-check` from the repo to run these checks from the current dev host.