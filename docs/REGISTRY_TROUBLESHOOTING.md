# Registry troubleshooting

This document contains quick diagnostics and checks for the internal Docker registry used by the cluster.

Quick checks
- Ping the host: `ping -c 2 192.168.1.12`
<<<<<<< HEAD
- Test TCP connectivity to the registry port(s): `nc -zv 192.168.1.12 30500` (preferred) or `nc -zv 192.168.1.12 5000` (legacy)
- Check the registry HTTP API (canonical port 30500): `curl -v http://192.168.1.12:30500/v2/` or `curl -v https://192.168.1.12:30500/v2/` (or legacy port `5000` if configured).
=======
- Test TCP connectivity to the registry port(s): `nc -zv 192.168.1.12 5000` or `nc -zv 192.168.1.12 30500`
- Check the registry HTTP API: `curl -v http://192.168.1.12:5000/v2/` or `curl -v https://192.168.1.12:5000/v2/`
>>>>>>> origin/main

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

If you prefer, run `make registry-check` from the repo to run these checks from the current dev host.