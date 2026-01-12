# Developer Build & Deploy Quickstart

This guide shows the minimal steps to build multi-arch images and deploy to the cluster.

Prerequisites
- Docker and Buildx available (`docker --version`, `docker buildx version`).
- SSH agent set up when using private git modules (see BuildKit SSH below).
- Credentials to the registry (e.g., `docker login 192.168.1.12:5000`).

Quick build example (recommended)

1) Start SSH agent and add key (if you need access to private git modules):

   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa

2) Run the distributed build with SSH forwarding (optional):

   # Use the cluster registry
   REGISTRY=192.168.1.12:5000 TAG=20260112-quick BUILDKIT_SSH=1 ./scripts/build-distributed.sh

Notes:
- `BUILDKIT_SSH=1` enables `--ssh default` so BuildKit can forward your local SSH agent into the build containers. This is useful when Docker build needs to fetch private git modules during `go get` or similar operations.
- If you use private Go modules, set `GOPRIVATE=github.com/yourorg/*` before running the build. The script will pass `GOPRIVATE` as a build-arg.

Fallback (no SSH, public-only):

   REGISTRY=ghcr.io/mfahsold TAG=latest ./scripts/build-distributed.sh

Deploy to cluster (canary)

1) Verify image exists in registry:

   docker buildx imagetools inspect ${REGISTRY}/montage-ai:${TAG}

2) Update the kustomization or set the image directly and restart deployment:

   kubectl set image deployment/montage-ai-worker montage-ai=${REGISTRY}/montage-ai:${TAG} -n montage-ai
   kubectl rollout status deployment/montage-ai-worker -n montage-ai

Troubleshooting
- Registry unreachable: `curl -v http://192.168.1.12:5000/v2/` or `nc -zv 192.168.1.12 5000`
- Build failing due to private module access: enable `BUILDKIT_SSH` + ensure `ssh-agent` has keys; or vendor private modules with `go mod vendor` in build step.
- If you prefer not to use SSH forwarding, consider vendoring private deps into your repo before building.

Security note
- Do not bake credentials into images. Use BuildKit SSH forwarding or ephemeral secrets.
