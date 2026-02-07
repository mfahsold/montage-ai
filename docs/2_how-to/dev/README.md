---
title: "Developer Workflow Hub"
summary: "Entry point for development workflows"
updated: 2026-02-06
---

# Developer Workflow Hub

## Canonical Multi-Arch Build (fast path)
- Run canonical build: `BUILD_MULTIARCH=true BUILD_PLATFORMS=linux/amd64,linux/arm64 ./deploy/k3s/build-and-push.sh`
- The build script auto-selects an existing native distributed builder (`montage-multiarch`, `simple-builder`, etc.) and enables registry cache by default.
- Optional overrides: `BUILDER_NAME=<builder>`, `BUILDER_CANDIDATES=<csv>`, `CACHE_REF=<registry/ref>`, `USE_REGISTRY_CACHE=false`.
- Inner loop speed: set `BUILD_MULTIARCH=false` for local amd64-only builds; run the command above for release tags.

## Deploy & Verify
- Deploy: `./deploy/k3s/deploy.sh cluster`
- Inspect image: `docker buildx imagetools inspect 192.168.1.12:30500/montage-ai:TAG`
- Smoke in cluster: `./scripts/ci/run-in-cluster-smoke.sh`

## Job submission (cluster API)
- Port-forward: `kubectl -n montage-ai port-forward svc/montage-ai-web 18080:80`
- Submit: `MONTAGE_API_BASE=http://localhost:18080 ./montage-ai.sh jobs submit --style dynamic --prompt "<prompt>" --quality-profile high`
- Status: `MONTAGE_API_BASE=http://localhost:18080 ./montage-ai.sh jobs status <JOB_ID>`
