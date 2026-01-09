# Deploy Folder Overview

Use this directory as the single source of truth for deployment assets.

## What lives here

- CLUSTER_WORKFLOW.md — end-to-end k3s workflow
- k3s/ — kustomize overlays and manifests
- CONFIGURATION.md — deployment config reference
- config.env, config-global.yaml — baseline env/config samples

## How to deploy (quick map)

1) Read CLUSTER_WORKFLOW.md for prerequisites and steps.
2) Apply manifests from k3s/ using kubectl/kustomize.
3) Keep edits in deploy/; avoid duplicating manifests in repo root.

## Notes

- Example-only manifests should stay here; remove or archive duplicates in the repo root when replaced.
- For on-call runbooks see docs/KUBERNETES_RUNBOOK.md.
