Claude Code Router — org/user-level deployment (musistudio/claude-code-router)

Quick guide to run the router centrally and point Montage AI at it.

Why run at org/user level?
- Centralized key storage and rotation (GitHub org secrets or K8s Secret).
- Single observable endpoint for billing, rate limits and auditing.

Recommended flow
1. Create a router instance (K8s / Docker / cloud).
2. Store the upstream Together API key in your org's secret manager (GitHub Secrets, AWS/GCP secrets).
3. Point Montage AI's `OPENAI_API_BASE` to the router (e.g. `https://claude-router.myorg.internal/v1`) and use `OPENAI_API_KEY` from the secret.

Examples
- GitHub Actions (use repo/org secret `TOGETHER_API_KEY`):

```yaml
env:
  OPENAI_API_BASE: "https://claude-router.myorg.internal/v1"
  OPENAI_API_KEY: "${{ secrets.TOGETHER_API_KEY }}"

steps:
  - name: Run preview
    run: |
      ./montage-ai.sh preview
```

- Kubernetes (create secret):

```bash
kubectl create secret generic claude-router-secret \
  --from-literal=TOGETHER_API_KEY="$TOGETHER_API_KEY" \
  -n montage-ai
kubectl apply -f deploy/claude-code-router/claude-deployment.yaml
```

Security notes
- NEVER commit API keys to the repository.
- Limit secret scope (use org-level secrets for shared infra, repo-level for per-repo usage).
- Rotate keys after adding them to CI or cluster.

References
- musistudio/claude-code-router: https://github.com/musistudio/claude-code-router
