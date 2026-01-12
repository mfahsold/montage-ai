# Business Plan: DevOps & Registry Resilience

## Executive summary
- Objective: Reduce developer friction, minimize downtime during registry outages, and avoid recurring CI costs while maintaining reproducible builds and secure delivery.
- Short term: centralize configuration, provide robust fallback workflows, and enable local CI/self-hosted runners to avoid GH Actions costs.
- Medium term: implement canary-deploy workflows, image promotion, and automated registry failover strategies.

## Key initiatives
1. Centralized configuration
   - Single source of truth: `deploy/config.env` for registry host/port, image names, service ports. This allows quick redirection to local registries or fallbacks.

2. Cost control for CI
   - Prefer `make ci-local` on developer machines or self-hosted runners for PR verification. GitHub workflows set to `workflow_dispatch` only.

3. Registry resilience & availability
   - Provide a small, documented set of fallbacks: local registry container, `docker save` -> node, or GHCR fallback.
   - Monitor registry availability and alert engineering on outages.

4. Reproducibility & security
   - Adopt `uv.lock` after private extras are resolved; pin UV versions across environments.
   - Keep secrets out of repo; use ephemeral credentials and OIDC/attestations for image provenance.

## Success metrics
- Mean Time To Recovery (MTTR) for registry outages < 30 minutes (via documented fallback procedures).
- % of PRs validated via `make ci-local` or self-hosted runners > 95% (reduces GH Action usage). 
- Time to canary deploy after successful build < 10 minutes (from build completion to canary running).

## Risks & mitigations
- Private extras (cgpu): need private index or vendorized packages. Mitigation: vendor critical optional deps and document private index process.
- Security: avoid storing credentials in repo; use secrets or ephemeral build-time auth.

## Next steps (business-owner tasks)
- Approve short-term budget for a small self-hosted runner if desired.
- Confirm policy for pushing to GHCR as a permitted fallback registry.
- Review and sign-off on the Operational Runbook (deploy/runbook updates committed in `docs/`).
