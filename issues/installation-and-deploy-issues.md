# Installation and Deployment Checks - Findings

Date: 2026-02-09

Overview:
- Reviewed `README.md`, `docs/getting-started.md`, `montage-ai.sh`, `scripts/ci-local.sh`, and `deploy/config.env`.
- `./montage-ai.sh --help` works and shows expected commands.
- `scripts/check-hardcoded-registries.sh` reported no obvious hardcoded registry strings.
- `deploy/` contains placeholder tokens (`<...>`) as expected, and docs already describe replacing them.
- `scripts/ci-local.sh` passes syntax check (`bash -n`).

Findings and recommendations (non-blocking):

1) Cluster placeholder replacement (expected but critical)
- Many manifests and examples still contain `<IMAGE_FULL>`, `<REGISTRY_URL>`, `<MONTAGE_HOSTNAME>` and similar placeholders.
- Recommendation: Add `deploy/k3s/validate-config.sh` or extend `pre-flight-check.sh` to validate placeholders and print checklist-style output (for example, "No placeholders remain").

2) `setup.sh` Docker Compose check
- `scripts/setup.sh` mixes Docker and Compose checks in a slightly inconsistent way (`command -v docker || command -v docker-compose`, then `docker compose version`).
- Recommendation: Standardize on Compose v2 check: verify `docker` first, then run `docker compose version`.

3) Remote/cloud LLMs and `cgpu`
- `montage-ai.sh` can auto-start `cgpu` for some commands, which is useful but may impact reproducibility/idempotence if `cgpu` is missing.
- Recommendation: Clarify in Getting Started which features are optional (`cgpu`, Gemini, MediaPipe) and how to run deterministic Docker-only mode (for example `CGPU_ENABLED=false`).

4) Test/CI side effects
- `scripts/ci-local.sh` can trigger `pipx`/`pip` installs and `uv sync` (network and install I/O). This is normal for local CI, but worth documenting.
- Recommendation: Document a dry-run mode (`CI_DRY_RUN=1`) for local syntax/static checks without package installation.

5) Docs readability and idempotence
- Docs are generally clear and structured.
- Cluster deployment still requires manual placeholder replacement. Acceptable, but `envsubst` rendering or a small templating helper would improve reliability.

Suggested issue template (if needed on GitHub):
- Title: "Improve deploy config validation and add dry-run for local CI"
- Body: short summary + repro steps + proposed PR scope (validation script + CI dry-run env var + docs update)

---

If needed, I can:
- create `deploy/k3s/validate-config.sh` (small Bash script, regex check for `<[A-Z_]+>`, clear exit codes),
- or open a GitHub issue directly (with your approval).

---

# Addendum (2026-02-10) - Verification Run

Performed an additional verification run on a local ARM64 machine with Docker and K3d.

Findings:

1) **ARM64 feature parity (MediaPipe)**
   - Docker build works on ARM64.
   - Runtime warning: `[WARN] MediaPipe not installed. Auto Reframe will fallback to center crop.`
   - This is expected due to missing aarch64 wheels in the default setup.
   - Recommendation: Document this limitation explicitly in `docs/getting-started.md`.

2) **Default `arch` mismatch risk**
   - `deploy/k3s/config-global.yaml.example` defaults to `arch: "amd64"`.
   - On ARM64 clusters, `make pre-flight` warns but does not fail.
   - Deployment can continue with pods stuck in Pending (node affinity mismatch) without strong deploy-time signaling.
   - Recommendation: Fail pre-flight (`exit 1`) on architecture mismatch.

3) **`verify-deployment` false positives**
   - In Docker mode, `./montage-ai.sh verify-deployment` can report read-only mounts (`/data/input`, etc.) as issues.
   - Those mounts are read-only by design in `docker-compose.yml`.
   - Recommendation: Detect container mode or allow read-only mounts for input/assets.

4) **Dependency management consistency**
   - `deploy/config.env` defines `UV_VERSION`, while `Dockerfile` currently installs via `pip`.
   - Recommendation: Align Dockerfile with `uv` usage or clearly document `uv` as local-dev-only.

5) **Local cluster idempotence**
   - `make -C deploy/k3s deploy-cluster` is idempotent (no changes on second run).
   - `scripts/ops/create-test-video.sh` is idempotent.

Status:
- Docker build: success
- Local preview run: success (video generated)
- K3d deployment: success (after config adjustment)
