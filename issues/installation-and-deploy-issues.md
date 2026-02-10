# Installation & Deploy Checks — Findings

Datum: 2026-02-09

Kurzüberblick:
- Ich habe README, `docs/getting-started.md`, `montage-ai.sh`, `scripts/ci-local.sh` und `deploy/config.env` geprüft.
- `./montage-ai.sh --help` lief und zeigt die erwarteten Kommandos.
- `scripts/check-hardcoded-registries.sh` meldet keine offensichtlichen harten Registry-Strings.
- `deploy/` enthält viele `<...>`-Platzhalter, wie erwartet; die Dokumentation weist klar auf das Ersetzen hin.
- `scripts/ci-local.sh` hat einen Syntax-Check (`bash -n`) bestanden.

Gefundene Punkte / Empfehlungen (nicht-blockierend):

1) Cluster-Placeholder-Ersetzung (erwartet, aber kritisch)
- Viele Manifeste und Beispielkonfigurationen enthalten `<IMAGE_FULL>`, `<REGISTRY_URL>`, `<MONTAGE_HOSTNAME>` und ähnliche Platzhalter.
- Empfehlung: Ergänze ein kurzes Skript `deploy/k3s/validate-config.sh` oder erweitere `pre-flight-check.sh`, das automatisch prüft und eine checklist-basierte Ausgabe liefert (z.B. "No placeholders remain").

2) `setup.sh` Docker-Compose-Prüfung
- `scripts/setup.sh` prüft Docker und `docker-compose` in einer leicht inkonsistenten Bedingung (prüft `command -v docker || command -v docker-compose` und dann `docker compose version`). Funktional, aber verwirrend.
- Empfehlung: Vereinheitliche Prüfung auf Compose v2: zuerst prüfen ob `docker` vorhanden ist, dann `docker compose version`.

3) Remote/Cloud LLMs und `cgpu`
- `montage-ai.sh` startet `cgpu` automatisch für einige Befehle; das ist praktisch, aber können Reproduzierbarkeit/Idempotenz beeinträchtigen, wenn `cgpu` nicht installiert ist.
- Empfehlung: Erwähne in der Getting-Started klar, welche features optional sind (cgpu, gemini, mediapipe) und wie man in Docker-only Setups deterministisch ohne sie läuft (z.B. `CGPU_ENABLED=false`).

4) Test/CI side-effects
- `scripts/ci-local.sh` kann `pipx`/`pip`-Installationen auslösen und `uv sync` ausführen (Netzwerk/Install-IO). Das ist normal für ein dev-CI, aber erwähnenswerte Seiteneffekte.
- Empfehlung: Dokumentiere einen "dry-run" oder einen `CI_DRY_RUN=1` Modus für lokale Syntax-/static checks ohne Pakete zu installieren.

5) Docs — Lesbarkeit & Idempotenz
- Dokumentation ist insgesamt gut strukturiert und ausführlich.
- Idempotenz: Cluster-Deploy-Anweisungen erfordern manuelles Ersetzen von Platzhaltern — akzeptabel, aber ein `render`-Schritt (envsubst) oder ein templating helper wäre hilfreich.

Vorgeschlagtes Issue-Template (falls gewünscht als GitHub-Issue):
- Titel: "Improve deploy config validation and add dry-run for local CI"
- Body: Kurze Zusammenfassung + Reproduktionsschritte + vorgeschlagene PR-Inhalte (validate script + CI dry-run env var + small docs update)

---

Wenn du möchtest, kann ich:
- das vorgeschlagene `deploy/k3s/validate-config.sh`-Skript anlegen (kleine Bash-Datei, grep auf `<[A-Z_]+>` und klare exit-codes),
- oder ein lokales Issue direkt auf GitHub öffnen (brauche Erlaubnis, das zu tun).

---

# Addendum (10.02.2026) - Verification Run

I have performed a verification run on a local ARM64 machine with Docker an K3d.

Findings:

1) **ARM64 Feature Parity (MediaPipe)**
   - Docker build works on ARM64.
   - Runtime warning: `[WARN] MediaPipe not installed. Auto Reframe will fallback to center crop.` - Expected due to lack of aarch64 wheels for MediaPipe in default configuration.
   - Recommendation: Explicitly document this feature limitation for ARM users in `getting-started.md`.

2) **Default Configuration 'arch' Mismatch**
   - `deploy/k3s/config-global.yaml.example` defaults to `arch: "amd64"`.
   - On an ARM64 cluster, `make pre-flight` warns about architecture mismatch (`[OK] Cluster node architecture(s): arm64 ... Verify these match your config-global.yaml`) but DOES NOT fail.
   - Deployment proceeds but pods remain Pending (node affinity mismatch) without clear error message in deploy output.
   - Recommendation: Make pre-flight check fail (exit 1) on architecture mismatch to prevent "silent" deployment hanging.

3) **'verify-deployment' False Positives**
   - When run inside Docker (as recommended), `./montage-ai.sh verify-deployment` flags read-only mounts (`/data/input`, etc.) as `⚠️ ... (not writable)` and counts them as issues.
   - These mounts are read-only by design in `docker-compose.yml`.
   - Recommendation: Update verification script to detect container environment or accept RO mounts for input/assets.

4) **Dependency Management Consistency**
   - `deploy/config.env` defines `UV_VERSION`, but `Dockerfile` uses standard `pip` for installation.
   - Recommendation: Align Dockerfile to use `uv` for faster and more consistent builds, or clarify that `uv` is for local dev only.

5) **Local Cluster Idempotence**
   - Verified that `make -C deploy/k3s deploy-cluster` is idempotent (resources unchanged on second run).
   - `scripts/ops/create-test-video.sh` is idempotent.

Status:
- Docker Build: ✅ Success
- Local Run (Preview): ✅ Success (generated video)
- Cluster Deploy (K3d): ✅ Success (after config adjustment)
