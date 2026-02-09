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
