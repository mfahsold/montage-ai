# Cleanup & Retention

This document describes cleanup policies and how to run them.

Default policy
- Archive proxies (`/tmp/*proxy*.mp4`) larger than 200MB and older than 1 day to `data/output/archive/` (tar + gzip).
- Compress monitoring JSON files older than 90 days into `data/output/monitoring_archive/`.
- Rotate `data/output/render.log` into `data/output/archive/render.log.<UTC_TIMESTAMP>.gz` and truncate the current log.

How to run
- `make cleanup` — runs `scripts/cleanup.sh` which implements the above policy.
- `scripts/cleanup.sh` accepts environment overrides:

  - `SIZE_THRESHOLD` e.g. `SIZE_THRESHOLD=100M`
  - `AGE_DAYS` e.g. `AGE_DAYS=0`
  - `MON_AGE_DAYS` e.g. `MON_AGE_DAYS=30`

CI and Automation
- Consider adding a daily cron or CI pipeline to run `make cleanup` on a schedule and push archives to long-term storage if needed.

Rollbacks & Forensics
- Archives are stored in `data/output/archive/` and `data/output/monitoring_archive/`. Keep at least 30 days by policy.
- If you need a removed file restored, please consult infra for backup/restore options if available.