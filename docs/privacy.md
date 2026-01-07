# Privacy Policy

**Last updated:** 2026-01-07

Montage AI is a local-first, privacy-first project. We minimize data collection and avoid cookies by default.

## Who we are

- Project: Montage AI (open source)
- Contact: [mfahsold@me.com](mailto:mfahsold@me.com)

## What we collect

- When analytics is enabled (see below), we collect anonymous page events (pageviews, referrers, device type). No cookies, no persistent identifiers, no IP storage beyond coarse, anonymized metrics.
- No user-generated media, transcripts, or project data leave your machine unless you explicitly upload them elsewhere.

## Analytics (optional, privacy-friendly)

- Tool: Plausible Analytics (cookie-free, GDPR-friendly).
- Mode: Anonymized events only; no cross-site tracking; no ad retargeting.
- Self-hosting: Recommended. If using the cloud service, data is processed by Plausible Insights OÜ (EU). Configure the endpoint accordingly.
- Opt-out: Disable analytics by removing the script or setting the Web UI env flag `ANALYTICS_ENABLED=false`.

## Data retention

- Analytics: Kept per your Plausible/Umami configuration. For self-hosted, you control retention.
- Logs: The Web UI logs stay local unless you forward them elsewhere.

## Your choices

- Run fully offline: default behavior when analytics is disabled.
- Clear cache/output: Remove `data/output/` and any browser cache to delete generated artifacts locally.

## Security

- Local-first processing; cloud GPU/offloading is optional and under your control.
- Use trusted networks and enable HTTPS/ingress TLS when deploying to a cluster.

## Contact

Questions or requests: [mfahsold@me.com](mailto:mfahsold@me.com)
