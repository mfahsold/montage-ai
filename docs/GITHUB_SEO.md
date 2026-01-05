# GitHub Repository Optimization

## Status (2026-01-05)
- ✅ Topics set and homepage configured in repo About.
- ✅ Description tuned for keywords: "Free, open-source AI video editor for rough cuts. Beat-sync, story arcs, OTIO/EDL export. Offline Descript alternative; local-first, privacy-first."
- 🟠 Ranking: Not in top 10 for GitHub queries "ai video editor" or "descript alternative" (gh search snapshot below).
- 🟠 Validation: OG/Twitter and Lighthouse checks still to run/record.

## Baseline Measurements (2026-01-05)
- GitHub search snapshot:
	- `gh search repos "ai video editor" --limit 10` → montage-ai not in top 10.
	- `gh search repos "descript alternative" --limit 10` → no visible hits.
- Site metadata: og/twitter/canonical/JSON-LD present in docs/index.html; og-image.svg published.
- README headline/intro now include "AI video editor", "rough cuts", and "Offline Descript alternative" keywords.

## Monitor & Re-run
- GitHub search (position check):
	- `gh search repos "ai video editor" --limit 20 --json fullName,stargazersCount | jq -r '.[]|"\(.fullName) | ★\(.stargazersCount)"'`
	- `gh search repos "descript alternative" --limit 20 --json fullName,stargazersCount`
- Social preview:
	- https://opengraph.xyz/ and https://cards-dev.twitter.com/validator with `https://mfahsold.github.io/montage-ai/` → confirm og-image.svg, title, description.
- Lighthouse SEO (desktop or mobile):
	- `npx lighthouse https://mfahsold.github.io/montage-ai --only-categories=seo --quiet --view`

## Next SEO Tasks
- Add short landing/blog posts targeting long-tail: e.g., offline Descript alternative (already in docs/blog/descript-alternative.md), OTIO handoff, beat-sync shorts.
- Consider per-page OG images for key docs (getting-started, features) if shared individually.
- Add JSON-LD breadcrumbs or FAQ to docs/index.html once doc tree grows.
