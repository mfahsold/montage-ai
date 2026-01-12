# Markdownlint fixes — summary

Date: 2026-01-12
Author: automation (GitHub Copilot)

Summary
- Ran: `npx -y markdownlint-cli2 "docs/**/*.md"` to audit documentation formatting.
- Fixed: `docs/STYLE_QUICK_REFERENCE.md` — resolved MD022, MD031, MD032, MD040, MD060, MD013 warnings by adding blank lines, specifying fenced code languages, normalizing table pipe spacing, and wrapping long lines.

Remaining findings
- The lint run reports many remaining warnings across `docs/` (tables missing spaces, fenced-code blocks without language, headings missing surrounding blank lines, and line-length violations).
- Top affected files (sample):
  - `docs/performance-tuning.md` (multiple MD022/MD031/MD032/MD013/MD060 issues)
  - `docs/responsible_ai.md` (many MD013 line-lengths + MD012)
  - `docs/README.md` (MD060 table column style issues)
  - Several `docs/roadmap/*.md` files (tables, list spacing, and line-lengths)

Next steps
1. Create a dedicated branch (e.g., `docs/markdownlint-fixes`) to incrementally fix these files in small PRs grouped by file/area.
2. Prioritize files that block CI (README, performance-tuning, responsible_ai) and those that cause many violations.
3. For long paragraphs, wrap lines at ~80 chars; for fenced code blocks, add language annotations (````bash`, ````python`, etc.); add blank lines around headings and fenced blocks; normalize table separators to use single spaces around pipes.
4. Optionally relax `MD013` in a config if project prefers longer lines for prose.

How to re-run
- `npx -y markdownlint-cli2 "docs/**/*.md"` (shows file-by-file issues)

Notes
- This run was non-exhaustive; please run locally and open small PRs for each group of fixes.
- I can continue batching fixes if you want — tell me which files to prioritize and I will open PRs with incremental commits.
