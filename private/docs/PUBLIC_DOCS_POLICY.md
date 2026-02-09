# Public Documentation Policy

**Last Updated:** 2026-01-09

## Public Repository Structure

This repository is public. The following guidelines apply:

### ✅ Public Docs (`docs/`, root `*.md`)

**Purpose:** User-facing documentation, technical guides, OSS compliance

**Allowed Content:**
- User guides (getting-started, features, configuration)
- Technical architecture (algorithms, models, architecture)
- Developer guides (contributing, llm-agents, CI)
- API reference (CLI_REFERENCE, PARAMETER_REFERENCE)
- OSS compliance (LICENSE, THIRD_PARTY_LICENSES, CODE_OF_CONDUCT)
- Public roadmap (roadmap/ROADMAP_2026.md)

**Examples:**
- `README.md` - Project overview
- `docs/getting-started.md` - Installation guide
- `docs/features.md` - Feature catalog
- `docs/architecture.md` - System design
- `CHANGELOG.md` - Version history

### 🔒 Private Docs (`private/`)

**Purpose:** Internal tracking, deployment logs, audit results, strategic planning

**Required Content:**
- Status documents (`*STATUS*.md`, `*SUMMARY*.md`)
- Audit reports (`*AUDIT*.md`, benchmark results)
- Deployment logs (`*DEPLOYMENT*.md`, `*K3S*.md`)
- Implementation reports (`*VERIFICATION*.md`, `*REPORT*.md`)
- Strategic planning (`*STRATEGY*.md`, `*BACKLOG*.md`)
- Session notes (`*SESSION*.md`, `*NEXT_STEPS*.md`)

**Examples:**
- `private/docs/status/REFACTORING_STATUS_2026.md`
- `private/docs/audits/PHASE5_STABILITY_AUDIT.md`
- `private/archive/TODO_PRIORITIES_2025.md`

### Pre-Commit Check

Before every commit:

```bash
# Fail if internal docs in root or docs/
find . -maxdepth 1 -type f \( \
  -name "*STATUS*" -o \
  -name "*STRATEGY*" -o \
  -name "*DEPLOYMENT*" -o \
  -name "*SESSION*" -o \
  -name "*AUDIT*" -o \
  -name "*VERIFICATION*" -o \
  -name "*REPORT*" -o \
  -name "*SUMMARY*" \
\) 2>/dev/null && echo "❌ Internal docs in root!" && exit 1

find docs/ -type f \( \
  -name "*STATUS*" -o \
  -name "*STRATEGY*" -o \
  -name "*DEPLOYMENT*" -o \
  -name "*SESSION*" -o \
  -name "*AUDIT*" \
\) 2>/dev/null && echo "❌ Internal docs in docs/!" && exit 1

echo "✅ No internal docs in public areas"
```

### GitHub Pages (`docs/`)

GitHub Pages serves content from `docs/`:
- Only user-facing documentation
- No internal status/strategy docs
- No deployment credentials
- No audit results

### Optional: Pre-Push Hook

```bash
# .git/hooks/pre-push
#!/bin/bash
./scripts/check_public_docs.sh || exit 1
```

---

**Enforcement:** Manual review + automated checks (pre-push hook). CI can call `scripts/check_public_docs.sh` if enabled.
