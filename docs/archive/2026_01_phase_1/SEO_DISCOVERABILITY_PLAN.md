# Montage AI - SEO & Discoverability Plan

> **Goal:** Maximize organic discovery of Montage AI across GitHub, search engines, and developer communities.

---

## Current Status Audit

### GitHub Repository ✅ Good

| Element | Status | Notes |
|---------|--------|-------|
| README.md | ✅ Excellent | Clear value prop, comparison table, quick start |
| LICENSE | ✅ Present | PolyForm Noncommercial |
| CONTRIBUTING.md | ✅ Present | Clear contribution guide |
| CHANGELOG.md | ✅ Detailed | Follows Keep a Changelog format |
| Issue Templates | ✅ Present | Bug report + Feature request |
| PR Template | ✅ Present | Standard template |
| FUNDING.yml | ✅ Added | GitHub Sponsors enabled |

### GitHub Pages ✅ Good

| Element | Status | Notes |
|---------|--------|-------|
| index.html | ✅ Excellent | Full SEO meta tags, JSON-LD schema |
| Open Graph | ✅ Present | og:title, og:description, og:image |
| Twitter Card | ✅ Present | Large image card |
| Canonical URL | ✅ Present | Proper canonical link |
| robots.txt | ✅ Added | Allows all, includes sitemap |
| sitemap.xml | ✅ Added | Key pages listed |
| 404.html | ✅ Added | Custom error page |
| .nojekyll | ✅ Present | Static site, no Jekyll processing |
| GitHub Actions | ✅ Added | Auto-deploy on push to docs/ |

### Documentation ✅ Good

| Element | Status | Notes |
|---------|--------|-------|
| Getting Started | ✅ Complete | 5-minute onboarding |
| Features | ✅ Complete | Comprehensive feature list |
| Configuration | ✅ Complete | All env vars documented |
| Architecture | ✅ Complete | Developer reference |
| Troubleshooting | ✅ Complete | Common issues |
| Competitive Analysis | ✅ Complete | Market positioning |

---

## Discoverability Improvement Plan

### Phase 1: GitHub Optimization (Immediate)

#### 1.1 Repository Settings (Manual via GitHub UI)

```yaml
# Set in GitHub Repository Settings:
Description: "Free AI video editor: beat-sync montages, transcript editing, OTIO/EDL export. Local-first Descript alternative."

Website: "https://mfahsold.github.io/montage-ai/"

Topics:
  - video-editor
  - ai-video-editor
  - beat-sync
  - otio
  - ffmpeg
  - descript-alternative
  - transcript-editing
  - shorts-creator
  - davinci-resolve
  - premiere-pro
  - open-source
  - local-first
  - privacy-first
```

#### 1.2 Social Preview Image

Current: SVG (may not render on all platforms)

**Action:** Convert og-image.svg to PNG (1200x630) for universal compatibility.

```bash
# Option 1: Use Inkscape
inkscape docs/images/og-image.svg -o docs/images/og-image.png -w 1200 -h 630

# Option 2: Use rsvg-convert
rsvg-convert -w 1200 -h 630 docs/images/og-image.svg > docs/images/og-image.png
```

### Phase 2: Content Marketing (Week 1-2)

#### 2.1 Blog Posts / Landing Pages

Create targeted landing pages for high-value search terms:

| Page | Target Keywords | Priority |
|------|-----------------|----------|
| `docs/blog/descript-alternative.md` | descript alternative offline, local video editor | ✅ Done |
| `docs/blog/podcast-to-shorts.md` | podcast to youtube shorts, auto clip podcast | High |
| `docs/blog/davinci-resolve-workflow.md` | otio davinci resolve, ai rough cut | High |
| `docs/blog/premiere-handoff.md` | ai premiere pro plugin, auto edit premiere | Medium |
| `docs/blog/beat-sync-tutorial.md` | sync video to music beats, beat detection | Medium |

#### 2.2 README Improvements

- Add animated GIF/video demo (15-30 seconds)
- Add "Featured In" section (if any press coverage)
- Add star history badge
- Add download/install count badges

### Phase 3: Community Building (Week 2-4)

#### 3.1 Cross-Posting Strategy

| Platform | Content Type | Frequency |
|----------|--------------|-----------|
| **Reddit** | r/VideoEditing, r/Filmmakers, r/podcasting | 1x/week |
| **Hacker News** | Show HN post | 1x (at v1.0) |
| **Product Hunt** | Launch | 1x (at v1.0) |
| **Twitter/X** | Feature updates, tips | 2-3x/week |
| **YouTube** | Tutorial videos | 1x/month |
| **Dev.to / Hashnode** | Technical articles | 2x/month |

#### 3.2 Reddit Strategy

**Target Subreddits:**
- r/VideoEditing (300k+ members)
- r/Filmmakers (1M+ members)
- r/podcasting (200k+ members)
- r/selfhosted (400k+ members)
- r/opensource (100k+ members)
- r/DataHoarder (for local-first angle)

**Post Types:**
1. "Show off your project" threads
2. Answer questions about video automation
3. Share before/after montage examples
4. Tutorials with links to repo

### Phase 4: Technical SEO (Week 3-4)

#### 4.1 Backlink Strategy

| Source | Type | Status |
|--------|------|--------|
| Awesome Video lists | GitHub lists | Pending |
| OTIO ecosystem docs | Reference | Pending |
| FFmpeg wiki | Tool mention | Pending |
| cgpu project | Integration | Pending |

#### 4.2 GitHub Awesome Lists

Submit to:
- awesome-video
- awesome-ffmpeg
- awesome-self-hosted
- awesome-ai-tools

### Phase 5: Analytics & Iteration (Ongoing)

#### 5.1 Tracking Setup

```html
<!-- Add to index.html (privacy-respecting) -->
<script defer data-domain="mfahsold.github.io/montage-ai"
        src="https://plausible.io/js/script.js"></script>
```

Or use GitHub's built-in traffic insights.

#### 5.2 Key Metrics

| Metric | Target (3 months) |
|--------|-------------------|
| GitHub Stars | 500+ |
| Unique Clones | 100/week |
| Page Views (GH Pages) | 1000/month |
| Contributors | 5+ |
| Forks | 50+ |

---

## Quick Wins Checklist

- [x] robots.txt added
- [x] sitemap.xml added
- [x] 404.html added
- [x] FUNDING.yml added
- [x] GitHub Actions for Pages
- [x] Issue template config
- [x] German docs translated to English
- [ ] Add repo topics (manual: GitHub UI)
- [ ] Add repo description (manual: GitHub UI)
- [ ] Convert og-image to PNG
- [ ] Create demo GIF for README
- [ ] Submit to awesome-video list
- [ ] Write first Reddit post

---

## Recommended Repo Topics

Copy-paste for GitHub Settings:

```
video-editor, ai-video-editor, beat-sync, otio, ffmpeg, descript-alternative,
transcript-editing, shorts-creator, davinci-resolve, premiere-pro,
open-source, local-first, privacy-first, python, docker, kubernetes
```

---

## Content Calendar (Sample)

### Week 1
- [ ] Monday: Finish SEO assets (this document)
- [ ] Wednesday: Create demo GIF
- [ ] Friday: Post to r/selfhosted

### Week 2
- [ ] Monday: Write "Podcast to Shorts" blog post
- [ ] Wednesday: Post to r/podcasting
- [ ] Friday: Submit to awesome-self-hosted

### Week 3
- [ ] Monday: Create YouTube tutorial
- [ ] Wednesday: Post to r/VideoEditing
- [ ] Friday: Write dev.to article

### Week 4
- [ ] Monday: DaVinci Resolve workflow post
- [ ] Wednesday: Submit to awesome-video
- [ ] Friday: Evaluate metrics, iterate

---

## Summary

**Montage AI is already well-structured for discoverability.** The main gaps are:

1. **GitHub Topics** - Need to add manually in repo settings
2. **Social Image** - SVG should be converted to PNG
3. **Demo Media** - README needs a visual demo (GIF/video)
4. **Community Outreach** - Need active posting on Reddit/HN

The technical SEO foundation (meta tags, JSON-LD, sitemap, etc.) is solid.
