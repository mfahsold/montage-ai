# UI Consolidation - Strategy Implementation

**Date:** January 2026  
**Status:** Phase 1 Complete  
**Related:** [STRATEGY.md](STRATEGY.md), [features.md](features.md)

---

## Overview

This document describes the UI consolidation effort aligned with the strategic product vision: **"AI rough cut + social-ready output + pro handoff"** with a focus on outcome-based workflows instead of toggle overload.

## Before & After

### Before (Toggle Overload)
```
❌ 12 separate checkboxes
❌ Technical terminology (enhance, stabilize, upscale, cgpu, cgpu_gpu)
❌ Unclear which combinations work together
❌ No guidance on when to use each feature
❌ Overwhelming for new users
```

### After (Outcome-Based)
```
✅ 4 quality profiles (Preview, Standard, High, Master)
✅ 1 cloud acceleration toggle
✅ Categorized advanced features (collapsible)
✅ Clear value propositions
✅ Visual cards with emojis and descriptions
✅ Three workflow landing page
```

---

## New Landing Page

**Route:** `GET /`  
**Template:** `index_strategy.html`

### Layout

```
┌─────────────────────────────────────────────────┐
│  Montage AI                                      │
│  Edit like text. Cut like music. Deliver pro.   │
├─────────────────────────────────────────────────┤
│  [🏠 Local] [⚡ Fast] [✨ Polish] [🎬 Pro Export] │
├─────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ 🎬       │  │ 📝       │  │ 📱       │      │
│  │ Montage  │  │ Text     │  │ Shorts   │      │
│  │ Creator  │  │ Editor   │  │ Studio   │      │
│  └──────────┘  └──────────┘  └──────────┘      │
├─────────────────────────────────────────────────┤
│  Core Features (8 cards)                        │
└─────────────────────────────────────────────────┘
```

### Workflow Cards

Each card includes:
- **Icon** (emoji for quick recognition)
- **Title** (clear, action-oriented)
- **Subtitle** (one-line value prop)
- **Features list** (5 bullet points)
- **CTA button** (links to specific workflow)

---

## Quality Profiles

### Visual Design

Replaced dropdown with 4 visual cards:

```
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ 🚀            │ │ 📺            │ │ ✨            │ │ 🎬            │
│ Preview       │ │ Standard      │ │ High          │ │ Master        │
│ 360p Fast     │ │ 1080p Social  │ │ 1080p Pro     │ │ 4K Broadcast  │
│ Iteration     │ │ Media Ready   │ │ Delivery      │ │ Quality       │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

### Specifications

| Profile | Resolution | Enhance | Stabilize | Upscale | Use Case |
|---------|-----------|---------|-----------|---------|----------|
| 🚀 Preview | 360p | ❌ | ❌ | ❌ | Fast iteration, rough cut review |
| 📺 Standard | 1080p | ✅ | ❌ | ❌ | Social media, general use |
| ✨ High | 1080p | ✅ | ✅ | ❌ | Professional delivery |
| 🎬 Master | 4K | ✅ | ✅ | ✅ | Broadcast, cinema, archival |

### Implementation

**JavaScript (app.js)**:
```javascript
const QUALITY_PROFILES = [
    { 
        id: 'preview', 
        name: '🚀 Preview', 
        desc: '360p Fast Iteration',
        details: 'No enhancements. Quick rough cut review.',
        settings: { enhance: false, stabilize: false, upscale: false, resolution: '360p' }
    },
    // ... more profiles
];

function selectQualityProfile(profileId) {
    // Update hidden input
    document.getElementById('qualityProfile').value = profileId;
    // Update visual selection
    document.querySelectorAll('.quality-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`[data-profile="${profileId}"]`).classList.add('selected');
    updateRunSummary();
}
```

**Backend Integration**:
```javascript
function buildJobPayload() {
    const qualityProfile = getVal('qualityProfile') || 'standard';
    const profileSettings = QUALITY_PROFILES.find(p => p.id === qualityProfile)?.settings || {};
    
    jobData.enhance = profileSettings.enhance || false;
    jobData.stabilize = profileSettings.stabilize || false;
    jobData.upscale = profileSettings.upscale || false;
    // ...
}
```

---

## Cloud Acceleration

### Consolidation

**Before:**
- `□ Cloud GPU` (cgpu)
- `□ Upscale` (separate)
- LLM features scattered

**After:**
- `□ Cloud Acceleration` (single toggle)
  - Enables: AI upscaling, fast transcription, LLM direction
  - Auto-fallback to local processing

### Implementation

**Frontend:**
```javascript
const cloudAcceleration = getCheck('cloud_acceleration');
jobData.cgpu = cloudAcceleration;  // Enable LLM features
jobData.cgpu_gpu = cloudAcceleration && jobData.upscale;  // GPU only if upscaling
```

**Backend mapping** (handled in `app.py`):
```python
# When cloud_acceleration=true arrives:
cgpu_enabled = job_data.get('cgpu', False)
cgpu_gpu_enabled = job_data.get('cgpu_gpu', False)

# Fallback logic in montage_builder.py
if cgpu_enabled and not is_cgpu_available():
    logger.warning("Cloud acceleration unavailable, falling back to local")
    cgpu_enabled = False
```

---

## Toggle Reorganization

### Categories

#### Core (Always Visible)
- **Shorts Mode**: 9:16 Vertical + Smart Crop
- **Burn-in Captions**: Auto-transcribed subtitles
- **Export Timeline**: OTIO/EDL for NLEs
- **Cloud Acceleration**: Offload AI tasks to cloud GPU

#### Advanced AI (Collapsible)
- **LLM Clip Selection**: Semantic scene analysis
- **Creative Loop**: LLM refines cuts iteratively
- **Story Engine**: Narrative tension-based editing

#### Pro Export (Collapsible)
- **Generate Proxies**: Faster NLE editing
- **Preserve Aspect**: Letterbox vs crop

### UI Rendering

```html
<div id="toggles-container">
    <!-- Core toggles (always shown) -->
    <div>
        [Shorts Mode checkbox]
        [Captions checkbox]
        [Export Timeline checkbox]
        [Cloud Acceleration checkbox]
    </div>
    
    <!-- Advanced (collapsible) -->
    <details>
        <summary>🤖 Advanced AI Features (Optional)</summary>
        [LLM Clip Selection checkbox]
        [Creative Loop checkbox]
        [Story Engine checkbox]
        <div class="helper">⚠️ These features use LLM and increase processing time.</div>
    </details>
    
    <!-- Pro Export (collapsible) -->
    <details>
        <summary>🎬 Pro Export Options</summary>
        [Generate Proxies checkbox]
        [Preserve Aspect checkbox]
    </details>
</div>
```

---

## Route Structure

### New Routes

| Route | Template | Description |
|-------|----------|-------------|
| `GET /` | `index_strategy.html` | Landing page with 3 workflows |
| `GET /?legacy` | `index.html` | Legacy full-featured UI |
| `GET /montage` | `index.html` | Montage Creator (beat-sync, style presets) |
| `GET /transcript` | `transcript.html` | Text-based video editor |
| `GET /shorts` | `shorts.html` | Vertical video studio |
| `GET /v2` | `index_v2.html` | Prototype UI (outcome-based) |

### Implementation (`app.py`)

```python
@app.route('/')
def index():
    """Main landing page with workflow selection (strategy-aligned UI)."""
    if request.args.get('legacy'):
        return render_template('index.html', version=VERSION, defaults=DEFAULT_OPTIONS)
    return render_template('index_strategy.html', version=VERSION)

@app.route('/montage')
def montage_creator():
    """Montage Creator - beat-sync, story arc, style presets."""
    return render_template('index.html', version=VERSION, defaults=DEFAULT_OPTIONS)
```

---

## Design System

### Colors

```css
--accent: #7C3AED;          /* Purple - primary actions */
--accent-soft: rgba(124, 58, 237, 0.15);
--success: #10B981;          /* Green - success states */
--warning: #F59E0B;          /* Amber - warnings */
--error: #EF4444;            /* Red - errors */
--bg: #0a0a0f;              /* Dark background */
--card-bg: #1a1a24;         /* Card background */
--border: #2a2a3a;          /* Borders */
--fg: #e0e0e8;              /* Foreground text */
--muted: #8a8a9a;           /* Muted text */
```

### Typography

- **Headlines**: Space Grotesk (modern, geometric sans-serif)
- **Body**: Inter (readable, professional)
- **Monospace**: Space Mono (code, terminal-style elements)

### Component Patterns

#### Workflow Card
```css
.workflow-card {
    background: var(--card-bg);
    border: 2px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.workflow-card:hover {
    border-color: var(--accent);
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(124, 58, 237, 0.2);
}
```

#### Quality Profile Card
```css
.quality-card {
    background: var(--card-bg);
    border: 2px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.2s;
}

.quality-card.selected {
    border-color: var(--accent);
    background: var(--accent-soft);
}
```

---

## Testing

### Manual Testing Checklist

- [ ] Landing page loads at `/`
- [ ] Three workflow cards are visible and clickable
- [ ] `/montage` route shows Montage Creator
- [ ] `/transcript` route shows Text Editor
- [ ] `/shorts` route shows Shorts Studio
- [ ] Quality profile cards render correctly
- [ ] Clicking quality profile updates selection
- [ ] Cloud acceleration toggle appears
- [ ] Advanced/Pro sections are collapsible
- [ ] Job payload includes quality profile settings
- [ ] Cloud acceleration maps to cgpu flags

### Validation Script

```python
# Run: python3 -c "exec(open('test_ui_changes.py').read())"
from pathlib import Path

# Test route changes
with open('src/montage_ai/web_ui/app.py', 'r') as f:
    content = f.read()
    assert 'index_strategy.html' in content
    assert '/montage' in content

# Test JS changes
with open('src/montage_ai/web_ui/static/app.js', 'r') as f:
    content = f.read()
    assert '🚀 Preview' in content
    assert 'cloud_acceleration' in content

# Test template exists
assert Path('src/montage_ai/web_ui/templates/index_strategy.html').exists()

print('✅ All validations passed')
```

---

## Migration Guide

### For Users

**Old UI** (at `/?legacy`):
- All 12 toggles still available
- Technical terminology
- Dropdown quality selector

**New UI** (at `/`):
- Landing page with workflow selection
- Visual quality profiles
- Simplified toggles

**Migration path**: Use `/?legacy` if you need the old interface. New UI is recommended.

### For Developers

**Payload changes**:
```javascript
// Old
{
  enhance: true,
  stabilize: true,
  upscale: true,
  cgpu: true,
  cgpu_gpu: true
}

// New (equivalent)
{
  quality_profile: 'master',
  cloud_acceleration: true
  // enhance/stabilize/upscale set automatically from profile
  // cgpu/cgpu_gpu set automatically from cloud_acceleration
}
```

**Backend compatibility**: Both formats supported. New UI sends `quality_profile` which is expanded to individual flags in `buildJobPayload()`.

---

## Next Steps

### Immediate (Phase 1 Completion)

1. **Time Estimates**: Add to quality profile cards
   ```
   🚀 Preview: <3 min for 10 min input
   📺 Standard: ~5 min
   ✨ High: ~10 min
   🎬 Master: ~20 min
   ```

2. **Tooltips**: Add hover tooltips explaining each profile

3. **Mobile Responsive**: Improve stacking on mobile devices

4. **Preview-First Default**: Make preview generation automatic

### Phase 2 Priorities

1. **Caption Styles**: Add style selector in Shorts Studio
2. **Highlight Detection**: MVP with audio energy analysis
3. **Phone Frame Preview**: Safe zone overlay for Shorts
4. **Better Error Handling**: User-friendly error messages

---

## References

- [STRATEGY.md](STRATEGY.md) - Full strategic document
- [features.md](features.md) - Feature documentation
- [BACKLOG.md](BACKLOG.md) - Upcoming work
- [architecture.md](architecture.md) - System architecture

---

*Last updated: January 2026*
