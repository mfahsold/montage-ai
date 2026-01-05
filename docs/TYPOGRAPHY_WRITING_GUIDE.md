# Montage AI — Typography & Writing Style Guide

**Version**: 2.0 (January 2026)  
**Language**: English (Global)  
**Tone**: Technical, elegant, concise

---

## 🎯 Core Principles

### 1. **Elegance Through Brevity**
- Say more with fewer words
- Remove redundancy
- Use active voice
- Avoid marketing fluff

### 2. **Technical Clarity**
- Assume users understand video/audio
- Reference industry standards (OTIO, EDL, FFmpeg)
- Explain features by outcome, not mechanism

### 3. **Consistent Voice**
- Professional but approachable
- Direct, not corporate
- Technical precision over casual tone

### 4. **Accessible Writing**
- Short paragraphs (2-3 sentences max)
- Clear hierarchy (H1 → H2 → H3)
- One idea per sentence

---

## 📐 Typography System

### **Font Stack**
```css
/* Monospace (Technical Content) */
font-family: 'Share Tech Mono', monospace;

/* Fallback for descriptions */
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
```

### **Size Scale (Responsive)**
| Element | Desktop | Mobile | CSS Variable |
|---------|---------|--------|--------------|
| **H1 (Title)** | 2.5rem | 1.8rem | `--fs-2xl` |
| **H2 (Section)** | 1.8rem | 1.4rem | `--fs-xl` |
| **H3 (Card Title)** | 1.1rem | 1rem | `--fs-lg` |
| **H4 (Subsection)** | 1rem | 0.9rem | `--fs-base` |
| **Body Text** | 0.9rem | 0.8rem | `--fs-base` |
| **Small Label** | 0.75rem | 0.7rem | `--fs-sm` |
| **Tiny Label** | 0.7rem | 0.65rem | `--fs-xs` |

### **Line Height**
```css
--lh-tight: 1.1;    /* Headers, technical labels */
--lh-base: 1.5;     /* Body text, descriptions */
--lh-relaxed: 1.8;  /* Long-form content */
```

### **Letter Spacing**
```css
--ls-tight: -0.02em;   /* Headlines */
--ls-normal: 0em;       /* Body */
--ls-wide: 0.05em;      /* All-caps labels */
```

---

## 🎨 Color System (UI Text)

### **Primary Colors**
| Usage | Color | Value | CSS Var |
|-------|-------|-------|---------|
| Headlines | Electric Blue | `#0055ff` | `--primary` |
| Accent text | Neon Orange | `#ff5500` | `--secondary` |
| Body text | Light Gray | `#e0e0e0` | `--fg` |
| Subtle text | Medium Gray | `#888888` | `--muted-fg` |

### **Status Colors**
```css
--success: #00ff88;   /* Green - for "COMPLETED", "ONLINE" */
--warning: #ffcc00;   /* Yellow - for "PROCESSING", "PENDING" */
--error: #ff3333;     /* Red - for "ERROR", "OFFLINE" */
```

### **Background**
```css
--bg: #050505;        /* Main background (almost black) */
--card-bg: #0d0d0d;   /* Card backgrounds (slightly lighter) */
--muted: #1a1a1a;     /* Muted backgrounds (input areas) */
```

---

## ✍️ Writing Style Rules

### **Headlines**
```
✅ GOOD:
  "Montage Creator" (clear, concise)
  "Beat-Sync Your Footage" (action-oriented)
  
❌ BAD:
  "The Montage Creator Module for Creating Montages"
  "Advanced Beat Synchronization Features"
```

### **Descriptions (≤60 characters per line)**
```
✅ GOOD:
  "Upload footage. Let AI assemble cinematic cuts. 
   Choose from 7+ styles."
  
❌ BAD:
  "This feature allows you to upload your video footage, 
   and the artificial intelligence will intelligently assemble..."
```

### **Buttons & CTAs**
```
✅ GOOD:
  "Launch Creator" (verb + noun)
  "Create Short" (action)
  "View Results" (clear outcome)
  
❌ BAD:
  "Click Here"
  "Submit"
  "Go to the creator page"
```

### **Feature Lists**
```
✅ GOOD:
  • Beat-sync to music (librosa)
  • Story arc narrative (5-phase)
  • OTIO export (Pro Resolve)
  
❌ BAD:
  • Advanced beat synchronization algorithms
  • Sophisticated narrative structure generation
  • Professional timeline interchange format export
```

### **Technical Terms**
```
✅ Always write:
  OTIO (not "OpenTimelineIO" on first use)
  EDL (not "Edit Decision List")
  FFmpeg (not "FFmpeg engine")
  
✅ Explain once per page:
  "OTIO (OpenTimelineIO) — modern NLE exchange format"
```

---

## 📝 Common Phrases & Alternatives

### **Instead of...**
| Wordy | Concise |
|-------|---------|
| "The system will process your footage" | "AI processes your footage" |
| "Click the button to initiate the render" | "Start render" |
| "In order to achieve this result" | "To" |
| "It is important to note that" | "Note:" |
| "Allows you to" | "[Verb directly]" |
| "Provides the ability to" | "[Verb directly]" |
| "A wide variety of" | "Multiple" or "7+" |

---

## 🏷️ UI Component Text Style

### **Navigation Labels**
- All caps, no spaces
- Examples: `DASHBOARD`, `CREATOR`, `GALLERY`, `SETTINGS`

### **Card Titles**
- Title Case or UPPERCASE
- 1-3 words max
- Examples: `MONTAGE CREATOR`, `SHORTS GENERATOR`, `SYSTEM STATUS`

### **Button Text**
- Verb + Optional Noun
- All caps if status/action-heavy: `START RENDER`, `DOWNLOAD`
- Title Case if exploration: `View Gallery`, `Settings`

### **Status Badges**
- All caps
- Short (1-2 words)
- Examples: `SYSTEM ONLINE`, `GPU READY`, `COMPLETED`, `PROCESSING`

### **Labels & Tags**
- All caps, letter-spaced
- Small font
- Examples: `STORAGE`, `QUEUE`, `FEATURES`

---

## 📋 Page-by-Page Templates

### **Dashboard (index.html)**

**Header:**
```
DASHBOARD
AI Post-Production Assistant
// Polish, Don't Generate.
```

**Feature Cards:**
```
Title: [VERB] [NOUN]
Desc: [Action]. [Outcome]. [Who benefits].

Example:
"Montage Creator"
"Upload footage and let AI assemble cinematic cuts. 
Choose from 7+ styles and customize with your vision."
```

**Status Section:**
```
Title: SYSTEM STATUS
Metrics: STORAGE, QUEUE (all caps)
Status: ONLINE, IDLE, PROCESSING (short, clear)
```

### **Creator (montage.html)**

**Step Headers:**
```
"STEP 1: [ACTION]"
"STEP 2: [CONFIGURE]"
...
```

**Step Descriptions:**
```
[What you're doing]. [Why it matters].
[Optional: example or tip]

Example:
"Choose the editing rhythm and visual language. 
Each style dictates pacing, transitions, and cut timing."
```

**Option Groups:**
```
Title: [NOUN PHRASE] (all caps)
Item: [Feature name] — [brief benefit]

Example:
"AI UPSCALE (4K) — Enhance resolution using Real-ESRGAN. 
Increases file size and render time."
```

### **Documentation (Markdown)**

**Top Level:**
```markdown
# Feature Name
> [Philosophy or context quote]

**What:** [One sentence definition]
**Why:** [One sentence benefit]
**How:** [Steps or mechanism]
```

---

## 🔤 Capitalization Rules

### **UPPERCASE for:**
- Page titles: `DASHBOARD`, `CREATOR`, `GALLERY`
- Navigation: `DASHBOARD`, `CREATOR`, `GALLERY`, `SETTINGS`
- Status/Badges: `SYSTEM ONLINE`, `COMPLETED`, `PROCESSING`
- Labels: `STORAGE`, `QUEUE`, `FEATURES`, `STEP 1`
- Technical terms in labels: `OTIO`, `EDL`, `GPU`

### **Title Case for:**
- Button text: `Launch Creator`, `View Gallery`, `Settings`
- Feature names: `Montage Creator`, `Shorts Generator`
- Card titles (if not technical): `System Status`, `Features`

### **lowercase for:**
- Descriptions and body text
- Unless it's a proper noun or product name

---

## 💡 Quick Reference — Good Examples

### ✅ Dashboard
```
DASHBOARD
AI Post-Production Assistant
// Polish, Don't Generate.

MONTAGE CREATOR
Upload footage and let the AI Director assemble a cinematic montage. 
Choose from 7+ styles and customize with your creative vision.

SHORTS GENERATOR
Auto-crop 16:9 footage to 9:16 vertical. 
Smart face detection, safe zones, and platform presets 
for TikTok, Instagram, and YouTube Shorts.

SYSTEM STATUS
Storage: 45%
Queue: Idle
```

### ✅ Creator
```
STEP 1: SELECT STYLE
Choose the editing rhythm and visual language. 
Each style dictates pacing, transitions, and cut timing.

DYNAMIC
Fast cuts on the beat. High energy. Perfect for action & sports.

STEP 2: CREATIVE DIRECTION
Guide the AI with natural language. Describe mood, pacing, 
and artistic intent. The AI director will interpret these instructions.

STEP 3: PROCESSING OPTIONS
📹 VIDEO ENHANCEMENT
AI UPSCALE (4K) — Enhance resolution using Real-ESRGAN. 
Increases file size and render time.
```

### ✅ Documentation (features.md)
```markdown
## Transcript Editor

Edit video by editing text — inspired by Descript's revolutionary approach.

**How it works:**
1. Upload video → automatic transcription via Whisper
2. View transcript with word-level timestamps
3. Delete text to remove video segments

**Features:**
- Live Preview: Auto-generates 360p preview 2 seconds after edits.
- Undo/Redo: Full history stack for non-destructive editing.
- Filler removal: Highlights "um", "uh", "like" for easy removal.
```

---

## 🔍 Review Checklist

Before shipping any text:

- [ ] Is it < 60 characters wide (readable)?
- [ ] Can I remove 20% of the words and keep meaning?
- [ ] Is the voice consistent (professional, not corporate)?
- [ ] Are technical terms explained once per page?
- [ ] Are buttons action-oriented (verb-first)?
- [ ] Is capitalization consistent (CAPS for labels, Title for buttons)?
- [ ] Does it follow the "say once" rule (no repetition)?
- [ ] Is it in English (consistent spelling)?
- [ ] Are status updates clear (ONLINE, COMPLETED, etc.)?
- [ ] Does color reinforce meaning (blue for primary, orange for secondary)?

---

## 🎯 Typography Audit — What to Check

### **Consistency Across Pages**
- [ ] All H1 titles: same font size and color?
- [ ] All buttons: same padding and font?
- [ ] All labels: same case (uppercase) and spacing?
- [ ] All body text: same line height?
- [ ] All cards: same title styling?

### **Responsive Scaling**
- [ ] Headings scale on mobile (using clamp())?
- [ ] Text remains readable on all breakpoints?
- [ ] Button text doesn't overflow?
- [ ] Labels have proper letter-spacing?

### **Color Consistency**
- [ ] Primary headings: always `--primary` (#0055ff)?
- [ ] Secondary accents: always `--secondary` (#ff5500)?
- [ ] Body text: always `--fg` (#e0e0e0)?
- [ ] Muted text: always `--muted-fg` (#888888)?
- [ ] Status colors correct (green=success, yellow=warning)?

---

## 📚 Examples by Component

### **Navbar Brand**
```html
<a class="nav-brand">
  <span>MONTAGE</span> AI
</a>
<!-- ✅ MONTAGE in primary color, AI in normal text -->
```

### **Page Title**
```html
<h1 class="voxel-title">DASHBOARD</h1>
<p class="text-xs text-muted">AI Post-Production Assistant</p>
<p class="text-xs text-muted">// Polish, Don't Generate.</p>
```

### **Feature Card**
```html
<div class="voxel-card">
  <div class="workflow-icon">📽️</div>
  <h3 class="text-primary">MONTAGE CREATOR</h3>
  <p class="text-xs text-muted">
    Upload footage and let the AI Director 
    assemble a cinematic montage. Choose from 
    7+ styles and customize with your vision.
  </p>
  <button class="voxel-btn voxel-btn-primary">
    LAUNCH CREATOR →
  </button>
</div>
```

### **Status Card**
```html
<div class="voxel-card">
  <h4 class="text-primary">⚙️ SYSTEM STATUS</h4>
  <div>
    <span class="text-muted">STORAGE</span>
    <span class="text-primary">45%</span>
  </div>
  <div>
    <span class="text-muted">QUEUE</span>
    <span class="text-success">IDLE</span>
  </div>
</div>
```

---

## 🚀 Implementation Checklist

- [ ] All pages use `--fs-*` variables (no hardcoded px)
- [ ] All headings use `--lh-tight`
- [ ] All body text uses `--lh-base`
- [ ] All labels use `--ls-wide`
- [ ] All colors use CSS variables
- [ ] All text is in English
- [ ] All buttons follow verb+noun pattern
- [ ] All status badges are UPPERCASE
- [ ] All cards have consistent heading style
- [ ] All descriptions fit on 2-3 lines

---

**Next:** Implement across all templates and update documentation.

