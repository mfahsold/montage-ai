# Quick Style Reference — Montage AI

> **Keep This Handy:** Use when writing UI copy, documentation, or marketing text.

---

## 🎯 Brand Voice

| Aspect | Rule |
| ------ | ------ |
| **Philosophy** | "Polish, don't generate." |
| **Tone** | Technical, concise, professional |
| **Tempo** | Fast-paced, action-oriented |
| **Audience** | Creators, video editors, developers |

---

## 📝 Capitalization Quick Rules

```text
Montage Creator     ← Title Case (feature name)

STEP 1: Select      ← STEP + Title Case
Generate Short      ← Title Case (button)
ENABLED             ← UPPERCASE (status)
beat-sync montages  ← lowercase (description)
```

---

## ✨ Condensation Checklist

- [ ] Remove "using X technology" → Keep just the tech name
- [ ] Combine 2-3 short sentences into 1
- [ ] Remove adjectives (amazing, powerful, revolutionary)
- [ ] Keep verbs active and imperative
- [ ] Preserve technical accuracy

**Before:** "Choose the editing rhythm and visual language for your project. You can select from multiple preset styles like Dynamic, MTV, and Minimal."

**After:** "Choose editing rhythm and visual language. Pick from dynamic, MTV, or minimal."

---

## 🔤 Typography System

### Font Stack

```css
Headings:  "Share Tech Mono", monospace
Body:      System font stack (Segoe UI, Roboto, sans-serif)
```

### Font Size (Responsive)

```text
H1: clamp(2.5rem, 7vw, 4rem)        /* 40–64px */
H2: clamp(2rem, 5vw, 3rem)          /* 32–48px */
H3: clamp(1.5rem, 3.5vw, 2rem)      /* 24–32px */
Body: 1rem                           /* 16px */
Small: 0.75rem                       /* 12px */
```

### Colors

```text
Primary Text:    #e0e0e0 (var(--fg))
Secondary:       #a0a0a0 (var(--muted-fg))
Accent Primary:  #0055ff (electric blue)
Accent Secondary: #ff5500 (neon orange)
Success:         #00ff88 (green)
Error:           #ff3333 (red)
```

---

## 🎨 Component Patterns

### Button Text

```text
✅ Launch Creator
✅ Generate Short
✅ Apply Edits & Render
✅ Create Montage

❌ LAUNCH CREATOR →
❌ Click Here
❌ Submit Form
```

### Step Labels

```text
✅ STEP 1: Select Style
✅ STEP 2: AI Settings
✅ STEP 3: Safe Zones (Optional)
✅ STEP 4: Generate Short

❌ STEP 1 - SELECT STYLE
❌ Step 1
```

### Descriptions (Max 1-2 Sentences)

```text
✅ "Edit video by removing text. AI handles the cuts."
✅ "Auto-reframe to 9:16. Face detection, safe zones, platform presets."

❌ "Our revolutionary AI-powered video editor lets you edit your videos in a 
   revolutionary way by removing text from your transcript and our AI will 
   intelligently handle all the cutting for you automatically."
```

### Feature List

```text
✅ • Beat-sync montages
✅ • Transcript editing  
✅ • OTIO/EDL export

❌ • Montages that are synced to the beat of your music using advanced 
    librosa beat detection
```

---

## 📊 Terminology Standards

| Tech | Usage | NOT |
| ------ | ------- | ----- |
| OTIO | "OTIO/EDL export to DaVinci" | "OpenTimelineIO export" |
| FFmpeg | "FFmpeg rendering" | "FFmpeg (the video encoder)" |
| MediaPipe | "MediaPipe face detection" | "the MediaPipe library by Google" |
| librosa | "librosa beat detection" | "librosa (audio analysis library)" |
| Whisper | "Whisper transcription" | "OpenAI Whisper STT" |

**Rule:** Use term + brief context, not full name on first mention in UI.

---

## 🚨 Common Mistakes to Avoid

| ❌ NO | ✅ YES |
| ------- | -------- |
| "Using Real-ESRGAN..." | "Real-ESRGAN enhancement" |
| "Which will allow you to..." | "Enables you to..." |
| "Click to configure" | "Configure" (use button label) |
| "The system will..." | "AI handles..." (active) |
| "Our cutting-edge AI" | "AI detects..." (no fluff) |
| "HELLO WORLD" | "Hello World" (except status) |

---

## 📋 Update Checklist

When adding new UI text:

- [ ] Is it in English only? (no German)
- [ ] Is it Title Case (headings) or lowercase (body)?
- [ ] Does it fit in 1-2 sentences?
- [ ] Does it use active voice?
- [ ] Are buttons Title Case with no arrows?
- [ ] Is technical terminology used correctly?
- [ ] Does it avoid marketing fluff?
- [ ] Can you remove 20% without losing meaning?

---

## 🎯 Messaging by Context

### For Dashboard

- Goal: Status overview, quick actions
- Tone: Informational, concise
- Length: Single sentence max
- Example: "System online. GPU ready. 3 jobs pending."

### For Workflow Pages (Creator, Shorts, Transcript)

- Goal: Guide user through steps
- Tone: Instructional, clear
- Length: 1-2 sentences per step
- Example: "Choose editing rhythm and visual language."

### For Documentation

- Goal: Explain capability
- Tone: Technical, thorough
- Length: 2-3 paragraphs max
- Example: "Beat-sync aligns cuts to music rhythm using librosa analysis."

### For Error Messages

- Goal: Help user fix problem
- Tone: Helpful, specific
- Length: 1 sentence + 1 suggestion
- Example: "Invalid clip format. Use MP4, MOV, or MKV."

---

## 🔗 References

- Full guide: [TYPOGRAPHY_WRITING_GUIDE.md](TYPOGRAPHY_WRITING_GUIDE.md)
- Audit log: [TYPOGRAPHY_AUDIT_COMPLETE.md](TYPOGRAPHY_AUDIT_COMPLETE.md)
- Phase 2 summary: [PHASE_2_COMPLETE.md](PHASE_2_COMPLETE.md)

---

**Last Updated:** 2026-01  
**Maintained By:** Montage AI Development Team  
**Version:** 1.0 (Post-Phase 2 Audit)
