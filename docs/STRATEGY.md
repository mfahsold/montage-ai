# Montage AI – Strategic Product Document

**Version:** 2.3 (PwC-Style Update)
**Date:** January 2026
**Classification:** Public

---

## Executive Summary

**Positionierung:** *"AI rough cut + social-ready output + pro handoff"*

Montage AI ist ein **local-first, privacy-first** Post-Production-Assistent, der konsequent dem Prinzip **"polish, don't generate"** folgt. Wir generieren keine Pixel – wir veredeln bestehendes Footage.

### Kernvorteil
Der Kernvorteil bleibt **"polish, don't generate"**: local-first + pro-handoff ist klar differenziert; jetzt zählt erlebbare Geschwindigkeit und Kontrolle.

### Priorität Q1
- **Transcript Editor produktisieren**
- **Shorts Studio 2.0 liefern**
- **Pro-Handoff + Audio-Polish stabilisieren**
- Alles auf **"Preview-first"** getrimmt.

### Scope
Strikt halten (Rough-Cut + Social-Ready + NLE-Export); generative Video-Erzeugung und Voll-NLE bewusst ausklammern.

### UI Vision
Vom "Toggle-Friedhof" zum **"Outcome-Studio"**: klare Workflows, sichtbare Story/Beat-Logik, stilprägende Motion.

---

## Implementation Snapshot (Codebase Reality)

| Feature | Status | Details |
|---------|--------|---------|
| **Preview Pipeline** | ✅ Ready | `ffmpeg_config.py` hat Preview-Preset (360p, ultrafast) + GPU-Auto-Encoding. Perfekte Basis für echte Preview-First-UX. |
| **Text-based Editing** | ⚠️ Partial | `transcript.html` liefert Word-Level-Edits, Filler-Removal und Export-Buttons. Preview-Flow ist aktuell Stub. |
| **Shorts/Reframing** | ⚠️ Partial | `shorts.html` liefert Safe-Zones, Caption-Style-Preview, Reframe-Modi, Highlight-UI. Highlights/Render sind teilweise Stub-Logik. |
| **Pro-Export** | ✅ Core | `video_metadata.py` bestimmt Output-Profile via gewichtete Mediane; passt ideal zu Quality-Profile-Automatik. |
| **Audio-Polish** | ⚠️ Stub | Toggle vorhanden, aber Implementierung (Voice Isolation) muss stabilisiert werden. |

*Doku/Status wirkt inkonsistent (z.B. STRATEGY.md vs. vorhandene Transcript-UI); das erzeugt Erwartungslücken.*

---

## Market Signals & Benchmarks

| Signal | Quelle | Implikation |
|--------|--------|-------------|
| **Video ist Mainstream** | [Wyzowl](https://www.wyzowl.com/video-marketing-statistics/) | Qualität beeinflusst Vertrauen → Quality-Profiles + Audio-Polish sind Pflicht. |
| **Text-basiertes Editing** | [Descript](https://www.descript.com/), [Adobe](https://helpx.adobe.com/premiere/desktop/edit-projects/edit-video-using-text-based-editing/transcribe-video.html) | Standard bei Marktführern → Transcript-UI muss first-class sein. |
| **Short-Form ist Kernformat** | [YouTube Shorts](https://support.google.com/youtube/answer/10059070?hl=en) | Vertical/Shorts-Workflow ist strategisch. |
| **AI-Reframe** | [Opus Clip](https://www.opus.pro/ai-reframe) | Klarer Wettbewerbsvorteil bei Repurposing-Tools → Smart-Reframe-Qualität entscheidet. |
| **Pro-NLEs & AI-Audio** | [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve) | Erwartung für "Clean Audio" (Voice Isolation) ist gesetzt. |
| **OTIO Standard** | [OpenTimelineIO](https://opentimelineio.readthedocs.io/en/stable/) | Pro-Export bleibt zentraler Differenziator. |

---

## Fokus Features (Q1 Priorities)

### 1. Transcript Editor Produktisierung
- **Live-Preview (360p):** Sofortiges Feedback bei Text-Löschung.
- **Word-Level-Cut-List:** Apply/Undo Stack.
- **Filler-Removal:** Automatische Erkennung und Entfernung von "äh", "um".
- **Pro-Export:** OTIO/EDL direkt aus dem Textschnitt; Transkription bleibt Whisper-kompatibel.

### 2. Shorts Studio 2.0
- **Smart-Reframe v2:** Subject Tracking + Motion Smoothing.
- **Caption-Styles:** Echte Styles (TikTok/Bold/Karaoke) mit Live-Preview.
- **Highlight-Detection:** MVP + Review-Cards.

### 3. Preview-First Pipeline
- **Default-Preview:** Sofort nach Upload, klarer ETA/Progress.
- **"Final Render":** Als separater Step.
- **Upscale:** Nur in High/Master via Real-ESRGAN.

### 4. Pro Handoff Pack
- **OTIO-Export:** Standard für DaVinci/Premiere.
- **Proxies:** Automatische Generierung.
- **Relink-README:** Anleitung für den Import.
- **Import-Smoke-Tests:** Mit Resolve/Premiere.

### 5. Audio-Polish
- **Clean Audio Toggle:** Voice Isolation + Denoise + Fallback bei Artefakten.
- **SNR-Check:** Erwartung deckt Pro-NLE-Standard.

---

## Scope

**In-Scope:**
- Rough-Cut aus vorhandenen Clips
- Transcript-Editing
- Shorts-Outputs
- Captions
- Smart-Reframe
- Audio-Polish
- OTIO/EDL-Handoff
- Local-first mit optionalem Cloud-Boost

**Out-of-Scope:**
- Generative Video
- Vollständiger NLE-Ersatz (Multitrack-Compositing, VFX)
- Social Hosting/Distribution
- After-Effects-artige Motion-Graphics

---

## Weglassen/Konsolidieren

- **LLM-Toggles:** Unter ein "AI Director"-Flag bündeln.
- **Creative Loop:** Als "Advanced-Drawer" verstecken.
- **UI-Varianten:** Reduzieren auf einen Outcome-Hub + drei Workflows (Legacy/v2/Strategy deprecaten).
- **Style-Preset-Katalog:** Straffen auf kuratierte Kern-Styles.
- **Silence-Removal:** Als Basisfeature behandeln (Utility-Baseline).
- **Cloud-Optionen:** Bündeln; Cloud-Toggle entscheidet, nicht einzelne Flags.

---

## UI/UX Vision: "Outcome Studio"

Damit es "hip & innovativ" wirkt:

1.  **Transcript-First Tri-Pane:** Video + Text + Beat/Story-Timeline mit Live-Marker (zeigt, warum ein Cut passiert).
2.  **Kinetische Beat-Timeline:** Energy-Curve, Beat-Ticks und Story-Arc-Phasen als Overlay (macht Differenzierung greifbar).
3.  **Shorts-Studio als "Phone-Rig":** Crop-Path-Overlay mit Keyframe-Handles, Safe-Zone-Presets pro Plattform, Caption-Composer mit Live-Styles.
4.  **"Preview vs Final":** Bewusstes Ritual, klarer Zustand, Vergleichssplit, schnelle A/B-Loops.
5.  **Typo & Motion:** Starke, eigenständige Headlines, subtile Motion-Reveals, UI-Sounding (Click-to-Cut).

---

## 90-Tage-Plan (Komprimiert)

| Wochen | Fokus | Deliverable |
|--------|-------|-------------|
| **0-4** | **Foundation** | Transcript-Editor Preview-Flow + Export stabilisieren, Doku-Sync, Telemetrie für Time-to-Preview. |
| **5-8** | **Growth** | Shorts Studio 2.0 (Reframe v2 + Caption-Styles + Highlight-MVP), UI-Polish. |
| **9-12** | **Pro** | Pro-Handoff Pack + Audio-Polish + Performance-Targets. |

## Kern-KPIs

1.  **Time-to-First-Preview:** < 2-3 Min, Preview-Success-Rate > 95%.
2.  **Transcript-Editing-Adoption:** > 40% der Sessions; Export-Success > 95%.
3.  **Shorts-Creation-Cycle:** < 10 Min; Reframe-Accuracy > 90%.
4.  **Audio-Improvement-Rate:** > 70% (SNR-Check).

---

## Risiken & Mitigation

- **Performance/Hardware-Streuung:** Preview-First + Proxy-Pfad + GPU-Fallback.
- **LLM-Unzuverlässigkeit:** Guardrails + deterministic defaults.
- **UI-Komplexität:** Outcome-Flows + progressive disclosure.
- **Cloud-Verfügbarkeit:** Harte Fallback-Strategie + klare UI-Kommunikation.

---

## Quellen

[1] [Wyzowl Video Marketing Statistics](https://www.wyzowl.com/video-marketing-statistics/)
[2] [Descript Video Editing](https://www.descript.com/video-editing)
[3] [Adobe Text-Based Editing](https://helpx.adobe.com/premiere/desktop/edit-projects/edit-video-using-text-based-editing/transcribe-video.html)
[4] [Opus Clip AI Reframe](https://www.opus.pro/ai-reframe)
[5] [YouTube Shorts Guide](https://support.google.com/youtube/answer/10059070?hl=en)
[6] [YouTube Captions](https://support.google.com/youtube/answer/2734796?hl=en)
[7] [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve)
[8] [OpenTimelineIO](https://opentimelineio.readthedocs.io/en/stable/)
[9] [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
[10] [Auto-Editor](https://github.com/WyattBlue/auto-editor)
[11] [OpenAI Whisper](https://github.com/openai/whisper)


Weg vom "Toggle-Friedhof" hin zu klaren Workflows:

1.  **Transcript-First Tri-Pane:** Video + Text + Beat/Story-Timeline.
2.  **Kinetische Beat-Timeline:** Energy-Curve und Story-Arc als Overlay.
3.  **Shorts-Studio als "Phone-Rig":** Crop-Path-Overlay, Safe-Zones, Caption-Composer.
4.  **Preview vs Final:** Bewusster Schritt, schneller A/B-Vergleich.
5.  **Typo & Motion:** Cyber-NLE-Look, aber hochwertig (Click-to-Cut Sounding).

---

## Roadmap 12 Monate

### Phase 1: Foundation (0-4 Wochen)
- Transcript-UI
- Preview-first Pipeline
- Toggle-Konsolidierung
- Besseres Onboarding

### Phase 2: Growth (5-12 Wochen)
- Shorts Studio
- Smart-Reframe-Upgrade
- Captions-Styles
- Highlight-Detection MVP

### Phase 3: Pro (3-6 Monate)
- Audio-Polish (Voice Isolation + Denoise)
- Pro-Export-Pack (OTIO+Proxies+Relink)

### Phase 4: Ecosystem (6-12 Monate)
- Creative Loop v2
- Style-Ecosystem/Marketplace
- Performance-Optimierung und Reliability

---

## UI/UX Vision

- **Transcript-first Layout:** Split-View (Video + Text), Wort-Highlights, One-click Cuts.
- **Energy/Beat Timeline:** Sichtbare Beat-Marker + Story-Arc-Kurve.
- **Shorts Studio:** Phone-Frame, Safe-Area-Guides, Crop-Path-Overlay.
- **Style-Moodboard:** Preset-Cards mit Farbfeld, Rhythmus-Label, Sample-Frames.
- **Visual System:** Mutige Typo, kinetische Einstiegsmotion, subtile Texturen.

---

## Risiken & Mitigation

- **Reframe/Caption-Fehler:** Live-Preview, manuelle Override-Griffe.
- **Performance:** Preview-first, Proxy-Generierung, GPU-Auto-Detect.
- **UI-Komplexitaet:** Outcome-Flows, Progressive Disclosure.
- **Wettbewerb:** Fokus auf zuverlaessigen Rough-cut + Pro-Handoff.
