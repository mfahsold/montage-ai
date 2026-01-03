# Montage AI – Strategic Product Document

**Version:** 2.2
**Date:** January 2026
**Classification:** Public

---

## Executive Summary

**Positionierung:** *"AI rough cut + social-ready output + pro handoff"*

Montage AI ist ein **local-first, privacy-first** Post-Production-Assistent, der konsequent dem Prinzip **"polish, don't generate"** folgt. Wir generieren keine Pixel – wir veredeln bestehendes Footage.

### Kernversprechen
- **Schneller AI Rough Cut** – Beat-Sync, Story-Arc, Smart Selection
- **Social-Ready Output** – Shorts, Captions, Smart Reframe
- **Pro Handoff** – OTIO/EDL Export für nahtlose NLE-Integration

### Strategische Differenzierung
| Aspekt | Montage AI | Wettbewerber |
|--------|-----------|--------------|
| Verarbeitung | Local-first, Cloud optional | Cloud-only |
| Datenschutz | Keine Upload-Pflicht | Daten auf fremden Servern |
| Fokus | Polish & Edit | Generative Features |
| Export | Pro-grade (OTIO/EDL) | Proprietär |
| Speed | Preview in <3 Min | Längere Wartezeiten |

---

## Market Signals & Wettbewerbsanalyse

### Validierte Marktbedürfnisse

| Signal | Quelle | Implikation |
|--------|--------|-------------|
| Text-based Editing ist Must-Have | [Descript](https://www.descript.com/), [Adobe Premiere Pro](https://helpx.adobe.com/premiere/desktop/edit-projects/edit-video-using-text-based-editing/transcribe-video.html) | Transcript-UI als Kernfeature |
| Short-form Clipping ist Kern-Workflow | [OpusClip](https://www.opus.pro/) | Shorts-Pipeline priorisieren |
| AI Reframe + Animated Captions | [OpusClip AI Reframe](https://www.opus.pro/ai-reframe) | Smart Reframing + Caption Styles |
| Auto-Captions sind Plattform-Erwartung | [YouTube Help](https://support.google.com/youtube/answer/2734796?hl=en) | Burn-in Captions als Standard |
| Pro-Workflows brauchen NLE-Handoff | [OpenTimelineIO](https://opentimelineio.readthedocs.io/en/stable/) | OTIO/EDL Export ausbauen |
| Audio-Polish wird AI-Feature | [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve) | Voice Isolation integrieren |
| Short-Form Reichweite | [TikTok 1B+ MAU](https://newsroom.tiktok.com/en-us/1-billion-people-on-tiktok) | Shorts als Wachstumsdriver |

---

## Strategic Positioning

**Claim:** "Montage AI = schneller AI rough cut + social-ready output + pro handoff", klar abgegrenzt von Full-NLEs.

- **Differenzierung:** Local-first/privacy + Geschwindigkeit bis zum ersten Cut; Cloud nur als Boost, nicht Pflicht.
- **Zielsegmente:** Creator/Marketing-Teams mit hohem Clip-Volumen; Pro-Editoren, die Rough Cuts importieren.
- **Produktstory:** "Editiere wie Text, schneide wie Musik, liefere wie ein Pro" verbindet Transcript + Beat-Sync + OTIO.
- **Go-to-market:** Fokus auf Output-Qualitaet (Captions, Reframe, Audio), nicht auf generative Gimmicks.

---

## Implementation Snapshot (Codebase Reality)

| Feature | Status | Details |
|---------|--------|---------|
| **Preview Pipeline** | ✅ Ready | `ffmpeg_config.py` hat Preview-Preset (360p, ultrafast). Basis für Preview-First-UX. |
| **Text-based Editing** | ⚠️ Partial | `transcript.html` liefert Word-Level-Edits & Export. Preview-Flow ist Stub. |
| **Shorts/Reframing** | ⚠️ Partial | `shorts.html` hat UI für Safe-Zones & Styles. Render-Logik teilweise Stub. |
| **Pro-Export** | ✅ Core | `timeline_exporter.py` unterstützt OTIO/EDL. Integration in Transcript Editor fertig. |
| **Audio-Polish** | ⚠️ Stub | Toggle vorhanden, aber Implementierung (Voice Isolation) muss stabilisiert werden. |

---

## Fokus Features (Q1 Priorities)

### 1. Transcript Editor Produktisierung
- **Live-Preview (360p):** Sofortiges Feedback bei Text-Löschung.
- **Word-Level-Cut-List:** Apply/Undo Stack.
- **Filler-Removal:** Automatische Erkennung und Entfernung von "äh", "um".
- **Pro-Export:** OTIO/EDL direkt aus dem Textschnitt (Done).

### 2. Shorts Studio 2.0
- **Smart-Reframe v2:** Subject Tracking + Motion Smoothing (via `scipy` Optimierung).
- **Caption-Styles:** Echte Styles (TikTok/Bold/Karaoke) mit Live-Preview.
- **Highlight-Detection:** MVP für automatische Clip-Vorschläge.

### 3. Preview-First Pipeline
- **Default-Preview:** Sofort nach Upload generieren (360p).
- **Klarer ETA:** Progress-Bar für Preview vs. Final Render.
- **Upscale:** Nur in High/Master Profilen (via Real-ESRGAN).

### 4. Pro Handoff Pack
- **OTIO-Export:** Standard für DaVinci/Premiere.
- **Proxies:** Automatische Generierung für smooth Editing.
- **Relink-README:** Anleitung für den Import im NLE.

### 5. Audio-Polish
- **Clean Audio Toggle:** Voice Isolation + Denoise.
- **SNR-Check:** Fallback auf Original bei Artefakten.

---

## UI/UX Vision: "Outcome Studio"

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
