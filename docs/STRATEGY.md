# Montage AI – Strategic Product Document

**Version:** 2.1
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

## Implementation Snapshot

| Feature | Status | Details |
|---------|--------|---------|
| **Text-based Editing** | ⚠️ Backend Ready | `text_editor.py` implementiert, CLI ready. **Web UI fehlt.** |
| **Shorts/Reframing** | ✅ MVP | `smart_reframing.py` integriert in `montage_builder.py`. |
| **Captions** | ✅ Core | Transcription + Burn-in in `editor.py`, UI Toggle vorhanden. |
| **Voice Isolation** | ✅ CGPU Job | `voice_isolation.py` vorhanden, integriert in Builder. |
| **Pro-Export** | 📋 Planned | OTIO dokumentiert, Roadmap-Scope verankert. |

---

## Fokus Features (Next Steps)

### 1. Text-based Editing UI
- Transcript-Panel im Web UI
- Klick-to-Cut
- Filler-Removal
- Segment-Tags
- Preview + OTIO/EDL Export direkt aus Transcript-Schnitt

### 2. Shorts-Pipeline 2.0
- Face/Subject-Tracking
- Segmentierte Crops
- Motion-Smoothing
- Caption-Styles (TikTok/YouTube/Karaoke) mit Live-Preview

### 3. Highlight Detection MVP
- Hooks/Peaks aus Audio-Energy + Speech-Phrases
- Score-basierte Vorschlaege mit manueller Bestaetigung

### 4. Audio-Polish
- Voice-Isolation + Noise-Reduction als ein "Clean Audio"-Schalter
- SNR-Check, Fallback auf Original

### 5. Preview-first & Quality-Profile
- Automatisch 360p/30s Preview
- 1-Click Final Render
- Profile buendeln Enhance/Stabilize/Upscale

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
