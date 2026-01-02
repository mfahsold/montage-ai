# Montage AI – Strategic Product Document

**Version:** 2.0  
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

### Wettbewerbslandschaft

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO EDITING LANDSCAPE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FULL NLEs                    AI-ASSISTED                        │
│  ┌─────────────┐             ┌─────────────┐                    │
│  │ Premiere    │             │ Descript    │                    │
│  │ DaVinci     │             │ CapCut Pro  │                    │
│  │ Final Cut   │             │ Runway      │                    │
│  └─────────────┘             └─────────────┘                    │
│         │                           │                            │
│         │    ┌─────────────────┐   │                            │
│         └───►│  MONTAGE AI     │◄──┘                            │
│              │ "Rough Cut Hub" │                                 │
│              │ Local + Fast    │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│              ┌────────▼────────┐                                 │
│              │  SHORTS TOOLS   │                                 │
│              │ OpusClip, Munch │                                 │
│              │ Vizard, Klap    │                                 │
│              └─────────────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Zielgruppen

### Primär: Creator & Marketing Teams
- **Profil:** Hoher Content-Output, Budget-sensitiv, brauchen Geschwindigkeit
- **Pain Points:** Zu viel Footage, zu wenig Zeit, repetitive Schnitte
- **Value Prop:** "1 Stunde Footage → 10 Social Clips in 30 Minuten"

### Sekundär: Professional Editors
- **Profil:** Nutzen Premiere/DaVinci, brauchen Rough Cuts
- **Pain Points:** Rough Cut Overhead, Selektionsaufwand
- **Value Prop:** "AI Rough Cut importieren, nur noch Fine-Tuning"

### Tertiär: Indie Filmmakers
- **Profil:** Solo oder Kleinteam, limited Resources
- **Pain Points:** Kein Budget für Editor, Zeit für alles andere
- **Value Prop:** "Professionelle Rough Cuts ohne Editing-Team"

---

## Produktstrategie

### Brand Story
> *"Editiere wie Text, schneide wie Musik, liefere wie ein Pro."*

Diese Tagline verbindet unsere drei Kernfeatures:
1. **Transcript-basiertes Editing** – Text = Video
2. **Beat-Sync Montage** – Musik = Rhythmus
3. **OTIO/EDL Export** – Pro = Handoff

### Feature-Priorisierung

#### ✅ IN SCOPE (Kernprodukt)

| Feature | Status | Priorität |
|---------|--------|-----------|
| Text-based Editing UI | MVP Ready | P0 |
| Shorts Studio | MVP Ready | P0 |
| Beat-Sync Montage | Production | P0 |
| Smart Clip Selection | Production | P1 |
| Auto-Captions | Production | P1 |
| Voice Isolation | Integrated | P1 |
| OTIO/EDL Export | Production | P1 |
| Preview-first Pipeline | Production | P0 |
| Quality Profiles | Implemented | P1 |

#### ❌ OUT OF SCOPE (Bewusste Abgrenzung)

| Feature | Grund | Alternative |
|---------|-------|-------------|
| Full Timeline Editing | Pro NLEs bedienen | OTIO Export |
| Generative Video/Avatare | "Polish, don't generate" | - |
| Social Publishing | Partner-Integrationen | Export-Formate |
| Cloud-only Rendering | Privacy-first | Optional Offload |
| Multicam Editing | NLE Feature | Single-Track Focus |
| Compositing/VFX | Spezialisierte Tools | - |

### Feature Consolidation

Bestehende Features werden in **Outcome-basierte Bundles** zusammengefasst:

```
VORHER (Toggle-Friedhof):          NACHHER (Outcome-Flows):
┌─────────────────────────┐        ┌─────────────────────────┐
│ □ Enhance               │        │ Quality Profile         │
│ □ Stabilize             │   ──►  │ ○ Preview (360p, fast)  │
│ □ Upscale               │        │ ○ Standard (1080p)      │
│ □ CGPU                  │        │ ○ High (+ Stabilize)    │
│ □ LLM Selection         │        │ ○ Master (4K + All)     │
│ □ Creative Loop         │        └─────────────────────────┘
│ □ Story Engine          │        
│ □ Captions              │        ┌─────────────────────────┐
│ □ Export Timeline       │        │ Cloud Acceleration      │
│ □ Generate Proxies      │   ──►  │ [====== ON ======]      │
│ □ Preserve Aspect       │        │ Auto-fallback to local  │
└─────────────────────────┘        └─────────────────────────┘
```

---

## UI/UX Strategie

### Design Principles

1. **Outcome-First:** Nutzer wählen Ziel, nicht Toggles
2. **Preview-Default:** Jeder Workflow startet mit schneller Vorschau
3. **Progressive Disclosure:** Erweiterte Optionen versteckt
4. **Decision Transparency:** AI-Entscheidungen sichtbar machen

### Neue UI-Struktur

```
┌─────────────────────────────────────────────────────────────────┐
│  MONTAGE AI                           [Create] [Text] [Shorts]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   MONTAGE       │  │   TRANSCRIPT    │  │   SHORTS        │ │
│  │   Creator       │  │   Editor        │  │   Studio        │ │
│  │                 │  │                 │  │                 │ │
│  │   Beat-sync     │  │   Text = Cut    │  │   9:16 + Auto   │ │
│  │   Story Arc     │  │   Click words   │  │   Reframe       │ │
│  │   Style Presets │  │   Remove filler │  │   Captions      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Quality: [Preview ○] [Standard ●] [High ○] [Master ○]       ││
│  │ Cloud:   [═══════════ ON ═══════════]                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [════════════════ CREATE MONTAGE ════════════════]             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Visual System

| Element | Spezifikation |
|---------|---------------|
| Typography | Space Grotesk (Headlines), Inter (Body) |
| Colors | Deep Purple (#7C3AED) Accent, Dark Mode Default |
| Motion | Kinetic Entrance, Subtle Hover States |
| Texture | Subtle Gradients, Noise Overlays |

---

## Technische Architektur

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         MONTAGE AI                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Web UI    │    │    CLI      │    │   Python    │          │
│  │  (Flask)    │    │  (Typer)    │    │   API       │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│         │                  │                  │                   │
│         └──────────────────┼──────────────────┘                   │
│                            │                                      │
│  ┌─────────────────────────▼─────────────────────────┐           │
│  │              CORE ENGINE                           │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │           │
│  │  │ Editor  │ │Montage  │ │ Scene   │ │ Audio   │ │           │
│  │  │  .py    │ │Builder  │ │Analysis │ │Analysis │ │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │           │
│  └───────────────────────┬───────────────────────────┘           │
│                          │                                        │
│  ┌───────────────────────▼───────────────────────────┐           │
│  │              PROCESSING LAYER                      │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │           │
│  │  │ FFmpeg  │ │ Whisper │ │ OpenCV  │ │ Librosa │ │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │           │
│  └───────────────────────┬───────────────────────────┘           │
│                          │                                        │
│  ┌───────────────────────▼───────────────────────────┐           │
│  │           OPTIONAL: CLOUD ACCELERATION             │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │           │
│  │  │ CGPU    │ │ Voice   │ │Upscale  │ │ Render  │ │           │
│  │  │ Utils   │ │Isolation│ │ Job     │ │ Job     │ │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │           │
│  └───────────────────────────────────────────────────┘           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### OSS Stack

| Component | Library | License | Purpose |
|-----------|---------|---------|---------|
| Video Processing | FFmpeg | LGPL/GPL | Encoding, Decoding, Filters |
| Audio Analysis | Librosa | ISC | Beat Detection, Energy |
| Computer Vision | OpenCV | Apache 2.0 | Scene Detection, Face Detection |
| Transcription | Whisper | MIT | Speech-to-Text |
| Voice Isolation | Demucs | MIT | Source Separation |
| Upscaling | Real-ESRGAN | BSD | AI Upscaling |
| Timeline Export | OpenTimelineIO | Apache 2.0 | NLE Interchange |
| Web Framework | Flask | BSD | Web UI |

---

## Roadmap 12 Monate

### Phase 1: Foundation (0-4 Wochen)
- [x] Transcript-UI im Web
- [x] Preview-first Pipeline
- [x] Toggle-Konsolidierung (Quality Profiles)
- [x] Cloud Acceleration Single-Toggle
- [ ] Onboarding Flow verbessern
- [ ] Error Handling & User Feedback

### Phase 2: Shorts Focus (5-12 Wochen)
- [x] Shorts Studio UI
- [ ] Smart Reframe Upgrade (Motion Smoothing)
- [ ] Caption Styles (TikTok/YouTube/Karaoke)
- [ ] Highlight Detection MVP
- [ ] Face/Subject Tracking verbessern

### Phase 3: Audio & Pro (3-6 Monate)
- [ ] Audio Polish ("Clean Audio" Toggle)
- [ ] Voice Isolation + Denoise Bundle
- [ ] Pro Export Pack (OTIO + Proxies + Relink)
- [ ] Batch Processing für Clips
- [ ] Decision Logs Export

### Phase 4: Scale (6-12 Monate)
- [ ] Creative Loop v2 (iteratives Verbessern)
- [ ] Style Ecosystem (Community Presets)
- [ ] Performance Optimierung
- [ ] Plugin Architecture
- [ ] Enterprise Features (Team, API)

---

## KPIs & Metriken

### Performance KPIs

| Metrik | Ziel | Messung |
|--------|------|---------|
| Time-to-Preview | <3 Min bei 10 Min Input | Automatisch |
| Time-to-Export | <10 Min bei 10 Min Input | Automatisch |
| Preview Acceptance | >70% ohne Änderung | User Action |

### Quality KPIs

| Metrik | Ziel | Messung |
|--------|------|---------|
| Caption WER | <10% | Stichprobe |
| Reframe Accuracy | >90% Subjects in Frame | Stichprobe |
| Export Success | >99% | Automatisch |
| NLE Relink Success | >95% | User Feedback |

### Adoption KPIs

| Metrik | Ziel | Messung |
|--------|------|---------|
| Shorts Output/Session | >3 | Automatisch |
| Captions Adoption | >60% | Feature Usage |
| Cloud Acceleration Usage | >40% (wenn verfügbar) | Feature Usage |
| Return Users (7-day) | >30% | Analytics |

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Reframe-Fehler sichtbar | Mittel | Hoch | Live-Preview, Manual Override |
| Caption-Qualität variiert | Hoch | Mittel | Edit-Option, WER-Monitoring |
| Performance auf Consumer HW | Hoch | Hoch | Preview-first, Proxy-System |
| UI-Komplexität steigt | Mittel | Mittel | Outcome-Flows, Defaults |
| Wettbewerber kopieren Features | Hoch | Mittel | Speed + Privacy Differenzierung |
| Datenschutz-Bedenken | Niedrig | Hoch | Local-first, Transparency Panel |

---

## Appendix

### A. Competitive Feature Matrix

| Feature | Montage AI | Descript | OpusClip | Premiere |
|---------|-----------|----------|----------|----------|
| Local Processing | ✅ | ❌ | ❌ | ✅ |
| Text-based Editing | ✅ | ✅ | ❌ | ✅ |
| Beat Sync | ✅ | ❌ | ❌ | ❌ |
| Smart Reframe | ✅ | ✅ | ✅ | ✅ |
| Auto Shorts | ✅ | ❌ | ✅ | ❌ |
| OTIO Export | ✅ | ❌ | ❌ | ✅ |
| Voice Isolation | ✅ | ✅ | ❌ | ❌ |
| Free Tier | ✅ | Limited | Limited | ❌ |

### B. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Local-first | Yes | Privacy, Speed, Offline |
| Python Backend | Yes | ML Ecosystem, Rapid Dev |
| FFmpeg Core | Yes | Industry Standard, Robust |
| Whisper for ASR | Yes | Quality, Open Source |
| Flask Web UI | Yes | Simple, Sufficient |

### C. Glossary

- **Rough Cut:** Erste Schnittversion ohne Fine-Tuning
- **OTIO:** OpenTimelineIO – Interchange Format für NLEs
- **EDL:** Edit Decision List – Legacy aber weit verbreitet
- **Smart Reframe:** AI-basiertes Cropping für vertikale Formate
- **Beat Sync:** Schnitte auf Musik-Beats ausrichten
- **Story Arc:** Dramaturgische Spannungskurve

---

*Document maintained by Montage AI Team. Last updated: January 2026.*
