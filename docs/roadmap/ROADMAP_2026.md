# Montage AI – Product Roadmap 2026

**Last Updated:** January 2026  
**Version:** 2.0

---

## Overview

Diese Roadmap definiert die Produktentwicklung für Montage AI über die nächsten 12 Monate. Sie basiert auf der [Produktstrategie](./STRATEGY.md) und priorisiert Features nach Marktrelevanz und technischer Machbarkeit.

---

## Roadmap Timeline

```
2026
│
├── Q1 (Jan-Mar)
│   ├── Phase 1: Foundation (Wochen 1-4)
│   │   ├── ✅ Transcript UI
│   │   ├── ✅ Quality Profiles
│   │   ├── ✅ Cloud Acceleration Toggle
│   │   └── 🔄 Onboarding & Error Handling
│   │
│   └── Phase 2a: Shorts Studio (Wochen 5-8)
│       ├── ✅ Shorts Studio UI
│       ├── 🔄 Caption Styles
│       └── 🔄 Highlight Detection MVP
│
├── Q2 (Apr-Jun)
│   ├── Phase 2b: Shorts Enhancement (Wochen 9-12)
│   │   ├── Smart Reframe Upgrade
│   │   ├── Motion Smoothing
│   │   └── Face/Subject Tracking v2
│   │
│   └── Phase 3a: Audio Polish (Wochen 13-16)
│       ├── Voice Isolation Integration
│       ├── Noise Reduction
│       └── "Clean Audio" Single Toggle
│
├── Q3 (Jul-Sep)
│   └── Phase 3b: Pro Export (Wochen 17-24)
│       ├── OTIO Export Enhancement
│       ├── Proxy Generation
│       ├── Relink Guidance
│       └── Batch Processing
│
└── Q4 (Oct-Dec)
    └── Phase 4: Scale & Polish
        ├── Creative Loop v2
        ├── Style Ecosystem
        ├── Performance Optimization
        └── Enterprise Features (API, Teams)
```

---

## Phase 1: Foundation (Januar 2026)

### Ziel
Solide Basis für outcome-basierte Workflows schaffen. Preview-first als Standard etablieren.

### Deliverables

| Feature | Status | Beschreibung |
|---------|--------|--------------|
| Transcript UI | ✅ Done | Text-basiertes Editing im Web UI |
| Quality Profiles | ✅ Done | Preview/Standard/High/Master Bundle |
| Cloud Acceleration | ✅ Done | Single Toggle für alle CGPU Features |
| Shorts Studio | ✅ Done | Phone-Frame Preview, Safe Zones |
| Onboarding Flow | 🔄 In Progress | Guided First-Run Experience |
| Error Handling | 🔄 In Progress | User-freundliche Fehlermeldungen |

### Acceptance Criteria
- [ ] Time-to-Preview <3 Minuten
- [ ] Transcript UI funktional (Upload → Transcribe → Edit → Export)
- [ ] Quality Profile wechseln ändert sichtbar Render-Settings
- [ ] Cloud Toggle aktiviert/deaktiviert alle CGPU Jobs

---

## Phase 2: Shorts Focus (Februar-März 2026)

### Ziel
Shorts Studio zum vollständigen Social-Video-Creator ausbauen.

### Phase 2a: Shorts Studio (Wochen 5-8)

| Feature | Priority | Beschreibung |
|---------|----------|--------------|
| Caption Styles | P0 | TikTok, YouTube, Karaoke Styles |
| Highlight Detection | P0 | Audio-Energy + Speech Peaks |
| Style Picker UI | P1 | Live Preview der Caption Styles |
| Safe Zone Guides | P1 | Platform-spezifische Overlays |

### Phase 2b: Shorts Enhancement (Wochen 9-12)

| Feature | Priority | Beschreibung |
|---------|----------|--------------|
| Smart Reframe v2 | P0 | Verbessertes Face/Subject Tracking |
| Motion Smoothing | P1 | Weichere Crop-Übergänge |
| Multi-Subject | P2 | Mehrere Personen tracken |
| Crop Path Editor | P2 | Manuelles Keyframe-Editing |

### Acceptance Criteria
- [ ] 4 Caption Styles verfügbar mit Live Preview
- [ ] Highlight Detection findet >80% relevanter Momente
- [ ] Smart Reframe hält Subject >90% der Zeit im Frame
- [ ] Export für TikTok/Reels/Shorts optimiert

---

## Phase 3: Audio & Pro Export (April-September 2026)

### Ziel
Audio-Qualität verbessern und Pro-Workflows mit nahtlosem NLE-Handoff ermöglichen.

### Phase 3a: Audio Polish (Wochen 13-16)

| Feature | Priority | Beschreibung |
|---------|----------|--------------|
| Clean Audio Toggle | P0 | Voice Isolation + Denoise kombiniert |
| SNR Detection | P1 | Automatische Qualitätsprüfung |
| Fallback Logic | P1 | Original nutzen wenn Isolation schadet |
| Audio Level Normalize | P2 | Konsistente Lautstärke |

### Phase 3b: Pro Export (Wochen 17-24)

| Feature | Priority | Beschreibung |
|---------|----------|--------------|
| OTIO Enhancement | P0 | Metadaten, Marker, Annotations |
| EDL v2 | P1 | Erweiterte Clip-Informationen |
| Proxy Generation | P1 | Automatische Proxies für NLE |
| Relink Documentation | P2 | Automatische README für Imports |
| Batch Export | P2 | Mehrere Outputs gleichzeitig |

### Acceptance Criteria
- [ ] Clean Audio verbessert SNR in >70% der Fälle
- [ ] OTIO importiert fehlerfrei in DaVinci/Premiere
- [ ] Proxy-Workflow dokumentiert und funktional
- [ ] Batch Export für min. 5 Clips gleichzeitig

---

## Phase 4: Scale & Polish (Oktober-Dezember 2026)

### Ziel
Performance optimieren, Community Features einführen, Enterprise-ready machen.

### Deliverables

| Feature | Priority | Beschreibung |
|---------|----------|--------------|
| Creative Loop v2 | P1 | Iteratives Verbessern mit Feedback |
| Style Ecosystem | P2 | Community Presets teilen |
| Performance | P0 | 2x Speedup auf Consumer Hardware |
| Plugin API | P2 | Erweiterbarkeit für Entwickler |
| Enterprise API | P3 | REST API, Rate Limits, Auth |
| Team Features | P3 | Shared Projects, Permissions |

### Acceptance Criteria
- [ ] Render-Zeit 50% reduziert vs. Q1
- [ ] Style Marketplace mit min. 20 Community Presets
- [ ] API dokumentiert mit OpenAPI Spec
- [ ] >99% Uptime auf Referenz-Hardware

---

## Feature Backlog (Priorisiert)

### P0 – Must Have (This Quarter)

| ID | Feature | Phase | Effort |
|----|---------|-------|--------|
| F001 | Caption Style Picker | 2a | M |
| F002 | Highlight Detection | 2a | L |
| F003 | Error Handling Overhaul | 1 | S |
| F004 | Onboarding Tutorial | 1 | M |

### P1 – Should Have (Next Quarter)

| ID | Feature | Phase | Effort |
|----|---------|-------|--------|
| F010 | Smart Reframe v2 | 2b | L |
| F011 | Clean Audio Toggle | 3a | M |
| F012 | OTIO Enhancement | 3b | M |
| F013 | Motion Smoothing | 2b | M |

### P2 – Nice to Have (This Year)

| ID | Feature | Phase | Effort |
|----|---------|-------|--------|
| F020 | Style Ecosystem | 4 | XL |
| F021 | Crop Path Editor | 2b | L |
| F022 | Batch Export | 3b | M |
| F023 | Plugin API | 4 | XL |

### P3 – Future Consideration

| ID | Feature | Phase | Effort |
|----|---------|-------|--------|
| F030 | Enterprise API | 4 | XL |
| F031 | Team Features | 4 | XL |
| F032 | Mobile App | Future | XXL |
| F033 | Browser Extension | Future | L |

---

## Dependencies & Risks

### Technical Dependencies

| Dependency | Features Affected | Mitigation |
|------------|-------------------|------------|
| Whisper Quality | Transcription, Captions | Model Selection, Fallback |
| FFmpeg Stability | All Video Processing | Version Pinning, Tests |
| CGPU Availability | Cloud Features | Graceful Fallback |
| GPU Memory | Upscale, Stabilize | Quality Profile Limits |

### External Dependencies

| Dependency | Impact | Mitigation |
|------------|--------|------------|
| OpenAI Whisper Updates | ASR Quality | Version Pinning |
| Demucs Updates | Voice Isolation | Tested Versions |
| OTIO Spec Changes | Pro Export | Compatibility Layer |
| Platform Guidelines | Shorts Formats | Configurable Presets |

### Known Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Performance Regression | Medium | High | Automated Benchmarks |
| Breaking API Changes | Low | High | Semantic Versioning |
| Scope Creep | High | Medium | Strict Phase Gates |
| Resource Constraints | Medium | Medium | MVP-first Approach |

---

## Success Metrics by Phase

### Phase 1 Metrics
- Time-to-Preview: <3 min (Target: <2 min)
- UI Error Rate: <5%
- Feature Adoption: >50% use Quality Profiles

### Phase 2 Metrics
- Shorts Created/Session: >3
- Caption Accuracy: >90%
- Reframe Success: >90%

### Phase 3 Metrics
- Audio Improvement Rate: >70%
- Export Success: >99%
- NLE Import Success: >95%

### Phase 4 Metrics
- Performance Improvement: 2x
- Community Presets: >20
- API Adoption: >10 Active Integrations

---

## Review Cadence

| Review | Frequency | Participants |
|--------|-----------|--------------|
| Sprint Review | Bi-weekly | Dev Team |
| Phase Gate | End of Phase | All Stakeholders |
| Roadmap Review | Quarterly | Leadership |
| Strategy Review | Bi-annually | All |

---

*Roadmap maintained by Montage AI Team. Subject to change based on user feedback and market conditions.*
