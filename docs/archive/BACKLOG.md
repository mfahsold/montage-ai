# Montage AI – Product Backlog

**Last Updated:** January 2026  
**Format:** Epic → User Stories → Tasks

---

## Epics Overview

| Epic | Priority | Phase | Status |
|------|----------|-------|--------|
| E01: Transcript-Based Editing | P0 | 1 | ✅ Done |
| E02: Quality Profiles | P0 | 1 | ✅ Done |
| E03: Cloud Acceleration | P0 | 1 | ✅ Done |
| E04: Shorts Studio | P0 | 2 | ✅ MVP Done |
| E05: Caption Styles | P0 | 2 | 🔄 In Progress |
| E06: Highlight Detection | P0 | 2 | 🔄 In Progress |
| E07: Audio Polish | P1 | 3 | 📋 Planned |
| E08: Pro Export Pack | P1 | 3 | 📋 Planned |
| E09: Creative Loop v2 | P2 | 4 | 📋 Backlog |
| E10: Style Ecosystem | P2 | 4 | 📋 Backlog |

---

## E01: Transcript-Based Editing ✅

### Description
Ermögliche Video-Editing durch Text-Manipulation. Nutzer können Wörter löschen/markieren und das Video wird entsprechend geschnitten.

### User Stories

#### US-01.1: Video für Transkription hochladen ✅
**Als** Content Creator  
**möchte ich** ein Video hochladen  
**damit** ich es per Text editieren kann

**Acceptance Criteria:**
- [x] Upload akzeptiert MP4, MOV, WebM, MKV
- [x] Maximale Dateigröße: 500MB
- [x] Progress-Anzeige während Upload
- [x] Fehlermeldung bei ungültigem Format

#### US-01.2: Video transkribieren ✅
**Als** Content Creator  
**möchte ich** mein Video automatisch transkribieren  
**damit** ich den gesprochenen Text sehe

**Acceptance Criteria:**
- [x] Whisper-basierte Transkription
- [x] Word-Level Timestamps
- [x] Progress-Anzeige
- [x] Unterstützt EN/DE/ES/FR

#### US-01.3: Text editieren = Video editieren ✅
**Als** Content Creator  
**möchte ich** Wörter anklicken um sie zu entfernen  
**damit** das Video automatisch geschnitten wird

**Acceptance Criteria:**
- [x] Klick auf Wort markiert als "removed"
- [x] Visuelle Markierung (durchgestrichen, rot)
- [x] Undo-Funktion
- [x] Stats zeigen entfernte Zeit

#### US-01.4: Filler-Words automatisch entfernen ✅
**Als** Content Creator  
**möchte ich** Füllwörter mit einem Klick entfernen  
**damit** mein Video professioneller klingt

**Acceptance Criteria:**
- [x] "Remove Fillers" Button
- [x] Erkennt: um, uh, er, like, you know, basically, actually
- [x] Bulk-Undo möglich

#### US-01.5: Bearbeitetes Video exportieren ✅
**Als** Content Creator  
**möchte ich** mein editiertes Video exportieren  
**damit** ich es veröffentlichen kann

**Acceptance Criteria:**
- [x] Export als MP4
- [x] Export als EDL
- [x] Export als OTIO
- [x] Download-Link nach Fertigstellung

---

## E02: Quality Profiles ✅

### Description
Bündle Enhance/Stabilize/Upscale in verständliche Quality-Stufen statt einzelner Toggles.

### User Stories

#### US-02.1: Quality Profile auswählen ✅
**Als** User  
**möchte ich** ein Quality Profile wählen  
**damit** ich nicht einzelne Toggles verstehen muss

**Acceptance Criteria:**
- [x] 4 Profile: Preview, Standard, High, Master
- [x] Jedes Profil hat klare Beschreibung
- [x] Geschätzte Render-Zeit sichtbar
- [x] Ein Klick wählt alle Settings

#### US-02.2: Preview Mode als Default ✅
**Als** User  
**möchte ich** schnell eine Vorschau sehen  
**damit** ich Änderungen iterieren kann

**Acceptance Criteria:**
- [x] Preview: 360p, keine Effekte
- [x] Render-Zeit <30% von Standard
- [x] "Quick Preview" Button prominent

---

## E03: Cloud Acceleration ✅

### Description
Konsolidiere alle CGPU-Optionen in einen einzelnen "Cloud Acceleration" Toggle.

### User Stories

#### US-03.1: Cloud mit einem Schalter aktivieren ✅
**Als** User  
**möchte ich** Cloud-Beschleunigung mit einem Toggle aktivieren  
**damit** ich nicht wissen muss welche Features Cloud nutzen

**Acceptance Criteria:**
- [x] Ein Toggle für alle Cloud-Features
- [x] Auto-Fallback wenn Cloud nicht verfügbar
- [x] Status-Anzeige: Available/Not Configured
- [x] Features-Liste zeigt was aktiviert wird

---

## E04: Shorts Studio ✅ (MVP)

### Description
Dedizierter Workspace für vertikale Video-Erstellung mit Smart Reframe und Captions.

### User Stories

#### US-04.1: Video in Phone-Frame Preview ✅
**Als** Social Creator  
**möchte ich** mein Video im Phone-Format sehen  
**damit** ich weiß wie es auf Mobile aussieht

**Acceptance Criteria:**
- [x] 9:16 Phone Frame UI
- [x] Video Player integriert
- [x] Responsive auf allen Screens

#### US-04.2: Safe Zones anzeigen ✅
**Als** Social Creator  
**möchte ich** Platform Safe Zones sehen  
**damit** wichtiger Content nicht verdeckt wird

**Acceptance Criteria:**
- [x] Top Safe Zone (Platform UI)
- [x] Bottom Safe Zone (Comments/Description)
- [x] Toggle zum Ein/Ausblenden
- [x] Platform-spezifische Presets

#### US-04.3: Reframe Mode wählen ✅
**Als** Social Creator  
**möchte ich** den Reframe-Modus wählen  
**damit** das Cropping meinen Bedürfnissen entspricht

**Acceptance Criteria:**
- [x] Auto (AI)
- [x] Face Track
- [x] Center
- [x] Manual

---

## E05: Caption Styles 🔄

### Description
Verschiedene Caption-Styles für unterschiedliche Plattformen und Ästhetiken.

### User Stories

#### US-05.1: Caption Style auswählen
**Als** Social Creator  
**möchte ich** einen Caption-Style wählen  
**damit** meine Captions zur Plattform passen

**Acceptance Criteria:**
- [ ] Mindestens 4 Styles: Default, Bold, Minimal, Gradient
- [ ] Live Preview im Phone Frame
- [ ] Style wirkt sich auf Burn-in Export aus

#### US-05.2: Caption Position anpassen
**Als** Social Creator  
**möchte ich** die Caption-Position anpassen  
**damit** sie nicht wichtigen Content verdeckt

**Acceptance Criteria:**
- [ ] Positionierung: Top, Center, Bottom
- [ ] Vertikaler Offset einstellbar
- [ ] Preview aktualisiert in Echtzeit

#### US-05.3: Karaoke-Style Captions
**Als** Music Creator  
**möchte ich** Karaoke-Style Captions  
**damit** Lyrics im Rhythmus hervorgehoben werden

**Acceptance Criteria:**
- [ ] Wort-für-Wort Highlighting
- [ ] Sync mit Audio-Timing
- [ ] Farbe für aktives Wort wählbar

---

## E06: Highlight Detection 🔄

### Description
Automatische Erkennung interessanter Momente für Shorts-Clipping.

### User Stories

#### US-06.1: Highlights automatisch erkennen
**Als** Content Creator  
**möchte ich** interessante Momente automatisch finden  
**damit** ich schneller Shorts erstellen kann

**Acceptance Criteria:**
- [ ] Audio-Energy Peaks erkennen
- [ ] Speech-Emphasis erkennen
- [ ] Score/Confidence für jeden Highlight
- [ ] Liste mit Timestamps

#### US-06.2: Highlight manuell bestätigen/ablehnen
**Als** Content Creator  
**möchte ich** vorgeschlagene Highlights bestätigen  
**damit** ich die Kontrolle behalte

**Acceptance Criteria:**
- [ ] Checkbox für jeden Highlight
- [ ] Preview-Jump bei Klick auf Highlight
- [ ] Bulk-Select/Deselect

#### US-06.3: Shorts aus Highlights generieren
**Als** Content Creator  
**möchte ich** aus bestätigten Highlights Shorts erstellen  
**damit** ich schnell mehrere Clips bekomme

**Acceptance Criteria:**
- [ ] Bulk-Export für ausgewählte Highlights
- [ ] Automatisches Reframing pro Clip
- [ ] Captions automatisch hinzufügen

---

## E07: Audio Polish 📋

### Description
Kombiniere Voice Isolation + Noise Reduction in einen "Clean Audio" Toggle.

### User Stories

#### US-07.1: Clean Audio aktivieren
**Als** Content Creator  
**möchte ich** Audio mit einem Toggle verbessern  
**damit** meine Videos professioneller klingen

**Acceptance Criteria:**
- [ ] Ein Toggle: "Clean Audio"
- [ ] Kombiniert Voice Isolation + Denoise
- [ ] A/B Vergleich möglich
- [ ] Auto-Fallback wenn Qualität sinkt

#### US-07.2: SNR automatisch prüfen
**Als** System  
**möchte ich** die Audio-Qualität automatisch prüfen  
**damit** ich weiß ob Cleaning hilft

**Acceptance Criteria:**
- [ ] SNR vor/nach Messung
- [ ] Warnung wenn Original besser
- [ ] User kann Override wählen

---

## E08: Pro Export Pack 📋

### Description
Erweiterte Export-Optionen für professionelle NLE-Workflows.

### User Stories

#### US-08.1: OTIO mit Metadaten exportieren
**Als** Professional Editor  
**möchte ich** OTIO mit allen Metadaten  
**damit** mein NLE alle Informationen hat

**Acceptance Criteria:**
- [ ] Clip-Namen, Timestamps
- [ ] Marker für Beat-Hits
- [ ] Annotations für AI-Decisions
- [ ] Media-Referenzen korrekt

#### US-08.2: Proxies automatisch generieren
**Als** Professional Editor  
**möchte ich** Proxies mit dem Export  
**damit** ich sofort in meinem NLE arbeiten kann

**Acceptance Criteria:**
- [ ] 1/4 Resolution Proxies
- [ ] Matching Naming Convention
- [ ] Relink-Instructions inkludiert

#### US-08.3: Batch Export
**Als** Content Creator  
**möchte ich** mehrere Outputs gleichzeitig exportieren  
**damit** ich Zeit spare

**Acceptance Criteria:**
- [ ] Multi-Select für Clips
- [ ] Queue-basierter Export
- [ ] Progress für alle Jobs sichtbar

---

## E09: Creative Loop v2 📋

### Description
Iteratives Verbessern des Rough Cuts mit LLM-Feedback.

### User Stories

#### US-09.1: Feedback auf Rough Cut geben
**Als** Content Creator  
**möchte ich** Feedback zum Rough Cut geben  
**damit** das System iterativ verbessert

**Acceptance Criteria:**
- [ ] Text-Feedback Feld
- [ ] Schnelle Reactions (👍/👎/🔄)
- [ ] Feedback wird für nächste Iteration genutzt

#### US-09.2: Verbesserung vorschlagen
**Als** System  
**möchte ich** Verbesserungen vorschlagen  
**damit** der User wählen kann

**Acceptance Criteria:**
- [ ] 2-3 Varianten vorschlagen
- [ ] Diff-View zeigt Änderungen
- [ ] User kann akzeptieren/ablehnen

---

## E10: Style Ecosystem 📋

### Description
Community-basiertes Teilen von Style-Presets.

### User Stories

#### US-10.1: Style Preset teilen
**Als** Power User  
**möchte ich** mein Style Preset teilen  
**damit** andere es nutzen können

**Acceptance Criteria:**
- [ ] Export als JSON
- [ ] Upload zu Community Hub
- [ ] Beschreibung/Tags hinzufügen

#### US-10.2: Community Presets durchsuchen
**Als** User  
**möchte ich** Community Presets finden  
**damit** ich neue Styles ausprobieren kann

**Acceptance Criteria:**
- [ ] Browse/Search Interface
- [ ] Preview mit Sample
- [ ] One-Click Import

---

## Technical Tasks (Cross-Cutting)

### Infrastructure

| Task | Priority | Epic | Status |
|------|----------|------|--------|
| Error Handling Overhaul | P0 | All | 🔄 |
| Logging & Monitoring | P1 | All | 📋 |
| Performance Benchmarks | P1 | All | 📋 |
| Automated Testing | P1 | All | 📋 |

### Documentation

| Task | Priority | Status |
|------|----------|--------|
| API Documentation | P1 | 📋 |
| User Guide | P1 | 📋 |
| Developer Guide | P2 | 📋 |
| Video Tutorials | P2 | 📋 |

### DevOps

| Task | Priority | Status |
|------|----------|--------|
| CI/CD Pipeline | P1 | ✅ |
| Docker Optimization | P1 | 🔄 |
| Release Automation | P2 | 📋 |
| Crash Reporting | P2 | 📋 |

---

## Definition of Done

Ein Feature/Story ist "Done" wenn:

- [ ] Code implementiert und reviewed
- [ ] Unit Tests vorhanden (>80% Coverage für neue Code)
- [ ] Integration Tests passieren
- [ ] Documentation aktualisiert
- [ ] UI/UX Review abgeschlossen
- [ ] Performance akzeptabel (<10% Regression)
- [ ] Accessibility geprüft
- [ ] Changelog Entry erstellt

---

*Backlog maintained by Montage AI Team. Prioritization subject to change based on feedback and market conditions.*
