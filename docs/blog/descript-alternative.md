# Descript Alternative (Offline) with OTIO Export

**TL;DR:** Montage AI = lokale, deterministische Rough-Cuts mit Beat-Sync, Story-Arcs und OTIO/EDL-Handoff. Kein Upload, keine Credits, volle Kontrolle.

## Warum offline?
- **Privacy:** Kundendaten und Rohmaterial verlassen die Maschine nicht.
- **Speed:** Keine Upload-Wartezeiten, große Files direkt von SSD/NAS.
- **Kosten:** Keine Minuten-Credits, keine Paywalls pro Export.
- **Deterministisch:** Gleiche Eingabe → gleiche Timeline, reproduzierbar für Revisionen.

## Was unterscheidet uns von Descript & Co?
- **OTIO/EDL Export:** Direkt in DaVinci/Premiere weiterarbeiten (Pro Handoff).
- **Beat-Sync + Story Arcs:** Automatische Musik-Synchronisation und 5-Phasen-Narrativ.
- **Local-First:** FFmpeg/Whisper/Auto-Editor laufen lokal; optional cgpu-Cloud nur wenn explizit gewünscht.
- **Batch/CLI:** `make dev-test` lokal, `make cluster` für Multi-Arch/Cluster. Skriptbar für CI.
- **Longform-tauglich:** Podcasts, Lectures, Streams ohne Upload-Limits.

## Quick Start
```bash
# Lokal: 5 Sekunden Feedback
make dev       # einmalig build
make dev-test  # edit → test (volume mounts)

# Cluster (multi-arch, shared cache)
make cluster   # build + push + deploy
```

## Workflow: Podcast → Shorts (offline)
1) Audio/Video in `data/input/` ablegen
2) `make dev-test` starten (nutzt lokalen Cache)
3) Montage AI erzeugt Rough Cut + optional 9:16 Clips
4) Export als `montage.otio` → Import in DaVinci/Premiere für Finishing

## SEO / Suchbegriffe
- „descript alternative offline“
- „ai video editor local“
- „beat sync video editor“
- „otio export video editor“
- „podcast to shorts offline“

## Warum OTIO?
OTIO ist ein offener Timeline-Standard. Damit:
- Sauberer Handoff zu NLEs (Premiere/Resolve/Avid via EDL/XML)
- Keine Black-Box-XML-Skripte
- Stabil für Round-Trips mit Audio/Video-Sync

## Roadmap (relevant für Offline-User)
- Verbesserte Caption-Styling-Presets (lokal)
- Batching von Hooks/Highlights für Shorts
- Weitere Audio-Cleanup-Presets (LUFS, DeReverb, Auto-Duck)

---

**Jetzt ausprobieren:** [GitHub Repo](https://github.com/mfahsold/montage-ai)
