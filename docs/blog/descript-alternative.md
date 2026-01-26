# Descript Alternative (Offline) with OTIO Export

**TL;DR:** Montage AI = local, deterministic rough-cuts with beat-sync, story arcs, and OTIO/EDL handoff. No upload, no credits, full control.

## Why Offline?
- **Privacy:** Client data and raw footage never leave your machine.
- **Speed:** No upload wait times, large files directly from SSD/NAS.
- **Cost:** No per-minute credits, no paywalls per export.
- **Deterministic:** Same input = same timeline, reproducible for revisions.

## What Sets Us Apart from Descript & Co?
- **OTIO/EDL Export:** Continue editing directly in DaVinci/Premiere (Pro handoff).
- **Beat-Sync + Story Arcs:** Automatic music synchronization and 5-phase narrative.
- **Local-First:** FFmpeg/Whisper/Auto-Editor run locally; optional cgpu cloud only when explicitly requested.
- **Batch/CLI:** `./montage-ai.sh run` locally, `make -C deploy/k3s deploy-production` for cluster.
- **Longform-ready:** Podcasts, lectures, streams without upload limits.

## Quick Start
```bash
# Local: 5-second feedback
./montage-ai.sh run

# Cluster (multi-arch, shared cache)
make -C deploy/k3s deploy-production
```

## Workflow: Podcast to Shorts (Offline)
1) Place audio/video in `data/input/`
2) Run `./montage-ai.sh run` (uses local cache)
3) Montage AI generates rough cut + optional 9:16 clips
4) Export as `montage.otio` -> Import into DaVinci/Premiere for finishing

## SEO / Search Terms
- "descript alternative offline"
- "ai video editor local"
- "beat sync video editor"
- "otio export video editor"
- "podcast to shorts offline"

## Why OTIO?
OTIO is an open timeline standard. This enables:
- Clean handoff to NLEs (Premiere/Resolve/Avid via EDL/XML)
- No black-box XML scripts
- Stable round-trips with audio/video sync

## Roadmap (Relevant for Offline Users)
- Improved caption styling presets (local)
- Batching of hooks/highlights for Shorts
- Additional audio cleanup presets (LUFS, DeReverb, Auto-Duck)

---

**Try it now:** [GitHub Repo](https://github.com/mfahsold/montage-ai)
