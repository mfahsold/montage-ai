# Montage AI

AI-powered video montage with beat-synchronized editing.

## Quick Start

```bash
# Build once
./montage-ai.sh build

# Run with default style
./montage-ai.sh run

# Or pick a style
./montage-ai.sh run hitchcock
./montage-ai.sh run mtv
./montage-ai.sh run documentary
```

## Commands

| Command | Description |
|---------|-------------|
| `run [STYLE]` | Create montage |
| `preview [STYLE]` | Fast preview |
| `hq [STYLE]` | High quality render |
| `list` | Show available styles |
| `build` | Build Docker image |

## Styles

| Style | Description |
|-------|-------------|
| `dynamic` | Position-aware pacing (default) |
| `hitchcock` | Slow build, explosive climax |
| `mtv` | Fast 1-2 beat cuts |
| `action` | Michael Bay rapid cuts |
| `documentary` | Natural pacing |
| `minimalist` | Long contemplative takes |

## Options

```bash
./montage-ai.sh run --stabilize           # Enable stabilization
./montage-ai.sh run --variants 3          # Generate 3 variants
./montage-ai.sh hq hitchcock --stabilize  # HQ + stabilize
```

## Data

```
data/
├── input/   # 51 video clips (gallery footage)
├── music/   # aggressive-techno-409194.mp3
└── output/  # Generated videos
```

## Modules

**Core:** editor, style_templates, footage_manager, monitoring  
**Experimental:** timeline_exporter (OTIO/EDL), footage_analyzer

## License

MIT
