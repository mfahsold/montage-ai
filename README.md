# Montage AI

AI-powered automatic video montage creation with beat-synchronized editing.

## Quick Start

```bash
# Build
docker build -t montage-ai .

# Run
./montage-ai.sh

# With style
CREATIVE_PROMPT="Edit like Hitchcock" ./montage-ai.sh
```

## Data Structure

```
data/
├── input/   # Your video files
├── music/   # Your audio track
└── output/  # Generated videos
```

## Styles

```bash
./montage-ai.sh                           # Default dynamic
CREATIVE_PROMPT="hitchcock" ./montage-ai.sh    # Suspense
CREATIVE_PROMPT="mtv" ./montage-ai.sh          # Fast cuts
CREATIVE_PROMPT="documentary" ./montage-ai.sh  # Natural pacing
CREATIVE_PROMPT="minimalist" ./montage-ai.sh   # Long takes
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CREATIVE_PROMPT` | - | Style: hitchcock, mtv, documentary, minimalist |
| `STABILIZE` | false | Video stabilization |
| `ENHANCE` | true | Color/sharpness boost |
| `NUM_VARIANTS` | 1 | Generate multiple versions |

## Modules

### Core (Stable)
- **editor.py** - Main editing engine with beat-sync
- **style_templates.py** - Predefined editing styles
- **footage_manager.py** - Clip tracking and story arc
- **monitoring.py** - Progress tracking

### Experimental
- **timeline_exporter.py** - Export to DaVinci/Premiere (OTIO/EDL) [WIP]
- **footage_analyzer.py** - Deep visual analysis for AI storytelling [WIP]

## Roadmap

- [x] Beat-synchronized editing
- [x] Style templates
- [x] Footage consumption tracking
- [ ] Timeline export (OTIO/EDL)
- [ ] AI-powered story generation from footage analysis

## License

MIT
