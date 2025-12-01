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

## Directory Structure

```
data/
├── input/   # Your video files
├── music/   # Your audio track
└── output/  # Generated videos
```

## Styles

| Style | Description |
|-------|-------------|
| hitchcock | Slow build-up, explosive climax |
| mtv | Fast 1-2 beat cuts |
| action | Michael Bay rapid cuts |
| documentary | Natural pacing |
| minimalist | Long contemplative takes |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CREATIVE_PROMPT` | - | Natural language style description |
| `STABILIZE` | false | Video stabilization |
| `ENHANCE` | true | Color/sharpness boost |
| `NUM_VARIANTS` | 1 | Generate multiple versions |

## License

MIT
