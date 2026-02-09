#!/usr/bin/env bash
# Create synthetic test media for montage-ai development/testing
# Generates minimal test video and audio files without needing external media

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_ROOT/test_data}"
DATA_INPUT_DIR="${DATA_INPUT_DIR:-$REPO_ROOT/data/input}"
DATA_MUSIC_DIR="${DATA_MUSIC_DIR:-$REPO_ROOT/data/music}"

# Create directories if they don't exist
mkdir -p "$TEST_DATA_DIR/input" "$TEST_DATA_DIR/music"
mkdir -p "$DATA_INPUT_DIR" "$DATA_MUSIC_DIR"

echo "🎬 Creating synthetic test media..."

# Create a 10-second test video using FFmpeg
# - Resolution: 1920x1080 (Full HD default; changeable)
# - Codec: H.264 (compatible)
# - Colors: Gradient pattern for visual interest
# - Format: MP4
# - Size: ~1-2 MB (depending on duration)

VIDEO_FILE="${TEST_DATA_DIR}/input/test-video.mp4"
if [ ! -f "$VIDEO_FILE" ]; then
    echo "  📹 Generating test video (10 sec, 1920x1080)..."
    ffmpeg -f lavfi -i "color=c=blue:s=1920x1080:d=10" \
           -f lavfi -i "sine=f=440:d=10" \
           -vf "drawtext=text='Test Video':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" \
           -pix_fmt yuv420p -y "$VIDEO_FILE" 2>&1 | grep -E "(frame|Duration|time=)" || true
    echo "  ✅ Created: $VIDEO_FILE"
fi

# Create a 10-second test audio file using FFmpeg
# - Format: MP3 or WAV
# - Sample rate: 44.1 kHz (standard)
# - Channels: Stereo
# - Duration: 10 seconds
# - Frequency: 440 Hz (standard A note tuning)

AUDIO_FILE="${TEST_DATA_DIR}/music/test-audio.mp3"
if [ ! -f "$AUDIO_FILE" ]; then
    echo "  🎵 Generating test audio (10 sec, stereo)..."
    ffmpeg -f lavfi -i "sine=f=440:d=10" \
           -f lavfi -i "sine=f=880:d=10" \
           -filter_complex "[0][1]amerge=inputs=2[a]" \
           -map "[a]" \
           -q:a 5 -y "$AUDIO_FILE" 2>&1 | grep -E "(frame|Duration|time=)" || true
    echo "  ✅ Created: $AUDIO_FILE"
fi

# Optional: Create multi-clip test video for scene detection testing
MULTI_CLIP_FILE="${TEST_DATA_DIR}/input/test-multi-scene.mp4"
if [ ! -f "$MULTI_CLIP_FILE" ]; then
    echo "  📹 Generating multi-scene test video (30 sec with 3 scenes)..."
    # Create 3 segments of different colors (each 10 sec)
    ffmpeg -f lavfi -i "color=c=red:s=1920x1080:d=10" \
           -f lavfi -i "color=c=green:s=1920x1080:d=10" \
           -f lavfi -i "color=c=blue:s=1920x1080:d=10" \
           -f lavfi -i "sine=f=440:d=30" \
           -filter_complex "[0][1][2]concat=n=3:v=1:a=0[v];[3]aformat=sample_fmts=s16:sample_rates=44100[a]" \
           -map "[v]" -map "[a]" \
           -c:v libx264 -c:a aac -y "$MULTI_CLIP_FILE" 2>&1 | grep -E "(frame|Duration|time=)" || true
    echo "  ✅ Created: $MULTI_CLIP_FILE (multi-scene for testing)"
fi

# Copy test files to data/ for quick access
cp "$VIDEO_FILE" "$DATA_INPUT_DIR/test-video.mp4"
cp "$AUDIO_FILE" "$DATA_MUSIC_DIR/test-audio.mp3"
if [ -f "$MULTI_CLIP_FILE" ]; then
    cp "$MULTI_CLIP_FILE" "$DATA_INPUT_DIR/test-multi-scene.mp4"
fi

echo ""
echo "✅ Test media created successfully!"
echo ""
echo "📂 Files created:"
echo "   Video: $DATA_INPUT_DIR/test-video.mp4"
echo "   Audio: $DATA_MUSIC_DIR/test-audio.mp3"
if [ -f "$MULTI_CLIP_FILE" ]; then
    echo "   Multi-scene: $DATA_INPUT_DIR/test-multi-scene.mp4"
fi
echo ""
echo "🚀 Quick start:"
echo "   QUALITY_PROFILE=preview docker compose run --rm montage-ai ./montage-ai.sh run"
echo ""
