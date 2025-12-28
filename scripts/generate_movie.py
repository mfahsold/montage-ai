#!/usr/bin/env python3
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

try:
    from montage_ai.open_sora import OpenSoraGenerator, OpenSoraConfig, OpenSoraResolution
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("ℹ️  This script requires dependencies (librosa, etc.) installed in the Docker container.")
    print("   Run with: ./montage-ai.sh run-script scripts/generate_movie.py")
    print("   Or: docker compose run --rm montage-ai python3 scripts/generate_movie.py")
    sys.exit(1)

def generate_movie():
    print("🎬 Starting Movie Generation (Open-Sora)...")
    
    # Configure generator
    config = OpenSoraConfig(
        resolution=OpenSoraResolution.LOW, # 256p for speed/compatibility
        num_frames=51, # ~2 seconds
        use_cgpu=True
    )
    
    generator = OpenSoraGenerator(config)
    
    prompt = "A cinematic shot of a futuristic city with flying cars, cyberpunk style, high detail, 4k"
    print(f"   Prompt: '{prompt}'")
    
    try:
        output_path = generator.generate(prompt, duration=2.0)
        
        # Move to data/output if successful
        if output_path and os.path.exists(output_path):
            final_path = os.path.join("/data/output", os.path.basename(output_path))
            os.rename(output_path, final_path)
            print(f"✅ Movie generated successfully: {final_path}")
            return final_path
        else:
            print(f"✅ Movie generated successfully: {output_path}")
            return output_path
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

if __name__ == "__main__":
    generate_movie()
