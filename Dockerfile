# Use a slim python image for smaller size and faster builds
FROM python:3.10-slim-bookworm

# Install system dependencies
# ffmpeg: for video processing
# libsndfile1: for audio analysis (librosa)
# libgl1: for opencv
# curl, unzip: for downloading assets
# build-essential: for compiling some python packages
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg     libsndfile1     libgl1     curl     unzip     build-essential     && rm -rf /var/lib/apt/lists/*

# Install cgpu CLI
# Note: nodejs is not installed in python-slim by default.
# We need to install nodejs first if we want npm.
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
RUN npm install -g cgpu@latest

WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install python dependencies
# Use --no-cache-dir to keep image small
# Use --prefer-binary to avoid compiling from source when possible
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Conditional download of Real-ESRGAN based on architecture
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then         echo "Downloading Real-ESRGAN for AMD64..." &&         curl -L -o realesrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip &&         unzip -q realesrgan.zip -d realesrgan_temp &&         find realesrgan_temp -name "realesrgan-ncnn-vulkan" -exec mv {} /usr/local/bin/ \; &&         chmod +x /usr/local/bin/realesrgan-ncnn-vulkan &&         mkdir -p /usr/local/share/realesrgan-models &&         find realesrgan_temp -name "*.param" -exec mv {} /usr/local/share/realesrgan-models/ \; &&         find realesrgan_temp -name "*.bin" -exec mv {} /usr/local/share/realesrgan-models/ \; &&         rm -rf realesrgan.zip realesrgan_temp;     else         echo "Skipping Real-ESRGAN for $TARGETARCH (not supported or not needed)";     fi

# Copy application code
COPY . .

# Expose port for web UI
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/montage_ai/web_ui/app.py

# Default command
CMD ["./montage-ai.sh", "web"]
