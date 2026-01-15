# Use a slim python image for smaller size and faster builds
FROM python:3.10-slim-bookworm

# Enable non-free repositories for Intel QSV drivers
# Required for intel-media-va-driver-non-free (fixes issue #3299)
RUN echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# Install system dependencies including Intel QSV hardware acceleration
# Intel QSV packages fix MFX_ERR_DEVICE_FAILED (-9) error
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core dependencies
    ffmpeg \
    libsndfile1 \
    libgl1 \
    curl \
    unzip \
    build-essential \
    nodejs \
    npm \
    # VA-API base libraries (Video Acceleration API)
    vainfo \
    libva2 \
    libva-drm2 \
    # Intel QSV drivers and runtime (non-free for full hardware support)
    intel-media-va-driver-non-free \
    # AMD VAAPI drivers (Mesa)
    mesa-va-drivers \
    # Intel Media SDK runtime library (required for MFX session creation)
    libmfx1 \
    # Intel Video Processing Library (VPL) - newer replacement for MFX
    libvpl2 \
    && rm -rf /var/lib/apt/lists/*

# Install cgpu CLI
RUN npm install -g cgpu@latest

WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install python dependencies
# Use --no-cache-dir to keep image small
# Use --prefer-binary to avoid compiling from source when possible
# Increased timeout to 600s for large packages (opencv-python-headless ~54MB)
# Added retries=5 for network resilience
RUN pip install --default-timeout=600 --retries 5 --no-cache-dir --prefer-binary -r requirements.txt

# Conditional download of Real-ESRGAN based on architecture
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then         echo "Downloading Real-ESRGAN for AMD64..." &&         curl -L -o realesrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip &&         unzip -q realesrgan.zip -d realesrgan_temp &&         find realesrgan_temp -name "realesrgan-ncnn-vulkan" -exec mv {} /usr/local/bin/ \; &&         chmod +x /usr/local/bin/realesrgan-ncnn-vulkan &&         mkdir -p /usr/local/share/realesrgan-models &&         find realesrgan_temp -name "*.param" -exec mv {} /usr/local/share/realesrgan-models/ \; &&         find realesrgan_temp -name "*.bin" -exec mv {} /usr/local/share/realesrgan-models/ \; &&         rm -rf realesrgan.zip realesrgan_temp;     else         echo "Skipping Real-ESRGAN for $TARGETARCH (not supported or not needed)";     fi

# Copy application code
COPY . .

# Clean Python cache files to ensure fresh imports
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -type f -name "*.pyo" -delete 2>/dev/null || true

# Create non-root user for security and correct file permissions
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} montage && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash montage && \
    chown -R montage:montage /app

# Expose port for web UI
# Allow override via build-arg/service config; default kept as 5000 for backwards compatibility
ARG SERVICE_PORT=5000
EXPOSE ${SERVICE_PORT}
ENV SERVICE_PORT=${SERVICE_PORT}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/montage_ai/web_ui/app.py
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Intel QSV hardware acceleration environment variables
# LIBVA_DRIVER_NAME=iHD selects the Intel iHD driver for Quick Sync Video
# MFX_IMPL_DEVICE specifies the render device for MFX session initialization
ENV LIBVA_DRIVER_NAME=iHD
ENV MFX_IMPL_DEVICE=/dev/dri/renderD128

# Set PYTHONPATH to include the source directory
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER montage

# Default command
WORKDIR /app/src
CMD ["sh", "-c", "../montage-ai.sh cgpu-start && python3 -m montage_ai.web_ui.app"]
