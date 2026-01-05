# =============================================================================
# Montage AI - Optimized Multi-Stage Dockerfile
# =============================================================================
# Stage 1: Build dependencies (cached)
# Stage 2: Runtime (minimal)
#
# Build with cache:
#   docker buildx build --builder multiarch-builder \
#     --platform linux/amd64,linux/arm64 \
#     --cache-from type=registry,ref=registry:5000/montage-ai:buildcache \
#     --cache-to type=registry,ref=registry:5000/montage-ai:buildcache,mode=max \
#     -t registry:5000/montage-ai:latest .
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: Base with system dependencies
# -----------------------------------------------------------------------------
FROM continuumio/miniconda3:latest AS base

# Build arguments
ARG GIT_COMMIT=dev
ENV GIT_COMMIT=${GIT_COMMIT}
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in one layer (cached unless apt changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core media tools
    ffmpeg \
    libsndfile1 \
    libgl1 \
    imagemagick \
    libvidstab-dev \
    # Vulkan for GPU acceleration
    mesa-vulkan-drivers \
    vulkan-tools \
    libvulkan-dev \
    # VAAPI for hardware video encoding (AMD/Intel)
    libva2 \
    libva-drm2 \
    mesa-va-drivers \
    vainfo \
    # Node.js prerequisites
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS for cgpu (separate layer, cached)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install global npm packages (separate layer to cache better)
RUN npm install -g cgpu@latest @google/gemini-cli && \
    npm cache clean --force

# -----------------------------------------------------------------------------
# STAGE 2: Python dependencies (cached unless requirements change)
# -----------------------------------------------------------------------------
FROM base AS python-deps

WORKDIR /app

# Force Python 3.10 and install conda packages with pinned versions
# Note: numba>=0.60 + librosa>=0.10.2 have compatible APIs (no get_call_template errors)
RUN conda install -y -c conda-forge \
    python=3.10 \
    librosa=0.10.2 \
    numba=0.60.0 \
    numpy=1.26.4 \
    scipy \
    wget && \
    conda clean -afy

COPY requirements.txt .

# Install pip dependencies (no-cache-dir to keep image small, but layer is cached by BuildKit)
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 3: Real-ESRGAN (cached, arch-specific)
# -----------------------------------------------------------------------------
FROM python-deps AS realesrgan

# Install Real-ESRGAN binary and models
RUN curl -L -o realesrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
    unzip -q realesrgan.zip -d realesrgan_temp && \
    if [ "$(uname -m)" = "x86_64" ]; then \
        mv realesrgan_temp/realesrgan-ncnn-vulkan /usr/local/bin/ && \
        chmod +x /usr/local/bin/realesrgan-ncnn-vulkan; \
    else \
        echo "WARNING: Skipping Real-ESRGAN build on non-x86 to speed up build. Upscaling will be unavailable." && \
        touch /usr/local/bin/realesrgan-ncnn-vulkan && \
        chmod +x /usr/local/bin/realesrgan-ncnn-vulkan; \
    fi && \
    mkdir -p /usr/local/share/realesrgan-models && \
    (find realesrgan_temp -name "*.param" -exec mv {} /usr/local/share/realesrgan-models/ \; 2>/dev/null || true) && \
    (find realesrgan_temp -name "*.bin" -exec mv {} /usr/local/share/realesrgan-models/ \; 2>/dev/null || true) && \
    rm -rf realesrgan.zip realesrgan_temp && \
    ln -sf /usr/local/share/realesrgan-models /usr/local/bin/models

# -----------------------------------------------------------------------------
# STAGE 4: Final runtime image
# -----------------------------------------------------------------------------
FROM realesrgan AS runtime

WORKDIR /app

# Copy only pyproject.toml first (for dependency metadata)
COPY pyproject.toml .

# Create minimal src structure for editable install
RUN mkdir -p /app/src/montage_ai && \
    touch /app/src/montage_ai/__init__.py

# Install package in editable mode (cached unless pyproject.toml changes)
RUN pip install --no-cache-dir -e .

# Copy application code (changes most frequently - final layer)
COPY src/ /app/src/
RUN pip install --no-cache-dir -e .

# Vulkan headless environment
# ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from montage_ai.config import get_settings; print('OK')" || exit 1

ENTRYPOINT ["python", "-u", "-m", "montage_ai.editor"]
