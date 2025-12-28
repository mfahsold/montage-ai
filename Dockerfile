FROM continuumio/miniconda3

# Build argument for version tracking (git commit hash)
ARG GIT_COMMIT=dev
ENV GIT_COMMIT=${GIT_COMMIT}

# Install system dependencies
# Add build tools for Real-ESRGAN-ncnn-vulkan
# Add Node.js for cgpu (cloud GPU via Google Colab)
# Add libvidstab for professional video stabilization
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    imagemagick \
    build-essential \
    cmake \
    git \
    libvulkan-dev \
    glslang-tools \
    glslang-dev \
    libgomp1 \
    unzip \
    wget \
    curl \
    libvidstab-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS for cgpu
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install cgpu globally
RUN npm install -g cgpu

WORKDIR /app

# Install Real-ESRGAN-ncnn-vulkan
# Optimized: Use pre-built binary for x86_64, compile from source only for ARM64
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        echo "Detected x86_64, downloading pre-built binary..." && \
        curl -L -o realesrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
        echo "Download complete. Extracting..." && \
        unzip -q realesrgan.zip -d realesrgan_temp && \
        mv realesrgan_temp/realesrgan-ncnn-vulkan /usr/local/bin/ && \
        chmod +x /usr/local/bin/realesrgan-ncnn-vulkan; \
    else \
        echo "Detected ARM64, building from source..." && \
        git config --global url."https://github.com/".insteadOf git@github.com: && \
        git clone https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan.git && \
        cd Real-ESRGAN-ncnn-vulkan && \
        git submodule update --init --recursive && \
        mkdir build && cd build && \
        cmake ../src && \
        make -j$(nproc) && \
        mv realesrgan-ncnn-vulkan /usr/local/bin/ && \
        cd ../.. && \
        rm -rf Real-ESRGAN-ncnn-vulkan && \
        # Download models separately for ARM64 case
        curl -L -o realesrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip && \
        unzip -q realesrgan.zip -d realesrgan_temp; \
    fi && \
    # Common step: Install models
    mkdir -p /usr/local/share/realesrgan-models && \
    find realesrgan_temp -name "*.param" -exec mv {} /usr/local/share/realesrgan-models/ \; && \
    find realesrgan_temp -name "*.bin" -exec mv {} /usr/local/share/realesrgan-models/ \; && \
    find realesrgan_temp -name "*.pth" -exec mv {} /usr/local/share/realesrgan-models/ \; && \
    rm -rf realesrgan.zip realesrgan_temp && \
    ln -s /usr/local/share/realesrgan-models /usr/local/bin/models

# Install librosa and numba from conda-forge (better ARM64 support)
# Force Python 3.10 to avoid issues with removed modules in 3.13 (like aifc)
# Note: ffmpeg is already installed via apt (line 8) with vidstab support
# Add wget for model downloads
RUN conda install -y -c conda-forge python=3.10 librosa numba numpy scipy wget && \
    conda clean -afy

COPY requirements.txt .
# Install other dependencies via pip, but exclude librosa/numpy since we have them
RUN pip install --no-cache-dir -r requirements.txt

# Copy package structure
COPY src/ /app/src/
COPY pyproject.toml .

# Install package
RUN pip install --no-cache-dir -e .

# Vulkan headless environment for GPU acceleration (AMD/Intel)
# Note: For NVIDIA, use VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

ENTRYPOINT ["python", "-u", "-m", "montage_ai.editor"]
