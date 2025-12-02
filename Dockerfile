FROM continuumio/miniconda3

# Install system dependencies
# Add build tools for Real-ESRGAN-ncnn-vulkan
# Add Node.js for cgpu (cloud GPU via Google Colab)
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
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS for cgpu
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install cgpu globally
RUN npm install -g cgpu

WORKDIR /app

# Build Real-ESRGAN-ncnn-vulkan from source for ARM64 optimization
RUN git config --global url."https://github.com/".insteadOf git@github.com: && \
    git clone https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan.git && \
    cd Real-ESRGAN-ncnn-vulkan && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake ../src && \
    make -j$(nproc) && \
    mv realesrgan-ncnn-vulkan /usr/local/bin/ && \
    cd ../.. && \
    rm -rf Real-ESRGAN-ncnn-vulkan

# Download Real-ESRGAN models
# Note: realesr-animevideov3 models are inside the release zip.
# We download the zip, extract models, and place them in a dedicated directory.
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip -O realesrgan.zip && \
    unzip realesrgan.zip -d realesrgan_temp && \
    # Create model directory
    mkdir -p /usr/local/share/realesrgan-models && \
    # Find and move all model files (.param, .bin, .pth) to the model directory
    find realesrgan_temp -name "*.param" -exec mv {} /usr/local/share/realesrgan-models/ \; && \
    find realesrgan_temp -name "*.bin" -exec mv {} /usr/local/share/realesrgan-models/ \; && \
    find realesrgan_temp -name "*.pth" -exec mv {} /usr/local/share/realesrgan-models/ \; && \
    # Cleanup
    rm -rf realesrgan.zip realesrgan_temp

# Install librosa and numba from conda-forge (better ARM64 support)
# Force Python 3.10 to avoid issues with removed modules in 3.13 (like aifc)
# Install ffmpeg via conda to avoid library conflicts
# Add wget for model downloads
RUN conda install -y -c conda-forge python=3.10 librosa numba numpy scipy ffmpeg wget && \
    conda clean -afy

COPY requirements.txt .
# Install other dependencies via pip, but exclude librosa/numpy since we have them
RUN pip install --no-cache-dir -r requirements.txt

# Copy package structure
COPY src/ /app/src/
COPY pyproject.toml .

# Install package
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-u", "-m", "montage_ai.editor"]
