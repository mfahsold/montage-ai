# Hybrid Workflow: Low Memory & Cloud GPU

This guide explains how to run Montage AI on a standard laptop (e.g., 16GB RAM) by offloading memory-intensive tasks to the cloud using `cgpu`.

## Concept

Running local LLMs (like Llama 3 70B) and AI upscalers requires significant VRAM and RAM. By using `cgpu`, we can offload:
1.  **Creative Intelligence (LLM):** Uses Google Gemini via `cgpu serve` (free tier).
2.  **Heavy Processing (Upscaling):** Uses Google Colab GPUs via `cgpu run`.

This leaves your local machine to handle only the lightweight orchestration and video cutting.

## Architecture

```mermaid
graph TD
    subgraph Local Laptop [Local Laptop (16GB RAM)]
        Orchestrator[Montage AI (Docker)]
        Editor[Video Editor (CPU)]
        CGPU_Client[cgpu Client]
    end
    
    subgraph Cloud [Google Cloud / Colab]
        Gemini[Gemini 2.0 Flash (LLM)]
        ColabGPU[Colab T4/A100 GPU]
    end
    
    Orchestrator -->|Prompt| CGPU_Client
    CGPU_Client -->|API Request| Gemini
    Gemini -->|JSON Instructions| Orchestrator
    
    Orchestrator -->|Video Upload| ColabGPU
    ColabGPU -->|Real-ESRGAN Upscale| ColabGPU
    ColabGPU -->|Download Result| Orchestrator
```

## Prerequisites

1.  **Node.js & npm** (for cgpu)
2.  **cgpu CLI**: `npm install -g cgpu`
3.  **gemini-chat-cli**: `pip install gemini-chat-cli` (or follow [gemini-cli instructions](https://github.com/google-gemini/gemini-cli))
4.  **Google Account** (for Colab and Gemini)

## Setup Guide

### 1. Configure cgpu

Authenticate with Google Colab:
```bash
cgpu connect
```

### 2. Start the LLM Proxy

On your host machine (not inside Docker), start the Gemini proxy:
```bash
# This exposes Gemini as an OpenAI-compatible API on port 8080
./montage-ai.sh cgpu-start
```

### 3. Run Montage AI in Hybrid Mode

Use the following flags to enable the hybrid workflow:

```bash
# Run with cloud offloading
./montage-ai.sh run \
  --cgpu \          # Use Gemini for Creative Director (saves ~8GB RAM vs local Llama)
  --cgpu-gpu \      # Use Cloud GPU for upscaling (saves local CPU/GPU)
  --upscale         # Enable upscaling (now runs on cloud)
```

## Configuration Details

For permanent configuration, update your `.env` or `docker-compose.yml`:

```yaml
environment:
  # Offload LLM to Cloud
  - CGPU_ENABLED=true
  - CGPU_HOST=host.docker.internal
  - CGPU_PORT=8080
  
  # Offload Upscaling to Cloud
  - CGPU_GPU_ENABLED=true
  - CGPU_TIMEOUT=1800
  
  # Optimize Local Resources
  - MEMORY_LIMIT_GB=8
  - MAX_PARALLEL_JOBS=2
  - OLLAMA_HOST=  # Disable local Ollama connection
```

## Performance Impact

| Task | Local (Laptop CPU) | Hybrid (Cloud GPU) |
|------|-------------------|-------------------|
| **Creative Director** | Slow / OOM (Llama 70B) | **Instant** (Gemini Flash) |
| **Upscaling (1min)** | ~20 mins | **~2 mins** |
| **RAM Usage** | 14GB+ | **< 6GB** |

## Troubleshooting

- **Connection Refused:** Ensure `cgpu serve` is running on the host and `CGPU_HOST` is set to `host.docker.internal`.
- **Timeout:** Large videos may take longer to upload/download. Increase `CGPU_TIMEOUT`.
