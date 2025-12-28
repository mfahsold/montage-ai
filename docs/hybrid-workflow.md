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

> **⚠️ Note:** If `cgpu connect` fails with "Access Denied" (Google verification issue), use **Option B** below.

### Option A: Full Hybrid (LLM + GPU via cgpu)

1.  **Start the LLM Proxy:**
    ```bash
    ./montage-ai.sh cgpu-start
    ```

2.  **Run with Cloud Offloading:**
    ```bash
    ./montage-ai.sh run --cgpu --cgpu-gpu --upscale
    ```

### Option B: Direct API (LLM Only) - Recommended Fallback

If `cgpu` is unavailable, you can still offload the LLM to Google's API directly. You lose Cloud GPU upscaling, but you save local RAM for the Creative Director.

1.  **Get a Free API Key:** [Google AI Studio](https://aistudio.google.com/app/apikey)
2.  **Run with API Key:**
    ```bash
    # Export key and run (disable upscaling to save local CPU)
    export GOOGLE_API_KEY="your_key_here"
    ./montage-ai.sh run
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
