# Deployment Scenarios & Installation Guide

Montage AI is designed to scale from a single laptop to a distributed cluster. This guide outlines the supported deployment models and their ideal use cases.

## 1. Local Installation (CLI / Script)
**Best for:** Developers, Contributors, Quick Testing

Run Montage AI directly on your host machine. This requires Python 3.10+ and FFmpeg installed locally.

*   **Setup:** `pip install -r requirements.txt`
*   **Execution:** `./montage-ai.sh run` (detects local environment)
*   **Pros:** Zero overhead, direct access to local files, easiest for debugging code.
*   **Cons:** Dependency management can be tricky (system FFmpeg versions, Python environments).

## 2. Docker Container (Standard)
**Best for:** Content Creators, Hobbyists, Consistent Production

The recommended way to run Montage AI. Encapsulates all dependencies (FFmpeg, Python libraries, ImageMagick) in a reproducible environment.

*   **Setup:** `docker-compose up`
*   **Execution:** `./montage-ai.sh web` or `./montage-ai.sh run` (wraps Docker commands)
*   **Pros:** "It just works", isolated environment, no system pollution.
*   **Cons:** Requires Docker Desktop/Engine.

## 3. Hybrid Cloud (Local + cgpu)
**Best for:** Power Users, Freelancers, Agencies

Run the orchestration locally (Docker or CLI) but offload heavy AI tasks (Upscaling, Transcription, LLM reasoning) to a Cloud GPU via `cgpu`.

*   **Setup:** Configure `CGPU_ENABLED=true` and `CGPU_API_KEY`.
*   **Execution:** `./montage-ai.sh run --cloud-only`
*   **Pros:** High performance on low-end hardware (e.g., MacBook Air), cost-effective (pay only for compute used).
*   **Cons:** Requires internet connection, small latency for asset upload.

## 4. Distributed Cluster (Kubernetes / K3s)
**Best for:** Enterprise, High-Volume Platforms, SaaS Providers

Deploy Montage AI as a scalable job queue on Kubernetes. Supports mixed-architecture clusters (AMD64/ARM64) and shared storage for asset caching.

*   **Setup:** `kubectl apply -f deploy/k3s/`
*   **Execution:** Job-based architecture triggered via API or Cron.
*   **Pros:** Infinite horizontal scaling, fault tolerance, automated resource management.
*   **Cons:** High complexity, requires infrastructure management.

---

## Real-World Use Cases

### 🎥 The Social Media Creator (Personal)
*   **Scenario:** You have 50 clips from a weekend trip and want a 30-second Reel set to trending audio.
*   **Deployment:** **Docker (Web UI)**.
*   **Workflow:** Upload clips -> Select "Viral" style -> Download result.
*   **Value:** Saves 2-3 hours of manual editing.

### 🏢 The Marketing Agency (Professional)
*   **Scenario:** Generating 100 variations of a product ad for A/B testing across different demographics and aspect ratios.
*   **Deployment:** **Hybrid Cloud**.
*   **Workflow:** Scripted CLI execution with different `CREATIVE_PROMPT`s. Offload upscaling to ensure 4K quality without tying up local workstations.
*   **Value:** Massive throughput, consistent branding, frees up editors for creative work.

### 🏭 The Media Platform (Enterprise)
*   **Scenario:** A sports broadcaster needs to generate highlight reels for every player immediately after a match.
*   **Deployment:** **Kubernetes Cluster**.
*   **Workflow:** Ingest live feed -> Distributed workers analyze scenes -> Auto-assemble highlights based on metadata (e.g., "Goal", "Save").
*   **Value:** Real-time content generation at scale, zero human intervention required.
