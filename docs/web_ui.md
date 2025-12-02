# Web UI Guide

Simple, self-hosted web interface for Montage AI.

---

## Quick Start

### Local (Docker Compose)

```bash
# Start web UI
make web

# Or manually:
docker-compose -f docker-compose.web.yml up
```

Open **http://localhost:5000** in your browser.

### Kubernetes

```bash
# Deploy web UI
make web-deploy

# Get service URL
kubectl get svc -n montage-ai montage-ai-web
```

---

## Usage

### 1. Upload Media

- **Videos**: MP4, MOV, AVI, MKV
- **Music**: MP3, WAV, FLAC

Drag & drop or click to upload.

### 2. Configure

- **Style**: Choose cinematic preset (Hitchcock, MTV, etc.)
- **Prompt** (optional): Natural language instructions
- **Options**:
  - ✨ Enhance (color, sharpness)
  - ⚖️ Stabilize
  - 🔍 AI Upscale
  - ☁️ Use Cloud GPU (cgpu)
  - 📽️ Export Timeline (OTIO/EDL)

### 3. Create Montage

Click **Create Montage** → job starts in background.

### 4. Download

Once completed, download:
- **Video** (MP4)
- **Timeline** (OTIO/EDL) - if enabled

---

## API Endpoints

### Health Check
```bash
GET /api/status
```

### List Files
```bash
GET /api/files
```

### Upload File
```bash
POST /api/upload
Content-Type: multipart/form-data

file: <file>
type: video | music
```

### Create Job
```bash
POST /api/jobs
Content-Type: application/json

{
  "style": "hitchcock",
  "prompt": "suspenseful editing",
  "stabilize": false,
  "upscale": false,
  "enhance": true,
  "cgpu": false,
  "export_timeline": false,
  "generate_proxies": false
}
```

### Get Job Status
```bash
GET /api/jobs/{job_id}
```

### List Jobs
```bash
GET /api/jobs
```

### Download File
```bash
GET /api/download/{filename}
```

---

## Architecture

**Backend**: Flask (Python)
**Frontend**: Vanilla JS (no build step)
**Storage**: Persistent volumes for input/output

```
┌─────────────────┐
│   Browser       │
│  (index.html)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Flask App     │
│   (app.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   editor.py     │
│ (subprocess)    │
└─────────────────┘
```

---

## Configuration

Environment variables:

```bash
INPUT_DIR=/data/input          # Video uploads
MUSIC_DIR=/data/music           # Audio uploads
OUTPUT_DIR=/data/output         # Rendered montages
FLASK_ENV=production            # Flask mode
OLLAMA_HOST=http://...          # LLM endpoint
CGPU_ENABLED=false              # Cloud GPU toggle
```

---

## Development

### Run in Debug Mode

```bash
docker-compose -f docker-compose.web.yml up
```

Edit `src/montage_ai/web_ui/` files → reload browser.

### Run Tests

```bash
pytest tests/test_web_ui.py -v
```

---

## Deployment Options

### 1. Docker Compose (Simplest)

```yaml
# docker-compose.web.yml
version: '3.8'
services:
  web-ui:
    image: ghcr.io/mfahsold/montage-ai:latest
    command: python -m montage_ai.web_ui.app
    ports:
      - "5000:5000"
    volumes:
      - ./data:/data
```

### 2. Kubernetes (Production)

```bash
kubectl apply -f deploy/k3s/base/web-service.yaml
```

Exposes LoadBalancer on port 80.

### 3. Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name montage.example.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Increase upload limit for videos
    client_max_body_size 500M;
}
```

---

## Limitations

### Current Implementation

- **Job Queue**: In-memory (lost on restart)
- **Concurrency**: 1 job at a time per instance
- **Storage**: Local filesystem only

### For Production (Future)

- **Job Queue**: Redis + Celery
- **Concurrency**: Multiple workers
- **Storage**: S3/MinIO object storage
- **Auth**: User accounts & API keys

---

## Troubleshooting

### "Connection refused" error

**Check if web UI is running:**

```bash
docker ps | grep web-ui
curl http://localhost:5000/api/status
```

### Upload fails

**Check disk space:**

```bash
df -h data/input/
```

**Check file size limit** (500 MB default):

Edit `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1 GB
```

### Job stuck in "running" state

**Check logs:**

```bash
docker-compose -f docker-compose.web.yml logs -f
```

**Restart web UI:**

```bash
docker-compose -f docker-compose.web.yml restart
```

---

## Security Notes

### Self-Hosted (Safe)

- No authentication (trust your network)
- Direct filesystem access
- For personal/internal use

### Public Deployment (Add Auth)

If exposing publicly, add:

1. **HTTP Basic Auth** (Nginx)
2. **API Keys** (Flask middleware)
3. **Rate Limiting** (Flask-Limiter)
4. **File Scanning** (ClamAV)

Example basic auth (Nginx):

```nginx
location / {
    auth_basic "Montage AI";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:5000;
}
```

---

## Further Reading

- [API Documentation](api.md)
- [Architecture](architecture.md)
- [Kubernetes Deployment](../deploy/k3s/README.md)
