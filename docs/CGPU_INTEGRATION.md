---
title: "cgpu Integration Plan"
summary: "Integration von cgpu für LLM-basierte kreative Steuerung"
updated: 2025-12-01
---

# cgpu Integration für Montage-AI

## Übersicht

**cgpu** (github.com/RohanAdwankar/cgpu) bietet zwei Hauptfunktionen:
1. **Free Cloud GPU** via Google Colab (für CUDA-Entwicklung)
2. **`cgpu serve`** - OpenAI-kompatibler API-Server, der Google Gemini proxyt

## Wichtig: Was cgpu kann und was nicht

### ✅ Geeignet für Montage-AI:
| Feature | Nutzen |
|---------|--------|
| `cgpu serve` (Gemini API) | Creative Director: NLP → Editing-Parameter |
| Free LLM Inference | Intelligente Clip-Analyse, Story-Arc-Generierung |
| OpenAI-kompatible API | Einfache Integration via `openai` Python-Client |

### ❌ NICHT geeignet (bleibt lokal):
| Feature | Grund |
|---------|-------|
| Real-ESRGAN Upscaling | Benötigt lokale GPU/CPU |
| Video Stabilization | OpenCV/FFmpeg lokal |
| FFmpeg Encoding | Lokaler Prozess |
| Beat Detection | librosa (CPU) |

## Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                        Montage-AI                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │   CLI/API    │───▶│ Creative        │───▶│ cgpu serve    │  │
│  │              │    │ Director        │    │ (Gemini API)  │  │
│  └──────────────┘    └─────────────────┘    └───────┬───────┘  │
│         │                    │                       │          │
│         │                    ▼                       ▼          │
│         │           ┌─────────────────┐     ┌───────────────┐  │
│         │           │ Footage         │     │ Google Gemini │  │
│         │           │ Analyzer        │     │ (Free LLM)    │  │
│         │           └─────────────────┘     └───────────────┘  │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     Editor (editor.py)                    │  │
│  │  - Beat Sync          - Scene Detection                   │  │
│  │  - Style Templates    - Composition                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Local Video Processing (FFmpeg/OpenCV)       │  │
│  │  - Upscaling (Real-ESRGAN)  - Stabilization              │  │
│  │  - Encoding (H.264/H.265)   - Color Grading              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Geplante Änderungen

### 1. Dependencies

```bash
# Neue Abhängigkeit
npm install -g cgpu  # Für cgpu serve

# Python-Client (bereits vorhanden)
pip install openai
```

### 2. Neue Umgebungsvariablen

```bash
# .env
CGPU_ENABLED=true
CGPU_HOST=127.0.0.1
CGPU_PORT=8080
CGPU_MODEL=gemini-2.0-flash
```

### 3. Zu ändernde Dateien

| Datei | Änderung |
|-------|----------|
| `requirements.txt` | `openai>=1.0.0` hinzufügen |
| `src/montage_ai/creative_director.py` | cgpu/Gemini-Backend statt lokales LLM |
| `src/montage_ai/footage_analyzer.py` | LLM-basierte Clip-Beschreibung |
| `docker-compose.yml` | `cgpu serve` als Sidecar-Service |
| `montage-ai.sh` | `cgpu serve` automatisch starten |

### 4. Creative Director Integration

```python
from openai import OpenAI

class CreativeDirector:
    def __init__(self, cgpu_url="http://localhost:8080/v1"):
        self.client = OpenAI(
            base_url=cgpu_url,
            api_key="unused"  # cgpu ignoriert API key
        )
    
    def interpret_prompt(self, user_prompt: str) -> dict:
        """Natural Language → Editing Parameters"""
        response = self.client.responses.create(
            model="gemini-2.0-flash",
            instructions="""You are a video editing AI. Convert the user's 
            creative direction into specific editing parameters.
            Return JSON with: style, energy, cut_frequency, mood""",
            input=user_prompt
        )
        return json.loads(response.output_text)
```

### 5. Docker Compose Erweiterung

```yaml
services:
  cgpu:
    image: node:20-slim
    command: npx cgpu serve --port 8080 --host 0.0.0.0
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  montage:
    build: .
    depends_on:
      - cgpu
    environment:
      - CGPU_URL=http://cgpu:8080/v1
```

## Einschränkungen

1. **cgpu serve benötigt `gemini` CLI** installiert (https://github.com/google-gemini/gemini-cli)
2. **Keine GPU-Beschleunigung für Video** - nur für LLM
3. **Rate Limits** von Google Gemini gelten
4. **Keine Offline-Fähigkeit** - Internet erforderlich für LLM

## Phasen-Plan

### Phase 1 (Priorität):
- [ ] `cgpu serve` für Creative Director (NLP → Parameter)
- [ ] Footage Analyzer mit LLM-Beschreibungen
- [ ] Fallback auf lokale Defaults wenn cgpu nicht verfügbar

### Phase 2 (Später):
- [ ] Story-Arc-Generator mit Gemini
- [ ] Automatische Musik-Mood-Matching
- [ ] Clip-Kategorisierung via Vision (wenn Gemini Vision unterstützt)

## Nicht ändern

- Video-Processing bleibt lokal (FFmpeg, OpenCV)
- Upscaling bleibt Real-ESRGAN (lokal)
- Beat Detection bleibt librosa (lokal)
- Style Templates bleiben Python-Dictionaries

## Referenzen

- cgpu GitHub: https://github.com/RohanAdwankar/cgpu
- Gemini CLI: https://github.com/google-gemini/gemini-cli
- OpenAI Python Client: https://github.com/openai/openai-python
