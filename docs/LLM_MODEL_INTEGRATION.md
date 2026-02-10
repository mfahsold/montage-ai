# LLM Model Integration: Local vs Cloud Strategy

## Current State (from creative_director.py)

Montage AI already supports multiple LLM backends:

```
Priority Order:
1. OPENAI_API_BASE (LiteLLM/llama-box or OpenAI-compatible)
2. GOOGLE_API_KEY (direct Gemini API)
3. CGPU_ENABLED (cgpu serve for cloud GPU)
4. OLLAMA_HOST (local Ollama fallback)
```

**Current Setup**: Likely using local Ollama or LiteLLM

---

## Three Deployment Options for Montage AI

### Option 1: Local LLM in Cluster (Self-Hosted)

**Technology**: Ollama or vLLM

#### Pros
✅ No API costs  
✅ No rate limits  
✅ Complete data privacy  
✅ Runs on cluster GPU (already have GPU budget)  
✅ OpenAI-compatible API (drop-in for creative_director.py)  

#### Cons
❌ Need to maintain model server  
❌ Requires GPU vRAM (8-24GB depending on model size)  
❌ Slower inference than cloud (but acceptable for batch processing)  

#### Best Models for Montage AI

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **Llama 3.1 8B** | 8B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast creative direction, scene analysis |
| **Mistral 7B** | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Lightweight, default choice |
| **Llama 3.1 70B** | 70B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High quality, needs A100 (expensive) |
| **Qwen 2.5 7B** | 7B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Best coding, good for FFmpeg prompts |

**Recommended for Montage**: **Llama 3.1 8B or Mistral 7B**
- Runs on single RTX 4090 or H100 with 24GB vRAM
- Good balance of speed & quality for creative tasks
- Model size ~7-8GB, fits in typical cluster GPU

#### Implementation

```bash
# Deploy Ollama to Kubernetes
kubectl create deployment ollama --image=ollama/ollama:latest -n montage-ai
kubectl port-forward svc/ollama 11434:11434

# Pull model
ollama pull mistral:7b

# Configure creative_director.py
export OLLAMA_HOST=http://ollama.montage-ai.svc.cluster.local:11434
export LLM_MODEL=mistral:7b
```

---

### Option 2: Cloud LLM API (External Service)

**Technology**: Gemini API, OpenAI, Claude, etc.

#### Comparison

| Provider | Model | Free Tier | Paid Cost | Speed | Quality | Multimodal |
|----------|-------|-----------|-----------|-------|---------|-----------|
| **Google Gemini** | Gemini 2.0 Flash | ⚠️ Limited* | $0.075/$0.30/M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Video |
| **OpenAI** | GPT-4o | ❌ No | $5/$15/M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| **Anthropic** | Claude 3.5 Sonnet | ❌ No | $3/$15/M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| **Groq** | LLaMA 3.1 70B | ⚠️ Limited* | $1/M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |

*Free tier: Limited requests/day (15-20 usually)

#### Pros
✅ No infrastructure management  
✅ Latest models (Gemini 2.0 Flash, Claude 3.5)  
✅ High quality (better than local 7B models)  
✅ Built-in multimodal (video analysis for Gemini)  

#### Cons
❌ Per-token costs (~$5-20/month for production)  
❌ API rate limits  
❌ Data privacy (sent to external service)  
❌ Network dependency  

#### Recommended for Montage

**Primary**: Google Gemini 2.0 Flash
- ✅ Cheapest: $0.075/M input tokens
- ✅ Fastest: 2-3x faster than GPT-4o
- ✅ Video-native: Understands video directly (for scene analysis)
- ⚠️ Free tier: 20 requests/day (enough for 2-4 test edits/day)

**Alternative**: OpenAI GPT-4o
- Quality is marginally better but ~10x more expensive
- Use only if Gemini insufficient

#### Cost Estimate

**Per video edit (2 min, 4-6 scenes):**
- ~6-8 API calls
- ~2,000-5,000 input tokens
- ~500-1,000 output tokens
- **Cost**: ~$0.0005-0.001 per edit (~$1-2 per month for 1-2 edits/day)

#### Implementation

```bash
# Using Gemini API
export GOOGLE_API_KEY="your-key-here"
export LLM_BACKEND=google_ai
export LLM_MODEL=gemini-2.0-flash

# creative_director.py will use GOOGLE_API_KEY automatically
```

---

### Option 3: Hybrid (Recommended for Production)

**Strategy**: Use both local + cloud with intelligent fallback

```
High-priority tasks (need quality) → Gemini API
Low-priority tasks (batch) → Local Ollama
API quota exceeded → Fallback to Ollama
Network issues → Use local only
```

#### Implementation Pattern

```python
class CreativeDirector:
    def __init__(self):
        self.primary = "gemini"      # Cloud (high quality)
        self.fallback = "ollama"     # Local (unlimited, lower quality)
        self.api_quota_remaining = 20  # Track free tier quota
    
    def generate_plan(self, prompt, quality="high"):
        if quality == "high" and self.api_quota_remaining > 0:
            return self.call_gemini(prompt)
        elif self.api_quota_remaining > 0:
            return self.call_gemini(prompt)
        else:
            logger.info("API quota exhausted, using Ollama")
            return self.call_ollama(prompt)
```

---

## Detailed Comparison for Montage AI

### Use Case: Creative Direction LLM

**Task**: "Analyze scene 2 (0:30-1:15) and suggest edit points for house music beat"

#### Option 1: Local Ollama (Mistral 7B)
```
Latency: 200-500ms
Cost: $0 (running 24/7)
Quality: Good (7/10 for creative)
Example output: "3 main beats detected at 0:35, 0:45, 1:00. Suggest quick cuts or audio sync."
```

#### Option 2: Cloud Gemini 2.0 Flash
```
Latency: 1-2 seconds (API round-trip)
Cost: ~$0.0001-0.0005 per call
Quality: Excellent (9/10 for creative)
Example output: "Dynamic rhythm with buildup at 0:40. Cut synchronized to hi-hat at 0:45. 
                 Climax cut suggestion at 0:58. Video energy peaks at 1:00-1:05."
```

#### Option 3: Hybrid
```
First 2 edits/day: Gemini (high quality for prototyping)
Remaining edits: Ollama (free, acceptable quality for iteration)
```

---

## Recommendation for Montage AI

### Phase 1 (Now - Week 1)
✅ **Keep current Ollama setup** for immediate testing
- Already working in cluster
- No API setup needed
- Free unlimited processing

### Phase 2 (Week 2-3)
⏳ **Add Gemini API** for quality experiments
- Set `GOOGLE_API_KEY` environment variable
- creative_director.py supports it natively
- Test high-quality creative direction
- Cost: ~$1-5/month for light usage

### Phase 3 (Production)
📊 **Implement hybrid strategy**
- Primary: Gemini 2.0 Flash (quality tasks)
- Fallback: Local Ollama (batch, quota overflow)
- Monitor cost vs. quality tradeoff

---

## Action Items

### Step 1: Verify Current Setup
```bash
# Check what's currently configured
kubectl -n montage-ai get configmap | grep llm
kubectl -n montage-ai logs deployment/creative-director | grep -i "llm\|model"
```

### Step 2: Option A - Keep Ollama (No Changes)
- Status: Already running ✅
- No additional config needed
- Accept current quality/speed tradeoff

### Step 3: Option B - Add Gemini API
```bash
# 1. Get API key from Google AI Studio
# https://aistudio.google.com/app/apikey

# 2. Add to cluster
kubectl -n montage-ai create secret generic gemini-api-key \
  --from-literal=GOOGLE_API_KEY="your-key-here"

# 3. Update deployment
kubectl -n montage-ai set env deployment/editor \
  -e GOOGLE_API_KEY="your-key-here" \
  -e LLM_BACKEND=google_ai

# 4. Test
./montage-ai.sh run dynamic --style hitchcock
```

### Step 4: Monitor Performance
```bash
# Measure latency & quality
# creative_director.py logs LLM response times
kubectl -n montage-ai logs deployment/editor | grep "LLM response time"
```

---

## Final Recommendation

**For Montage AI in February 2026:**

| Scenario | Recommendation |
|----------|-----------------|
| **Quick testing** | Keep Ollama (working, free, local) |
| **Quality experimentation** | Add Gemini API ($1-5/month) |
| **Production** | Hybrid: Gemini primary + Ollama fallback |
| **Very low budget** | Local Ollama only (acceptable for non-critical) |
| **Premium quality** | GPT-4o or Claude (expensive, not justified) |

**Go with: Hybrid approach** (Gemini + Ollama)
- Gemini for key creative decisions (natural language quality matters)
- Ollama for batch processing, fallback, unlimited capacity
- Minimal cost (~$2-10/month)
- Best reliability (no single point of failure)

---

## References

- **Ollama**: https://ollama.ai/
- **vLLM**: https://github.com/vllm-project/vllm
- **Gemini API Docs**: https://ai.google.dev/
- **LLM Pricing Comparison**: https://intuitionlabs.ai/articles/llm-api-pricing-comparison-2025
- **Montage AI creative_director.py**: Already supports all these backends!
