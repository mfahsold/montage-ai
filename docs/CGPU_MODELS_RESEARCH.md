# CGPU + Google Colab Models Research

## Executive Summary

**Best Option for Montage AI Free Tier Experimentation: `gemini-2.0-flash`**

- ✅ **Pros**: Fast, free tier available, multimodal (text, image, video), optimized for production
- ✅ **Best use cases**: Scene analysis, creative direction, quick transcription
- ⚠️ **Considerations**: Free tier has 20 requests/day limit (recently reduced from 250)

---

## Available Models via cgpu (Colab Integration)

### 1. **Gemini 2.0 Flash** (🌟 Recommended for Montage AI)

**Availability**: Free tier ✅  
**Access**: Via cgpu serve (OpenAI-compatible API)

**Specs**:
- **Speed**: Ultra-fast (~2x faster than 1.5 Pro)
- **Context**: 200K tokens
- **Multimodal**: Text, image, video, audio
- **Output**: Text + image generation
- **Cost**: Free tier (limited), $0.075/M input, $0.30/M output tokens (paid)

**Pros for Montage AI**:
- ✅ Fast creative direction LLM calls
- ✅ Video analysis (analyze scenes, shots, color palettes)
- ✅ Script-to-clip semantic matching
- ✅ Beat detection validation (describe rhythm patterns)
- ✅ Native video understanding (no separate video processing)

**Cons**:
- ❌ Free tier: **20 requests/day** (recently throttled from 250)
- ❌ Not ideal for batch processing full projects
- ❌ API rate limits for experimentation

**Use Case in Montage**:
```python
# Scene analysis: "Describe the visual rhythm and energy of this scene"
# Creative direction: "Suggest 3 creative edit points based on music tempo"
# Color grading suggestions: "Analyze color palette and suggest LUT filters"
```

---

### 2. **Gemini 1.5 Flash** (Budget Alternative)

**Availability**: Free tier ✅  
**Access**: Via cgpu serve or direct API

**Specs**:
- **Speed**: Fast (slightly slower than 2.0 Flash)
- **Context**: 1M tokens (10x larger than 2.0)
- **Multimodal**: Text, image, video
- **Cost**: Free tier (limited), $0.075/M input, $0.30/M output tokens (paid)

**Pros**:
- ✅ Larger context window (good for analyzing full scripts)
- ✅ More mature, stable API
- ✅ Free tier available
- ✅ Better for long-form video analysis

**Cons**:
- ❌ Slower than Gemini 2.0 Flash
- ❌ Same free tier limits (20 requests/day)

**Use Case**:
```python
# Full video script analysis with full context
# Long-form transcription review
# Complex beat pattern detection from audio descriptions
```

---

### 3. **Gemini 1.5 Pro** (Premium, NOT Recommended for Free Tier)

**Availability**: Paid tier only ❌  
**Access**: Via API (not free)

**Specs**:
- **Speed**: Slower but most capable
- **Context**: 2M tokens
- **Cost**: $1.50/M input, $6.00/M output tokens

**Decision**: Skip for now—too expensive for experimentation. Use Flash models instead.

---

### 4. **Gemma Models** (Local Alternative via Colab)

**Availability**: Free tier ✅  
**Access**: Direct Python library (no cgpu needed)

**Variants**:
- `Gemma 2B`: ~2 billion parameters (lightweight, CPU-capable)
- `Gemma 7B`: ~7 billion parameters (needs GPU, better quality)
- `Gemma 2 27B`: Newer, larger (needs significant GPU)

**Pros**:
- ✅ No API quotas (runs locally in Colab)
- ✅ Unlimited requests
- ✅ Open-source, privacy-preserving
- ✅ Good for continuous batch processing

**Cons**:
- ❌ Slower inference than Gemini (runs on free Colab T4 GPU)
- ❌ Less capable for multimodal tasks
- ❌ Lower quality for creative tasks
- ❌ Limited context (~8K tokens for 7B version)

**Use Case**:
```python
# Batch scene descriptions without API limits
# Local beat detection heuristics
# Fallback when Gemini quotas exhausted
```

---

### 5. **Ollama + Llama 3** (DIY Alternative)

**Availability**: Free, self-hosted  
**Access**: Docker container on Colab + local port tunneling

**Models**:
- `llama3:70b` (most capable, needs Colab Pro with A100)
- `llama3:8b` (balanced, works on free Colab T4)
- `mistral:7b` (faster, smaller)

**Pros**:
- ✅ Unlimited requests
- ✅ Privacy (runs locally)
- ✅ No API quotas
- ✅ Customizable fine-tuning

**Cons**:
- ❌ Slow on free Colab (T4 GPU: ~20-50 tokens/sec)
- ❌ No multimodal support (text-only)
- ❌ Complex setup (requires ngrok tunneling)
- ❌ Higher latency than Gemini

**Use Case**:
```python
# Fallback: low-cost local processing
# Not recommended for creative direction (Gemini is better)
# Good for: batch transcription cleanup, scene tagging
```

---

## Recommendation: Hybrid Strategy for Free Tier Experimentation

### Phase 1: Quick Testing (Weeks 1-2)

**Primary**: Gemini 2.0 Flash via cgpu serve
- **Daily quota**: 20 requests/day
- **Budget**: Estimate 5-10 requests per full video edit
- **Cost**: Free

**Tasks**:
1. ✅ Test creative direction prompt (2 requests)
2. ✅ Scene analysis prompts (3 requests)
3. ✅ Color palette suggestions (2 requests)
4. ✅ Beat/rhythm descriptions (3 requests)

**Expected time**: 10-15 requests → covers 1-2 full edits per day

---

### Phase 2: Scale Testing (Weeks 3-4)

**Add**: Gemma 7B locally (fallback for batch)
- **Reasoning**: When Gemini quota exhausted, use Gemma for non-critical tasks
- **Cost**: Free (CPU in Colab)

**Workflow**:
```
High-priority creative tasks → Gemini 2.0 Flash (fast, best quality)
Low-priority tasks → Gemma 7B (unlimited, lower quality)
```

---

### Phase 3: Production (Month 2+)

**Decision tree**:
- **High volume**: Use Gemma 7B as primary (unlimited)
- **Quality matters**: Subscribe to Gemini API ($10-20/month for production tier)
- **Video analysis**: Keep Gemini 2.0 Flash (multimodal beats Gemma)

---

## Technical Integration: cgpu serve

### Setup
```bash
# Already have cgpu installed ✅

# Start Gemini API server
cgpu serve --default-model gemini-2.0-flash --port 8080

# Montage AI uses OpenAI-compatible client
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="cgpu"  # dummy key for local
)
```

### Usage Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="cgpu")

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[{
        "role": "user",
        "content": "Analyze this scene for creative edit points based on music rhythm..."
    }],
    max_tokens=1024
)
```

---

## Pros & Cons Comparison Table

| Model | Speed | Quality | Context | Multimodal | Free Tier | Cost (Paid) | Setup |
|-------|-------|---------|---------|-----------|-----------|------------|-------|
| **Gemini 2.0 Flash** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 200K | ✅ Text/Image/Video | ⚠️ 20/day | $0.075/$0.30 | Easy (cgpu) |
| **Gemini 1.5 Flash** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 1M | ✅ | ⚠️ 20/day | $0.075/$0.30 | Easy (cgpu) |
| **Gemma 7B** | ⭐⭐⭐ | ⭐⭐⭐ | 8K | ❌ | ✅ Unlimited | Free | Moderate |
| **Llama 3 8B** | ⭐⭐⭐ | ⭐⭐⭐ | 8K | ❌ | ✅ Unlimited | Free | Complex |
| **Gemini 1.5 Pro** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 2M | ✅ | ❌ | $1.50/$6.00 | Easy | Not recommended |

---

## Free Tier Limitations & Workarounds

### Problem: 20 Requests/Day Limit on Gemini Free Tier

**Impact**: Can't test full production workflow daily

**Workarounds**:

1. **Batch requests** (Process once per day)
   - Run full edit in morning (using all 20 quota)
   - Validate results rest of day

2. **Fallback to Gemma**
   - Keep Gemma 7B ready for overflow
   - Use Gemini for key creative decisions only

3. **Request upgrade (Optional)**
   - Free tier → "Google AI Studio" account: 0 cost, same limits
   - Paid tier: $5-20/month for 1000+ requests/day
   - Cost-benefit: Not worth for experimentation only

---

## Recommendation for Montage AI

### ✅ Start Here (This Week)

```bash
# 1. Verify cgpu is authenticated
cgpu status

# 2. Start Gemini server
cgpu serve --default-model gemini-2.0-flash --port 8080

# 3. Test with creative director
cd src/montage_ai
python3 -c "
from creative_director import generate_edit_plan
# Modify to use cgpu endpoint instead of Ollama
"
```

### 📊 Expected Results

**Input**: 1 video (2 minutes, 4 scenes)  
**Requests needed**: 5-8 (creative direction, scene analysis, beat description)  
**Time for experimentation**: ~15-20 minutes per edit  
**Daily capacity**: 2-4 full edits per day (on free tier)

---

## Next Steps

1. ✅ Confirm cgpu OAuth setup complete
2. ⏳ Modify `creative_director.py` to support cgpu endpoint (HTTP → OpenAI-compatible)
3. ⏳ Run test edit with Gemini 2.0 Flash backend
4. 📊 Measure latency, quality, quota usage
5. 🎬 Compare results vs. local Ollama or OpenAI API

---

## References

- **cgpu GitHub**: https://github.com/RohanAdwankar/cgpu
- **Gemini API Docs**: https://ai.google.dev/gemini-api/docs/models
- **Gemini Pricing**: https://ai.google.dev/gemini-api/docs/pricing
- **Gemma Models**: https://huggingface.co/collections/google/gemma-release-65d5aefc5c6c28ae281fe882
- **Colab AI Library**: https://medium.com/google-colab/all-colab-users-now-get-access-to-gemini-and-gemma-models-via-colab-python-library-at-no-cost
