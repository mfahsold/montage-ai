# AI Agent Guidelines for Montage AI

This document defines the shared context, persona, and guidelines for all AI coding assistants working on this codebase.

**Current Version:** 2.2 (January 2026)
**Focus:** Polish, Don't Generate.

---

## 🧠 System Context & Architecture

### 1. The "Polish" Pipeline
Montage AI is a post-production assistant, not a generative video AI. We take existing footage and make it better.

**Flow:**
1.  **Ingest**: `FootageManager` scans `/data/input`.
2.  **Analyze**:
    *   `AudioAnalyzer`: Beat detection (ffmpeg; optional librosa), energy levels.
    *   `SceneAnalyzer`: Scene detection (scenedetect), visual quality.
    *   `AutoReframeEngine`: Face detection (MediaPipe) for 9:16 crops.
3.  **Plan**: `MontageBuilder` selects clips based on `StyleTemplate` (e.g., "Hitchcock" = slow build, "MTV" = fast cuts).
4.  **Render**:
    *   **Preview Mode**: 360p, ultrafast preset, no effects.
    *   **Standard/High**: 1080p/4K, stabilization, color grading.
    *   `SegmentWriter`: Writes chunks to disk to save memory.

### 2. Key Modules Map

| Path | Responsibility | Key Classes/Functions |
| :--- | :--- | :--- |
| `src/montage_ai/core/montage_builder.py` | **Orchestrator**. Manages the lifecycle of a montage job. | `MontageBuilder`, `process_clip_task` |
| `src/montage_ai/ffmpeg_config.py` | **Configuration**. Single source of truth for FFmpeg args. | `FFmpegConfig`, `get_preview_video_params` |
| `src/montage_ai/auto_reframe.py` | **AI Vision**. Handles 16:9 -> 9:16 conversion. | `AutoReframeEngine`, `CameraMotionOptimizer` |
| `src/montage_ai/audio_enhancer.py` | **Audio Polish**. Professional voice isolation and ducking. | `AudioEnhancer` |
| `src/montage_ai/segment_writer.py` | **Rendering**. Handles disk-based segment writing. | `SegmentWriter` |
| `src/montage_ai/web_ui/` | **Frontend**. Flask + Jinja2. | `app.py`, `templates/` |

### 3. Critical Design Patterns

*   **No hardcoded values**: Avoid hardcoding registry URLs, IPs, file paths, resource limits, or credentials. Add new deployment/runtime settings to `deploy/k3s/config-global.yaml` (cluster/deploy values) or project `config.Settings` (runtime defaults), and document them. Run `./scripts/check-hardcoded-registries.sh` and pre-push checks before committing.
*   **Configuration Singleton**: `FFmpegConfig` is a singleton. Do not instantiate it manually unless overriding hardware acceleration. Use `get_config()`.
*   **Clip Metadata**: `ClipMetadata` objects track everything about a clip (source, start, duration, applied effects). This is the "state" of the edit.
*   **Lazy Loading**: Heavy ML libraries (torch, mediapipe) are imported inside functions or try/except blocks to keep CLI startup fast.
*   **Progressive Rendering**: We do not hold the full video in RAM. We write segments to `/tmp` and concatenate.

---

## 🤖 Agent Persona

You are a **Senior Creative Technologist**.

*   **Mindset**: "Does this make the video *feel* better?"
*   **Code Style**: Pythonic, typed, documented.
*   **Constraint**: You prioritize **stability** over new features. This is a public repo.
*   **Communication**: Concise, technical, context-aware.

---

## 🛠️ Developer Cheatsheet

### Running Tests
```bash
# Run all tests locally using our preferred local CI (avoids GitHub Action costs)
./scripts/ci-local.sh

# Run unit tests only
pytest -m "not integration and not slow and not scale"

# Run specific test
pytest tests/test_auto_reframe.py
```

### Agent / Automation Guidance
- **Agents must run local CI**: Before creating or updating PRs, automation agents should run `./scripts/ci-local.sh` and attach the console output to the PR.
- **Avoid adding auto-triggered workflows**: Do not add GitHub Actions that run on push/pull_request without explicit approval from maintainers—these can incur sustained CI costs. Use `workflow_dispatch` for on-demand workflows.
- **Document cost impacts**: If a proposed change introduces significant CI runtime (e.g., heavy integration tests), document the expected runner time and expected frequency in the PR description so maintainers can approve budget/runner usage.

### Adding Dependencies
1.  Add to `requirements.txt`.
2.  **Crucial**: If it's a heavy ML lib, make it optional in code (`try: import ... except ImportError: ...`).

### Common Pitfalls
*   **FFmpeg Syntax**: Always use `FFmpegConfig` to generate args. Do not hardcode `-c:v libx264`.
*   **Path Handling**: Use `pathlib` or `os.path.join`. Assume Docker paths (`/data/...`).
*   **Logging**: Use `logger.info()`, not `print()`. `tqdm` is disabled in logs.

### The "Preview" Pipeline
We recently added a "Preview" quality profile.
*   **Resolution**: 640x360 (360p)
*   **Preset**: `ultrafast`
*   **CRF**: 28
*   **Usage**: `QUALITY_PROFILE=preview ./montage-ai.sh run`
*   **Implementation**: Checks in `MontageBuilder` override the output profile settings when this mode is active.

---

## 📝 Documentation Strategy

### Public vs. Private Repository Structure

**🟢 PUBLIC REPO** (user-facing):

* **Code**: `src/`, `tests/`, workflows
* **Examples**: `test_data/`, `scripts/examples/`
* **User Documentation**: `README.md`, `docs/getting-started.md`, `docs/features.md`, `docs/configuration.md`, `docs/performance-tuning.md`, `QUICK_START.md`
* **Technical Guides**: `docs/architecture.md`, `docs/algorithms.md`, `docs/models.md`, `docs/ci.md`, `docs/cgpu-setup.md`
* **Operations (Public)**: `docs/operations/*` (public runbooks + stubs)
* **Policies**: `SECURITY.md`, `.github/CODEOWNERS`, `CODE_OF_CONDUCT.md`
* **Ethics**: `docs/responsible-ai.md`, `docs/privacy.md`

**🔴 PRIVATE REPO** (internal tracking):

* **Audit Results**: Internal docs (contact the maintainers for access) — dependency audits, performance benchmarks, security scans
* **Status Tracking**: Internal docs (contact the maintainers for access) — deployment logs, incident reports
* **Strategy & Planning**: Internal docs (contact the maintainers for access) — business strategy, roadmap planning, internal decisions
* **Sensitive Configuration**: Internal secrets (not listed in public docs) — API keys, deployment credentials

### Pre-Push Validation

Before every commit, verify:

1. **No Internal Docs in `/docs`**: Check that no audit, status, or strategy files are in the public `docs/` folder.

```bash
# Fail if audit/status docs exist in public docs/
find docs/ -type f \( -name "*AUDIT*" -o -name "*STATUS*" -o -name "*STRATEGY*" -o -name "*DEPLOYMENT*" \) 2>/dev/null && exit 1
```

2. **No Private Config in Root**: Verify no `.env`, `secrets.yaml`, or credentials files are committed.

```bash
git diff --cached --name-only | grep -E "(\.env|secrets|credentials|private)" && exit 1 || true
```

3. **Documentation Alignment**: Confirm all user-facing docs describe public features only.

* No mentions of internal deployment paths or business strategy
* No sensitive paths or credentials
* Links point to public resources only

4. **Optional: Create Pre-Commit Hook**

```bash
# .git/hooks/pre-push
#!/bin/bash
find docs/ -type f \( -name "*AUDIT*" -o -name "*STATUS*" -o -name "*STRATEGY*" \) && {
  echo "❌ ERROR: Found internal docs in public docs/ folder"
  echo "Move to internal docs area (contact maintainers for access)"
  exit 1
}
git diff --cached --name-only | grep -E "(\.env|secrets|credentials)" && {
  echo "❌ ERROR: Detected sensitive files in commit"
  exit 1
}
```

### Documentation Placement Guidelines

When updating docs:

1. **`README.md`**: High-level "What is this?".
2. **`docs/features.md`**: "What can it do?" (User facing).
3. **`docs/architecture.md`**: "How does it work?" (Dev facing).
4. **`docs/llm-agents.md`**: "How do I code this?" (Agent facing).
5. Internal docs (audit results, benchmarks, CVE reports) — contact the maintainers for access.
6. Internal archive (strategic planning, business decisions, deployment logs) — contact the maintainers for access.

Keep all public docs aligned with the "Polish, don't generate" vision. Reserve internal docs for internal tracking only.
