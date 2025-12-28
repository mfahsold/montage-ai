# Montage AI — 2025 Tech Vision

Strategische Einordnung der 2024/2025 Trends mit Abgleich zu unabhängigen Quellen und konkreter Integration in die bestehende Codebase.

## Executive Summary
- Fokus bleibt auf **Polish & Kontrolle**, nicht auf Pixel-Halluzination. Agentische Workflows (Editor+Kritiker), AI-gestützte Farbgestaltung (LUT), langes Kontextverständnis und hochwertige Zeitinterpolation sind die relevanten Hebel.
- Priorität: sichtbare Qualitätsgewinne bei vertretbarer Komplexität. Text-to-Video, Streaming und Dubbing bleiben bewusst out-of-scope.

## Trend-Check mit Quellen
- **Multi-Agent / Critic Loops**: Best Practices aus Agentenforschung (z.B. Reflexion-ähnliche Feedback-Loops, OpenAI o1-preview systematische self-critique 2024) und “double-pass” Prompts (Google DeepMind Gemini 1.5 Modelle mit selbstkritischem Pass). -> Validiert: Mehrstufige LLM-Feedbacks liefern robustere Struktur-Outputs als Single-Shot.
- **Neural Color Grading via LUT-Gen**: Veröffentlichungen zu diffusion-basiertem Color Grading (z.B. “Diffusion-based Color Grading”, Li et al. 2023), Industrial Pattern: Resolve/Adobe setzen zunehmend auf LUT-generierende Modelle statt Frame-Repaint, um Flicker zu vermeiden. -> Validiert: LUT-Gen reduziert Halluzination, erhält Geometrie, skaliert für Video.
- **Long-Context Video Understanding**: Modelle wie Gemini 1.5 Pro (1M Token Kontext) und Claude 3.5 Sonnet unterstützen lange Sequenzen; Forschungsprojekte wie MovieChat/MovieLLM (2023/24) zeigen Nutzen von Memory-Mechanismen für Plot-Kohärenz. -> Validiert: Längere Kontexte + episodische Speicher verbessern B-Roll-/Narrativ-Suche.
- **Motion-Aware Interpolation**: Industriereife Open-Source-Ansätze wie FILM (Google Research 2022) und RIFE 4.x (2023/24) liefern artefaktarme Frame-Interpolation; prosumer-Tools nutzen sie für 60fps Slow-Mo. -> Validiert: Bereits verfügbar, GPU-tauglich, klarer Qualitätsgewinn.

## Gap-Analyse (Stand der Codebase)
- **Orchestrierung**: Refactor in Phasen (audio/scene/video metadata, semantic_matcher, analysis_cache) ist im Gang; MontageBuilder noch zu straffen. -> Stabil weiterführen.
- **Creative Director**: Single-Shot JSON-Response (creative_director.py). -> Fehlt Kritiker-Schleife/Score-Feedback.
- **Styling/Color**: Statische JSON-Templates + FFmpeg LUT-Auswahl. -> Kein dynamisches LUT-Gen, kein Referenzbild-Flow.
- **Verständnis**: SemanticMatcher nutzt Embeddings + Caches (Rule 6). -> Kein Long-Context Memory; B-Roll-Suche ist query-basiert, nicht plot-aware.
- **Upscaling/Interpolation**: cgpu_upscaler + FFmpeg fallback; keine Frame-Interpolation. -> Platz für FILM/RIFE-Integration (lokal oder cgpu-job).

## Handlungsfelder & Empfehlungen
1) **Agentic Creative Loop (Editor + Critic)**  
   - Backend: creative_director.py erweitern um Zweipass (Editor erzeugt Instruktionen, Critic bewertet gegen Prompt/Heuristiken, Editor korrigiert).  
   - Metriken: Ziel-/Pacing-Abweichung, Effekt-Budgets, Stil-Treue.  
   - Tests: deterministische Prompts + Gold-JSON, Regression in tests/test_creative_director_agentic.py.

2) **AI Colorist (LUT-Generator)**  
   - Flow: Referenzbild-Upload → kleines Model (lokal oder cgpu) generiert 3D-LUT (.cube) → apply in FFmpeg/MoviePy.  
   - Guardrails: Fallback zu bestehenden LUTs, Caching per Referenzhash, Flicker-free (kein Frame-Repaint).  
   - UI: “Match reference look” Upload + “Apply LUT” Toggle; CLI: `--lut-ref path`.

3) **Episodic Memory für B-Roll/Story**  
   - Ergänzung zu semantic_matcher: Session-Memory (analysis_cache) mit Story-Phase-Labels; optionaler Langkontext-Lauf (LLM mit Shots-Outline statt einzelner Frames).  
   - API: `/api/broll/analyze` erweitert um “phase tags”; Query versteht “build-up before finale”.  
   - Ressourcen: Long-context LLM optional (Gemini 1.5 / Claude 3.5) via cgpu.

4) **Motion-Aware Interpolation**  
   - Integration von FILM/RIFE als cgpu_job (GPU) + CPU-Fallback; UI/CLI-Toggle “Smooth Slow-Mo”.  
   - Placement: eigener Schritt in clip_enhancement pipeline; optional nur auf ausgewählte Shots (high-motion scenes).  
   - Perf: Warnung bei CPU-only, sensible Defaults (2x → 60fps).

5) **UX/Transparency**  
   - Web: Phase-Chips, Kosten-/Zeit-Badges (bereits in Plan 3.1–3.5).  
   - CLI: `--explain` Preflight (Output-Profile, GPU-Encoder, LLM-Backend) + Heartbeat bei Long-Tasks.

## Risiken & Leitplanken
- LLM-Kosten/Latency: Critic-Loops nur für “hq” oder wenn CGPU/OpenAI backend verfügbar; Budget-Kappung.  
- Flicker/Temporal Consistency: LUT-Gen statt Diffusion-Repaint; Color-Ref strikt, kein Frame-Neu-Sampling.  
- Memory: Long-context Requests optional; Fallback auf chunked analysis + cache.  
- User Trust: Immer Fallback-Wege und klare Hinweise (Kosten/Zeiten/Abschaltungen).

## Konkrete nächste Schritte
1. Dokumentation verankern (dieses Dokument) und Referenz im Roadmap-Index ergänzen.  
2. Phase 3 (UI/UX) weiterführen: Backend-Phase-Feld + CLI `--explain` (geringe Komplexität, hohe Klarheit).  
3. Agentic Creative Loop Prototyp: Editor/Critic-Pass in creative_director.py, kleiner Testset.  
4. LUT-Gen Spike: Minimaler Service (lokal oder cgpu) + FFmpeg apply; UI/CLI Toggle.  
5. Interpolation Spike: FILM/RIFE als cgpu_job + CLI/Web Toggle “Smooth Slow-Mo”.
