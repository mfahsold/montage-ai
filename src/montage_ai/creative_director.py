"""
Creative Director: LLM-powered natural language video editing

Translates natural language prompts into structured editing instructions
using local LLMs (Llama 3.1, DeepSeek-R1) via Ollama, or Google Gemini via cgpu.

Architecture (2024/2025 industry standard):
  User Prompt → LLM → JSON Instructions → Validation → Editing Engine

Backends:
  - Ollama (local): OLLAMA_HOST environment variable
  - cgpu/Gemini (cloud): CGPU_ENABLED=true, requires cgpu serve running

Based on research:
- Descript Underlord: Conversational editing interface
- DirectorLLM: Llama-based cinematography orchestration
- LAVE: Structured JSON output for video editing
"""

import os
import json
import requests
from typing import Dict, Optional, Any
from .style_templates import get_style_template, list_available_styles

# Try importing OpenAI client for cgpu/Gemini support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

VERSION = "0.3.0"

# Backend configuration
# Priority: OPENAI_API_BASE (KubeAI/OpenAI-compatible) > GOOGLE_API_KEY (direct Gemini) > CGPU_ENABLED (cgpu serve) > Ollama (local)

# OpenAI-compatible API (KubeAI, vLLM, LocalAI, etc.)
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")  # e.g. http://kubeai.kubeai-system.svc.cluster.local/openai/v1
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")  # KubeAI ignores this
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")  # e.g. gemma3-4b, qwen2-5-32b

# Google AI (direct API)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_AI_MODEL = os.environ.get("GOOGLE_AI_MODEL", "gemini-2.0-flash")
GOOGLE_AI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# cgpu serve (Gemini via OpenAI Responses API)
CGPU_ENABLED = os.environ.get("CGPU_ENABLED", "false").lower() == "true"
CGPU_HOST = os.environ.get("CGPU_HOST", "127.0.0.1")
CGPU_PORT = os.environ.get("CGPU_PORT", "8090")  # Updated default port to match montage-ai.sh
CGPU_MODEL = os.environ.get("CGPU_MODEL", "gemini-2.0-flash")

# Ollama (local fallback)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("DIRECTOR_MODEL", "llama3.1:70b")  # or deepseek-r1:70b

# System prompt for the Creative Director LLM (style list is injected at runtime)
DIRECTOR_SYSTEM_PROMPT = """You are {persona}.

Your role: Translate natural language editing requests into structured JSON editing instructions.

Available cinematic styles:
{styles_list}

You MUST respond with ONLY valid JSON matching this structure:
{{
  "style": {{
    "name": {style_name_options},
    "mood": "suspenseful" | "playful" | "energetic" | "calm" | "dramatic" | "mysterious",
    "description": "Custom style description (only if name=custom)"
  }},
  "pacing": {{
    "speed": "very_slow" | "slow" | "medium" | "fast" | "very_fast" | "dynamic",
    "variation": "minimal" | "moderate" | "high" | "fibonacci",
    "intro_duration_beats": 4-32,
    "climax_intensity": 0.0-1.0
  }},
  "cinematography": {{
    "prefer_wide_shots": true | false,
    "prefer_high_action": true | false,
    "match_cuts_enabled": true | false,
    "invisible_cuts_enabled": true | false,
    "shot_variation_priority": "low" | "medium" | "high"
  }},
  "transitions": {{
    "type": "hard_cuts" | "crossfade" | "mixed" | "energy_aware",
    "crossfade_duration_sec": 0.1-2.0
  }},
  "energy_mapping": {{
    "sync_to_beats": true | false,
    "energy_amplification": 0.5-2.0
  }},
  "effects": {{
    "color_grading": "none" | "neutral" | "warm" | "cool" | "high_contrast" | "desaturated" | "vibrant",
    "stabilization": true | false,
    "upscale": true | false,
    "sharpness_boost": true | false
  }},
  "constraints": {{
    "target_duration_sec": null | number,
    "min_clip_duration_sec": 0.5-10.0,
    "max_clip_duration_sec": 2.0-60.0
  }}
}}

Examples:

User: "Edit this like a Hitchcock thriller"
Response:
{{
  "style": {{"name": "hitchcock", "mood": "suspenseful"}},
  "pacing": {{"speed": "dynamic", "variation": "high", "intro_duration_beats": 16, "climax_intensity": 0.9}},
  "cinematography": {{"prefer_wide_shots": false, "prefer_high_action": true, "match_cuts_enabled": true, "invisible_cuts_enabled": true, "shot_variation_priority": "high"}},
  "transitions": {{"type": "hard_cuts", "crossfade_duration_sec": 0.3}},
  "effects": {{"color_grading": "high_contrast", "stabilization": false, "sharpness_boost": true}}
}}

User: "Make it calm and meditative with long shots"
Response:
{{
  "style": {{"name": "minimalist", "mood": "calm"}},
  "pacing": {{"speed": "very_slow", "variation": "minimal", "intro_duration_beats": 32, "climax_intensity": 0.3}},
  "cinematography": {{"prefer_wide_shots": true, "prefer_high_action": false, "match_cuts_enabled": true, "invisible_cuts_enabled": true, "shot_variation_priority": "low"}},
  "transitions": {{"type": "crossfade", "crossfade_duration_sec": 2.0}},
  "effects": {{"color_grading": "desaturated", "stabilization": true, "sharpness_boost": false}},
  "constraints": {{"min_clip_duration_sec": 4.0, "max_clip_duration_sec": 60.0}}
}}

User: "Fast-paced music video style"
Response:
{{
  "style": {{"name": "mtv", "mood": "energetic"}},
  "pacing": {{"speed": "very_fast", "variation": "high", "intro_duration_beats": 2, "climax_intensity": 1.0}},
  "cinematography": {{"prefer_wide_shots": false, "prefer_high_action": true, "match_cuts_enabled": false, "invisible_cuts_enabled": true, "shot_variation_priority": "high"}},
  "transitions": {{"type": "hard_cuts"}},
  "effects": {{"color_grading": "vibrant", "sharpness_boost": true}},
  "energy_mapping": {{"energy_amplification": 1.5}}
}}

CRITICAL RULES:
1. Return ONLY valid JSON - no markdown, no explanations
2. Use predefined styles when possible (hitchcock, mtv, etc.)
3. For unknown requests, use "custom" style and describe intent
4. Always include "style" and "pacing" (required fields)
5. Be conservative with effects (stabilization/upscale are slow!)
6. Match the user's creative intent while staying technically feasible

Think like a professional film editor who understands both art and constraints.
"""


class CreativeDirector:
    """
    LLM-powered Creative Director for video editing.

    Translates natural language prompts into structured editing instructions.
    Supports multiple backends:
      - OpenAI-compatible API (KubeAI, vLLM, LocalAI) - use OPENAI_API_BASE
      - Google AI (direct API with GOOGLE_API_KEY)
      - cgpu serve (OpenAI-compatible proxy for Gemini)
      - Ollama (local LLM fallback)
    """

    def __init__(
        self,
        ollama_host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        timeout: int = 60,
        use_cgpu: bool = None,
        use_google_ai: bool = None,
        use_openai_api: bool = None,
        persona: str = "the Creative Director for the Fluxibri video editing system"
    ):
        """
        Initialize Creative Director.

        Args:
            ollama_host: Ollama API endpoint
            model: LLM model to use (llama3.1:70b, deepseek-r1:70b, etc.)
            timeout: Request timeout in seconds
            use_cgpu: Force cgpu backend (None = auto-detect from env)
            use_google_ai: Force Google AI backend (None = auto-detect from env)
            use_openai_api: Force OpenAI-compatible backend (None = auto-detect from env)
            persona: The persona/role description for the LLM
        """
        self.ollama_host = ollama_host
        self.ollama_model = model
        self.timeout = timeout
        
        # Determine backend priority: OpenAI-compatible > Google AI > cgpu > Ollama
        if use_openai_api is None:
            self.use_openai_api = bool(OPENAI_API_BASE and OPENAI_MODEL)
        else:
            self.use_openai_api = use_openai_api
            
        if use_google_ai is None:
            self.use_google_ai = bool(GOOGLE_API_KEY) and not self.use_openai_api
        else:
            self.use_google_ai = use_google_ai and not self.use_openai_api
            
        if use_cgpu is None:
            self.use_cgpu = CGPU_ENABLED and OPENAI_AVAILABLE and not self.use_google_ai and not self.use_openai_api
        else:
            self.use_cgpu = use_cgpu and OPENAI_AVAILABLE and not self.use_google_ai and not self.use_openai_api
        
        # Log backend selection
        if self.use_openai_api:
            print(f"   🌐 Creative Director using OpenAI-compatible API ({OPENAI_MODEL} @ {OPENAI_API_BASE})")
        elif self.use_google_ai:
            print(f"   🌐 Creative Director using Google AI ({GOOGLE_AI_MODEL})")
        elif self.use_cgpu:
            # cgpu serve exposes OpenAI-compatible API at root, not /v1
            cgpu_url = f"http://{CGPU_HOST}:{CGPU_PORT}"
            print(f"   🌐 Creative Director using cgpu/Gemini at {cgpu_url}")
        else:
            print(f"   🏠 Creative Director using Ollama ({self.ollama_model})")
        
        # Initialize OpenAI client for OpenAI-compatible or cgpu backend
        self.openai_client = None
        if self.use_openai_api and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(
                    base_url=OPENAI_API_BASE,
                    api_key=OPENAI_API_KEY
                )
            except Exception as e:
                print(f"   ⚠️ Failed to initialize OpenAI client: {e}")
                self.use_openai_api = False
        
        # Initialize cgpu client if enabled
        self.cgpu_client = None
        if self.use_cgpu:
            # cgpu serve exposes OpenAI-compatible API at /v1
            cgpu_url = f"http://{CGPU_HOST}:{CGPU_PORT}/v1"
            try:
                self.cgpu_client = OpenAI(
                    base_url=cgpu_url,
                    api_key="unused"  # cgpu ignores API key
                )
            except Exception as e:
                print(f"   ⚠️ Failed to initialize cgpu client: {e}")
                self.use_cgpu = False

        # Build system prompt with available styles from presets
        available_styles = list_available_styles()
        styles_list = "\n".join(
            [f"- {name}: {get_style_template(name)['description']}" for name in available_styles]
        )
        style_name_options = " | ".join([f'\"{name}\"' for name in available_styles] + ['"custom"'])
        self.system_prompt = DIRECTOR_SYSTEM_PROMPT.format(
            persona=persona,
            styles_list=styles_list,
            style_name_options=style_name_options,
        )

    def interpret_prompt(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Interpret natural language prompt and return editing instructions.

        Args:
            user_prompt: Natural language editing request

        Returns:
            Dictionary of editing instructions (validated JSON schema)
            None if LLM fails or returns invalid JSON

        Example:
            >>> director = CreativeDirector()
            >>> instructions = director.interpret_prompt("Edit like Hitchcock")
            >>> print(instructions['style']['name'])
            'hitchcock'
        """
        print(f"🎬 Creative Director analyzing: '{user_prompt}'")

        # Check if prompt directly references a template
        user_lower = user_prompt.lower()
        for style_name in list_available_styles():
            if style_name in user_lower or style_name.replace('_', ' ') in user_lower:
                print(f"   🎯 Detected style template: {style_name}")
                template = get_style_template(style_name)
                return template['params']

        # Extended keyword matching (no LLM needed)
        style_keywords = {
            'hitchcock': ['hitchcock', 'suspense', 'thriller', 'tension', 'mystery'],
            'action': ['action', 'blockbuster', 'explosive', 'michael bay', 'fast cuts', 'adrenaline'],
            'mtv': ['mtv', 'music video', 'fast-paced', 'fast paced', 'energetic', 'rapid'],
            'documentary': ['documentary', 'doc', 'realism', 'natural', 'observational', 'vérité', 'verite'],
            'minimalist': ['minimalist', 'art film', 'meditative', 'calm', 'contemplative', 'long takes', 'slow'],
            'wes_anderson': ['wes anderson', 'whimsical', 'symmetry', 'symmetrical', 'pastel', 'quirky'],
        }
        
        # Also match mood-based keywords
        mood_to_style = {
            'cinematic': 'hitchcock',  # Cinematic masterpiece → Hitchcock (elegant, professional)
            'elegant': 'hitchcock',
            'masterpiece': 'hitchcock',
            'professional': 'documentary',
            'smooth': 'documentary',
            'gallery': 'minimalist',
            'beautiful': 'wes_anderson',
            'artistic': 'minimalist',
        }
        
        # Check extended keywords
        for style_name, keywords in style_keywords.items():
            for keyword in keywords:
                if keyword in user_lower:
                    print(f"   🎯 Keyword match '{keyword}' → {style_name}")
                    template = get_style_template(style_name)
                    return template['params']
        
        # Check mood keywords
        for mood_keyword, style_name in mood_to_style.items():
            if mood_keyword in user_lower:
                print(f"   🎯 Mood match '{mood_keyword}' → {style_name}")
                template = get_style_template(style_name)
                return template['params']

        # Otherwise, query LLM for creative interpretation
        try:
            response = self._query_llm(user_prompt)
            if not response:
                print("   ❌ LLM returned empty response")
                return None

            # Parse and validate JSON
            instructions = self._parse_and_validate(response)
            if instructions:
                print(f"   ✅ Generated editing instructions (style: {instructions['style']['name']})")
                return instructions
            else:
                print("   ⚠️ LLM response failed validation")
                return None

        except Exception as e:
            print(f"   ❌ Creative Director error: {e}")
            return None

    def _query_llm(self, user_prompt: str) -> Optional[str]:
        """
        Query LLM with user prompt (OpenAI-compatible, Google AI, cgpu, or Ollama).

        Args:
            user_prompt: User's natural language request

        Returns:
            LLM response text (should be JSON)
        """
        # Try backends in priority order: OpenAI-compatible > Google AI > cgpu > Ollama
        if self.use_openai_api and self.openai_client:
            return self._query_openai_api(user_prompt)
        elif self.use_google_ai:
            return self._query_google_ai(user_prompt)
        elif self.use_cgpu and self.cgpu_client:
            return self._query_cgpu(user_prompt)
        else:
            return self._query_ollama(user_prompt)

    def _query_openai_api(self, user_prompt: str) -> Optional[str]:
        """
        Query OpenAI-compatible API (KubeAI, vLLM, LocalAI, etc.).
        
        Uses standard /v1/chat/completions endpoint.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                # Clean up response - some models wrap JSON in markdown
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                return content.strip()
            else:
                print("   ⚠️ OpenAI API returned empty response")
                return None
                
        except Exception as e:
            error_str = str(e)
            # Check if it's a model-not-found or similar error
            if "response_format" in error_str.lower() or "json" in error_str.lower():
                # Model doesn't support JSON mode, retry without it
                print("   ⚠️ Model doesn't support JSON mode, retrying without...")
                return self._query_openai_api_no_json_mode(user_prompt)
            print(f"   ⚠️ OpenAI API error: {e}")
            # Fallback to Ollama
            print("   🔄 Falling back to Ollama...")
            return self._query_ollama(user_prompt)

    def _query_openai_api_no_json_mode(self, user_prompt: str) -> Optional[str]:
        """
        Query OpenAI-compatible API without JSON mode (for models that don't support it).
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                # Clean up response
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                return content.strip()
            else:
                return None
        except Exception as e:
            print(f"   ⚠️ OpenAI API error (no JSON mode): {e}")
            return self._query_ollama(user_prompt)

    def _query_google_ai(self, user_prompt: str) -> Optional[str]:
        """
        Query Google AI directly using API Key (no cgpu/gemini-cli).
        
        Uses the generativelanguage.googleapis.com REST API.
        This bypasses cgpu serve and gemini-cli entirely.
        """
        try:
            url = f"{GOOGLE_AI_ENDPOINT}/{GOOGLE_AI_MODEL}:generateContent"
            
            # Build request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{self.system_prompt}\n\nUser request: {user_prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topP": 0.9,
                    "maxOutputTokens": 1024,
                    "responseMimeType": "application/json"  # Force JSON output
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": GOOGLE_API_KEY
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract text from Gemini response structure
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        # Clean up response - Gemini sometimes wraps JSON in markdown
                        if text.startswith("```json"):
                            text = text[7:]
                        if text.startswith("```"):
                            text = text[3:]
                        if text.endswith("```"):
                            text = text[:-3]
                        return text.strip()
                print("   ⚠️ Google AI returned empty response")
                return None
            else:
                error_msg = response.json().get("error", {}).get("message", response.text)
                print(f"   ⚠️ Google AI error ({response.status_code}): {error_msg}")
                # Fallback to Ollama
                print("   🔄 Falling back to Ollama...")
                return self._query_ollama(user_prompt)
                
        except requests.exceptions.Timeout:
            print(f"   ⚠️ Google AI request timeout ({self.timeout}s)")
            return self._query_ollama(user_prompt)
        except Exception as e:
            print(f"   ⚠️ Google AI error: {e}")
            # Fallback to Ollama if Google AI fails
            print("   🔄 Falling back to Ollama...")
            return self._query_ollama(user_prompt)

    def _query_cgpu(self, user_prompt: str) -> Optional[str]:
        """
        Query cgpu/Gemini for creative direction.
        
        Uses OpenAI Chat Completions API provided by `cgpu serve`.
        """
        try:
            # cgpu serve (Gemini) via OpenAI Chat Completions API
            response = self.cgpu_client.chat.completions.create(
                model=CGPU_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
                # response_format={"type": "json_object"} # Gemini via cgpu might not support this param yet
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                # Clean up response - Gemini sometimes wraps JSON in markdown
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                return content.strip()
            else:
                print("   ⚠️ cgpu/Gemini returned empty response")
                return None
                
        except Exception as e:
            print(f"   ⚠️ cgpu/Gemini error: {e}")
            # Fallback to Ollama if cgpu fails
            print("   🔄 Falling back to Ollama...")
            return self._query_ollama(user_prompt)

    def _query_ollama(self, user_prompt: str) -> Optional[str]:
        """
        Query Ollama LLM with user prompt (local fallback).
        """
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": user_prompt,
                "system": self.system_prompt,
                "stream": False,
                "format": "json",  # Force JSON output (Ollama native feature)
                "options": {
                    "temperature": 0.3,  # Low temp for consistent, reliable output
                    "top_p": 0.9,
                    "num_predict": 1024  # Max tokens for JSON response
                }
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"   ⚠️ Ollama API error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print(f"   ⚠️ Ollama request timeout ({self.timeout}s)")
            return None
        except requests.exceptions.ConnectionError:
            print(f"   ⚠️ Cannot connect to Ollama at {self.ollama_host}")
            print(f"      Make sure Ollama is running: ollama serve")
            return None
        except Exception as e:
            print(f"   ❌ LLM query error: {e}")
            return None

    def _parse_and_validate(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse and validate LLM JSON response.

        Args:
            llm_response: Raw text from LLM (should be JSON)

        Returns:
            Validated editing instructions dict, or None if invalid
        """
        try:
            # Parse JSON
            instructions = json.loads(llm_response)

            # Basic validation: required fields
            if "style" not in instructions or "pacing" not in instructions:
                print(f"   ⚠️ Missing required fields (style/pacing)")
                return None

            # Validate style name
            style_name = instructions["style"].get("name")
            valid_styles = list_available_styles() + ["custom"]
            if style_name not in valid_styles:
                print(f"   ⚠️ Invalid style name: {style_name}")
                return None

            # TODO: Full JSON schema validation (jsonschema library)
            # For now, basic checks are sufficient

            return instructions

        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON from LLM: {e}")
            print(f"      Response: {llm_response[:200]}...")
            return None
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return None

    def get_default_instructions(self) -> Dict[str, Any]:
        """
        Get safe default editing instructions (elegant cinematic style).

        Returns:
            Default editing parameters - balanced, professional look
        """
        return {
            "style": {
                "name": "cinematic",  # Balanced default for gallery/professional content
                "mood": "elegant"
            },
            "pacing": {
                "speed": "dynamic",  # Position-aware (intro → build → climax → outro)
                "variation": "moderate",
                "intro_duration_beats": 8,
                "climax_intensity": 0.7
            },
            "cinematography": {
                "prefer_wide_shots": True,  # More elegant/cinematic
                "prefer_high_action": False,  # Balanced
                "match_cuts_enabled": True,
                "invisible_cuts_enabled": True,
                "shot_variation_priority": "medium"
            },
            "transitions": {
                "type": "energy_aware",  # Smart crossfade on low energy
                "crossfade_duration_sec": 0.5
            },
            "energy_mapping": {
                "sync_to_beats": True,
                "energy_amplification": 1.0
            },
            "effects": {
                "color_grading": "neutral",
                "stabilization": False,
                "upscale": False,
                "sharpness_boost": True
            }
        }


# Convenience function for direct usage
def interpret_natural_language(prompt: str) -> Dict[str, Any]:
    """
    Convenience function: Interpret natural language prompt.

    Args:
        prompt: Natural language editing request

    Returns:
        Editing instructions (falls back to defaults if LLM fails)

    Example:
        >>> instructions = interpret_natural_language("Make it suspenseful")
        >>> apply_editing_instructions(instructions)
    """
    director = CreativeDirector()
    instructions = director.interpret_prompt(prompt)

    if instructions:
        return instructions
    else:
        print("   ℹ️ Falling back to default editing style")
        return director.get_default_instructions()


if __name__ == "__main__":
    # Test suite
    print(f"🎬 Creative Director v{VERSION}")
    print(f"   Model: {OLLAMA_MODEL}")
    print(f"   Host: {OLLAMA_HOST}\n")

    test_prompts = [
        "Edit this like a Hitchcock thriller",
        "Make it calm and meditative",
        "Fast-paced MTV style",
        "Documentary realism",
        "Create suspense with long takes then explosive action"
    ]

    director = CreativeDirector()

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}")

        result = director.interpret_prompt(prompt)

        if result:
            print(json.dumps(result, indent=2))
        else:
            print("❌ Failed to generate instructions")
