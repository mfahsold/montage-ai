"""Integration tests for Creative Director LLM logic with mocks (issue #127).

Verifies that:
1. A realistic LLM JSON response is correctly parsed by _parse_and_validate
2. The parsed instructions contain the expected structure (DirectorOutput schema)
3. interpret_prompt correctly dispatches to LLM when no template matches
4. Backend fallback order is respected
5. Malformed / truncated LLM responses are handled gracefully
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from montage_ai.creative_director import CreativeDirector
from montage_ai.prompts import DirectorOutput, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Realistic LLM response fixture (complex, production-like)
# ---------------------------------------------------------------------------

REALISTIC_LLM_RESPONSE = json.dumps({
    "schema_version": SCHEMA_VERSION,
    "director_commentary": (
        "The user wants a suspenseful Hitchcock-style edit. "
        "I will use slow pacing with high climax intensity, "
        "crossfade transitions, and desaturated color grading."
    ),
    "style": {
        "name": "hitchcock",
        "mood": "suspenseful",
        "description": "Suspenseful thriller with methodical pacing"
    },
    "story_arc": {
        "type": "fichtean_curve",
        "tension_target": 0.7,
        "climax_position": 0.85
    },
    "pacing": {
        "speed": "slow",
        "variation": "moderate",
        "intro_duration_beats": 32,
        "climax_intensity": 0.9,
        "breathing_offset_ms": 60,
        "micro_pacing_jitter": 0.03
    },
    "cinematography": {
        "prefer_wide_shots": True,
        "prefer_high_action": False,
        "match_cuts_enabled": True,
        "invisible_cuts_enabled": True,
        "shot_variation_priority": "low",
        "continuity_weight": 0.6,
        "kuleshov_weight": 0.3,
        "variety_weight": 0.1,
        "contrast_weight": 0.4,
        "symmetry_weight": 0.05
    },
    "transitions": {
        "type": "crossfade",
        "crossfade_duration_sec": 1.2
    },
    "energy_mapping": {
        "sync_to_beats": True,
        "energy_amplification": 0.8
    },
    "effects": {
        "color_grading": "desaturated",
        "stabilization": True,
        "upscale": False,
        "sharpness_boost": False
    },
    "constraints": {
        "target_duration_sec": 60.0,
        "min_clip_duration_sec": 2.0,
        "max_clip_duration_sec": 15.0
    }
})


@pytest.fixture
def director(monkeypatch):
    """Create a CreativeDirector with all LLM backends disabled."""
    monkeypatch.setenv("TEST_NO_LLM", "0")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CGPU_ENABLED", raising=False)
    with patch.object(CreativeDirector, '__init__', lambda self, **kw: None):
        cd = CreativeDirector.__new__(CreativeDirector)
        # Manually init minimal state needed for parse/validate
        cd.use_openai_api = False
        cd.use_cgpu = False
        cd.use_google_ai = False
        cd.openai_client = None
        cd.cgpu_client = None
        from montage_ai.config import get_settings
        cd.llm_config = get_settings().llm
        cd.ollama_host = "http://localhost:11434"
        cd.ollama_model = "llama3.1:8b"
        cd.timeout = 30
        cd._resolved_openai_model = None
        cd._resolved_openai_vision_model = None
        cd.system_prompt = "Test system prompt"
        cd._base_system_prompt = "Test system prompt"
        from montage_ai.regisseur_memory import get_regisseur_memory
        cd._regisseur_memory = get_regisseur_memory()
    return cd


class TestParseAndValidate:
    """Test that _parse_and_validate correctly handles various LLM outputs."""

    def test_realistic_response_parses_correctly(self, director):
        """A well-formed, complex LLM JSON response should parse without errors."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result is not None
        assert isinstance(result, dict)

    def test_parsed_style_matches(self, director):
        """The style config from the LLM response should be preserved."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result["style"]["name"] == "hitchcock"
        assert result["style"]["mood"] == "suspenseful"

    def test_parsed_pacing_values(self, director):
        """Pacing parameters should match the LLM's suggested values."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result["pacing"]["speed"] == "slow"
        assert result["pacing"]["climax_intensity"] == 0.9
        assert result["pacing"]["intro_duration_beats"] == 32

    def test_parsed_cinematography(self, director):
        """Cinematography settings should be preserved."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result["cinematography"]["prefer_wide_shots"] is True
        assert result["cinematography"]["invisible_cuts_enabled"] is True
        assert result["cinematography"]["continuity_weight"] == 0.6

    def test_parsed_effects(self, director):
        """Effects should match the LLM's suggestions."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result["effects"]["color_grading"] == "desaturated"
        assert result["effects"]["stabilization"] is True
        assert result["effects"]["upscale"] is False

    def test_parsed_constraints(self, director):
        """Constraints should be correctly parsed."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result["constraints"]["target_duration_sec"] == 60.0
        assert result["constraints"]["min_clip_duration_sec"] == 2.0

    def test_schema_version_present(self, director):
        """Schema version should be set."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert result["schema_version"] == SCHEMA_VERSION

    def test_director_commentary_present(self, director):
        """Director commentary (CoT) should be preserved."""
        result = director._parse_and_validate(REALISTIC_LLM_RESPONSE)
        assert "Hitchcock" in result["director_commentary"]


class TestMalformedResponses:
    """Test graceful handling of broken LLM responses."""

    def test_markdown_wrapped_json(self, director):
        """JSON wrapped in markdown code blocks should still parse."""
        wrapped = f"```json\n{REALISTIC_LLM_RESPONSE}\n```"
        result = director._parse_and_validate(wrapped)
        assert result is not None
        assert result["style"]["name"] == "hitchcock"

    def test_truncated_json_returns_defaults(self, director):
        """Truncated JSON should fall back to safe defaults."""
        truncated = REALISTIC_LLM_RESPONSE[:100]  # Cut off mid-JSON
        result = director._parse_and_validate(truncated)
        # Should return defaults rather than None
        assert result is not None
        assert "style" in result

    def test_empty_response_returns_defaults(self, director):
        """Empty response should return defaults."""
        result = director._parse_and_validate("")
        assert result is not None

    def test_plain_text_returns_defaults(self, director):
        """Non-JSON text should return defaults."""
        result = director._parse_and_validate("I think you should use a dynamic style.")
        assert result is not None


class TestInterpretPromptWithMockedLLM:
    """Test interpret_prompt with mocked LLM backend."""

    def test_known_style_skips_llm(self, director):
        """A prompt matching a known style template should not call the LLM."""
        with patch.object(director, '_query_llm') as mock_query:
            result = director.interpret_prompt("Edit like Hitchcock")
            mock_query.assert_not_called()
            assert result is not None
            assert result["style"]["name"] == "hitchcock"

    def test_custom_prompt_calls_llm(self, director):
        """A prompt that doesn't match a template should call the LLM."""
        with patch.object(director, '_query_llm', return_value=REALISTIC_LLM_RESPONSE):
            result = director.interpret_prompt("Make it feel like a rainy afternoon in Paris")
            assert result is not None
            assert result["style"]["name"] == "hitchcock"  # from our mock

    def test_llm_failure_returns_none(self, director):
        """If the LLM fails, interpret_prompt returns None."""
        with patch.object(director, '_query_llm', return_value=None):
            result = director.interpret_prompt("Something completely unique and unexpected")
            assert result is None


class TestQueryBackendFallback:
    """Test the backend fallback logic in query()."""

    def test_openai_success_skips_ollama(self, director):
        """If OpenAI succeeds, Ollama should not be called."""
        director.use_openai_api = True
        director.openai_client = MagicMock()
        director._resolved_openai_model = "test-model"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'
        director.openai_client.chat.completions.create.return_value = mock_response

        with patch.object(director, '_query_ollama_unified') as mock_ollama:
            result = director.query("Test prompt")
            assert result is not None
            mock_ollama.assert_not_called()

    def test_openai_failure_falls_back_to_ollama(self, director):
        """If OpenAI fails, should fall back to Ollama."""
        director.use_openai_api = True
        director.openai_client = MagicMock()
        director.openai_client.chat.completions.create.side_effect = Exception("API down")

        with patch.object(director, '_query_ollama_unified', return_value='{"fallback": true}') as mock_ollama:
            result = director.query("Test prompt", max_retries=0)
            assert result is not None
            mock_ollama.assert_called()

    def test_test_no_llm_returns_none(self, director, monkeypatch):
        """TEST_NO_LLM env var should skip all backends."""
        monkeypatch.setenv("TEST_NO_LLM", "1")
        result = director.query("Test prompt")
        assert result is None


class TestDirectorOutputSchema:
    """Test that DirectorOutput Pydantic model validates correctly."""

    def test_valid_full_response(self):
        """A full, valid response should validate."""
        output = DirectorOutput.model_validate_json(REALISTIC_LLM_RESPONSE)
        assert output.style.name == "hitchcock"
        assert output.pacing.speed.value == "slow"

    def test_minimal_response(self):
        """A minimal response with just required fields should validate."""
        minimal = json.dumps({
            "director_commentary": "Simple dynamic edit.",
        })
        output = DirectorOutput.model_validate_json(minimal)
        assert output.director_commentary == "Simple dynamic edit."
        assert output.style is None
        assert output.schema_version == SCHEMA_VERSION
