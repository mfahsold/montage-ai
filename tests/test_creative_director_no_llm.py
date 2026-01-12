import os
import pytest
from montage_ai.creative_director import CreativeDirector


def test_query_backend_respects_test_no_llm(monkeypatch):
    monkeypatch.setenv("TEST_NO_LLM", "1")
    cd = CreativeDirector()
    with pytest.raises(RuntimeError):
        cd._query_backend("Test prompt")
