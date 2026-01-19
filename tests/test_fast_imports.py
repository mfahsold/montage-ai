import importlib
import sys
import time
from unittest.mock import patch


def test_web_ui_app_import_is_fast():
    """Importing the web UI module should be fast (no heavy ML/CV imports at module import time)."""
    if 'montage_ai.web_ui.app' in sys.modules:
        importlib.reload(importlib.import_module('montage_ai.web_ui.app'))
    start = time.perf_counter()
    mod = importlib.import_module('montage_ai.web_ui.app')
    elapsed = time.perf_counter() - start
    # Conservative threshold to avoid flaky CI; original import cost was multiple seconds.
    assert elapsed < 1.0, f"web_ui.app import is too slow: {elapsed:.2f}s"


def test_autoreframe_lazy_and_patchable():
    """Verify the module exposes a patchable AutoReframeEngine and that importing the module
    does not eagerly import the heavy `montage_ai.auto_reframe` submodule."""
    importlib.reload(importlib.import_module('montage_ai.web_ui.app'))
    app_mod = importlib.import_module('montage_ai.web_ui.app')

    # If the heavy implementation was already imported earlier in the test session
    # (other tests may import it), skip the strict presence check to avoid order
    # dependent failures. Otherwise ensure the lazy-load behaviour.
    if 'montage_ai.auto_reframe' in sys.modules:
        import pytest
        pytest.skip("montage_ai.auto_reframe already imported in this session")

    # The heavy implementation should not be loaded just by importing the web UI module
    assert 'montage_ai.auto_reframe' not in sys.modules

    # The symbol must exist and be patchable by tests (maintains existing test behaviour)
    with patch('montage_ai.web_ui.app.AutoReframeEngine') as MockAR:
        MockAR.return_value.analyze.return_value = []
        inst = app_mod.AutoReframeEngine()
        MockAR.assert_called()
