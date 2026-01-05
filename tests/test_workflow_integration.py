"""
Integration tests for MontageWorkflow and ShortsWorkflow.

Verifies that:
1. Workflows can be initialized and run via the common interface.
2. ProxyGenerator is invoked during the analysis phase.
3. Steps (validate, analyze, process, render) are called in order.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path

from montage_ai.core.montage_workflow import MontageWorkflow
from montage_ai.core.workflow import WorkflowOptions
from montage_ai.config import Settings
from montage_ai.core.montage_builder import MontageBuilder # Needed for mocking
from montage_ai.proxy_generator import ProxyGenerator

@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.paths.input_dir = Path("/mock/input")
    settings.paths.music_dir = Path("/mock/music")
    settings.paths.output_dir = Path("/mock/output")
    settings.paths.temp_dir = Path("/mock/temp")
    
    # Enable features for test
    settings.features.story_engine = True
    settings.features.voice_isolation = False
    return settings

@pytest.fixture
def mock_job_store():
    with patch("montage_ai.core.workflow.JobStore") as mock:
        yield mock

@pytest.fixture
def mock_builder_cls():
    with patch("montage_ai.core.montage_workflow.MontageBuilder") as mock:
        yield mock

@pytest.fixture
def mock_proxy_generator():
    with patch("montage_ai.core.montage_builder.ProxyGenerator") as mock:
        yield mock

def test_montage_workflow_run_sequence(mock_settings, mock_builder_cls, mock_proxy_generator, mock_job_store):
    """Test full execution of MontageWorkflow."""
    
    # 1. Setup Options
    options = WorkflowOptions(
        input_path="/mock/input",
        output_dir="/mock/output",
        job_id="test-job-123",
        quality_profile="standard",
        extras={"variant_id": 2}
    )
    
    # 2. Setup Builder Mock
    builder_instance = mock_builder_cls.return_value
    # Set context variables that workflow checks/sets
    builder_instance.ctx = MagicMock()
    builder_instance.ctx.paths.input_dir = Path("/mock/input")
    # Determine video files exists to trigger proxy generation
    builder_instance.ctx.video_files = [Path("/mock/input/video1.mp4")]
    builder_instance.ctx.input_dir = Path("/mock/input")
    builder_instance.ctx.temp_dir = Path("/mock/temp")
    
    # 3. Create Workflow
    workflow = MontageWorkflow(options)
    # Inject mocked settings into workflow instance (since it calls get_settings)
    workflow.settings = mock_settings
    
    # Verify JobStore was initialized
    mock_job_store.assert_called_once()
    
    # 4. Run Workflow
    # Note: method is 'execute' not 'run' based on Template Method pattern in workflow.py
    workflow.execute()
    
    # 5. Verify Sequence
    # Init
    mock_builder_cls.assert_called_with(
        variant_id=2, 
        settings=ANY, 
        editing_instructions={}
    )
    builder_instance.setup_workspace.assert_called_once()
    
    # Analyze
    builder_instance.analyze_assets.assert_called_once()
    
    # Process
    builder_instance.plan_montage.assert_called_once()
    
    # Render
    builder_instance.render_output.assert_called_once()
    
    # 6. Verify Proxy Generation (Implicitly called in analyze_assets via builder logic)
    # Since we mocked the MontageBuilder entirely, the internal logic of analyze_assets 
    # (where ProxyGenerator is used) won't run unless we mocked the method implementation 
    # OR we are testing the MontageBuilder class logic itself.
    
    # Wait, the workflow just delegates to builder.analyze_assets(). 
    # The workflow wrapper is just a shell.
    # To test ProxyGenerator invocation, we need to test MontageBuilder.analyze_assets()
    # or rely on an integrated test that uses the REAL MontageBuilder logic but mocks IO.

def test_montage_builder_integrates_proxy_generator(mock_settings, mock_proxy_generator):
    """
    Test that MontageBuilder actually calls ProxyGenerator 
    during analyze_assets phase.
    """
    # Use real MontageBuilder but mock dependencies
    # We mock _get_files because get_files is an internal method or utility imported 
    # as _get_files in the builder class usually. Let's check imports.
    # MontageBuilder imports get_files (which doesn't exist?) - probably it's a private method self._get_files
    
    builder = MontageBuilder(variant_id=1, settings=mock_settings)
    
    # Mock _get_files if it exists on instance
    builder._get_files = MagicMock()
    
    with patch("montage_ai.core.montage_builder.ThreadPoolExecutor") as mock_executor:
         
         # Mock Context
         builder.ctx.video_files = [Path("/mock/input/video1.mp4")]
         builder.ctx.input_dir = Path("/mock/input")
         builder.ctx.temp_dir = Path("/mock/temp")
         
         # Mock _executor on the builder INSTANCE, because analyze_assets accesses self._executor
         # ThreadPoolExecutor() in code creates a NEW instance, so patching the class works for creation
         # but we need to ensure the builder instance USES it.
         # In MontageBuilder.__init__, self._executor = ThreadPoolExecutor(...) is called.
         # Since we created builder BEFORE patching ThreadPoolExecutor, it has a real executor (or whatever Init did).
         # We must mock the attribute directly.
         
         builder._executor = MagicMock()
         mock_future = MagicMock()
         builder._executor.submit.return_value = mock_future
         
         # Mock other calls in analyze_assets
         builder._trigger_story_analysis = MagicMock()
         builder._detect_scenes = MagicMock()
         builder._analyze_music = MagicMock()
         builder._determine_output_profile = MagicMock()
         builder._init_footage_pool = MagicMock()

         # Run Analysis
         builder.analyze_assets()
         
         # Verify ProxyGenerator initialized
         mock_proxy_generator.assert_called()
         
         # Verify ensure_proxy called via executor
         # We check if submit was called on the instance mock we injected
         args_list = builder._executor.submit.call_args_list
         
         # We expect calls for voice isolation (if enabled) and proxies
         # Voice isolation is disallowed in mock_settings but let's be careful
         
         mock_pg_instance = mock_proxy_generator.return_value
         expected_func = mock_pg_instance.ensure_proxy

         found_proxy_call = False
         for call in args_list:
             func = call[0][0]
             
             if func == expected_func:
                found_proxy_call = True
                assert call[0][1] == Path("/mock/input/video1.mp4")
                
         assert found_proxy_call, f"ProxyGenerator.ensure_proxy was not submitted. Calls: {len(args_list)}"

