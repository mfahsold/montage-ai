"""
Tests for Multi-GPU Encoder Router.

TDD: Tests written BEFORE implementation.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock


class TestEncoderNodeConfig:
    """Tests for EncoderNode configuration dataclass."""

    def test_encoder_node_basic_attributes(self):
        """EncoderNode should have basic attributes."""
        from montage_ai.encoder_router import EncoderNode

        node = EncoderNode(
            name="fluxibri",
            hostname="192.168.1.12",
            encoder_type="vaapi",
            max_concurrent=2,
            priority=10,
        )

        assert node.name == "fluxibri"
        assert node.hostname == "192.168.1.12"
        assert node.encoder_type == "vaapi"
        assert node.max_concurrent == 2
        assert node.priority == 10

    def test_encoder_node_with_ssh_config(self):
        """EncoderNode should support SSH configuration for remote execution."""
        from montage_ai.encoder_router import EncoderNode

        node = EncoderNode(
            name="jetson",
            hostname="192.168.1.100",
            encoder_type="nvmpi",
            ssh_user="codeai",
            ssh_key_path="/home/codeai/.ssh/id_rsa",
        )

        assert node.ssh_user == "codeai"
        assert node.ssh_key_path == "/home/codeai/.ssh/id_rsa"

    def test_encoder_node_local_flag(self):
        """EncoderNode should detect local vs remote."""
        from montage_ai.encoder_router import EncoderNode

        local_node = EncoderNode(name="local", hostname="localhost", encoder_type="vaapi")
        remote_node = EncoderNode(name="remote", hostname="192.168.1.100", encoder_type="nvmpi")

        assert local_node.is_local is True
        assert remote_node.is_local is False


class TestEncoderRouter:
    """Tests for EncoderRouter main class."""

    def test_router_initialization_auto_detect(self):
        """Router should auto-detect local GPU on init."""
        from montage_ai.encoder_router import EncoderRouter

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="vaapi", is_gpu=True, encoder="h264_vaapi")

            router = EncoderRouter()

            assert len(router.nodes) >= 1
            assert router.local_node is not None

    def test_router_add_remote_node(self):
        """Router should allow adding remote nodes."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False, encoder="libx264")

            router = EncoderRouter()
            jetson_node = EncoderNode(
                name="jetson",
                hostname="192.168.1.100",
                encoder_type="nvmpi",
                ssh_user="codeai",
            )
            router.add_node(jetson_node)

            assert len(router.nodes) == 2
            assert any(n.name == "jetson" for n in router.nodes)

    def test_router_select_best_node_by_priority(self):
        """Router should select node with highest priority."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            router = EncoderRouter()
            router.nodes = []  # Clear auto-detected

            # Add nodes with different priorities
            router.add_node(EncoderNode(name="low", hostname="a", encoder_type="cpu", priority=1))
            router.add_node(EncoderNode(name="high", hostname="b", encoder_type="vaapi", priority=10))
            router.add_node(EncoderNode(name="med", hostname="c", encoder_type="nvmpi", priority=5))

            best = router.select_best_node()
            assert best.name == "high"

    def test_router_select_node_by_encoder_type(self):
        """Router should filter by encoder type."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            router = EncoderRouter()
            router.nodes = []

            router.add_node(EncoderNode(name="amd", hostname="a", encoder_type="vaapi", priority=5))
            router.add_node(EncoderNode(name="jetson", hostname="b", encoder_type="nvmpi", priority=10))

            best = router.select_best_node(encoder_type="vaapi")
            assert best.name == "amd"

    def test_router_track_node_load(self):
        """Router should track current load per node."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            router = EncoderRouter()
            router.nodes = []

            node = EncoderNode(name="gpu", hostname="a", encoder_type="vaapi", max_concurrent=2)
            router.add_node(node)

            assert router.get_node_load("gpu") == 0

            router.increment_load("gpu")
            assert router.get_node_load("gpu") == 1

            router.decrement_load("gpu")
            assert router.get_node_load("gpu") == 0

    def test_router_skip_overloaded_nodes(self):
        """Router should skip nodes at max capacity."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            router = EncoderRouter()
            router.nodes = []

            node_a = EncoderNode(name="a", hostname="a", encoder_type="vaapi", priority=10, max_concurrent=1)
            node_b = EncoderNode(name="b", hostname="b", encoder_type="vaapi", priority=5, max_concurrent=2)
            router.add_node(node_a)
            router.add_node(node_b)

            # Saturate high-priority node
            router.increment_load("a")

            best = router.select_best_node()
            assert best.name == "b"  # Falls back to lower priority


class TestEncoderRouterDistributedTasks:
    """Tests for distributed encoding tasks."""

    def test_encode_task_local(self):
        """Should encode locally when local GPU available."""
        from montage_ai.encoder_router import EncoderRouter, encode_segment
        import asyncio

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="vaapi", is_gpu=True, encoder="h264_vaapi")

            router = EncoderRouter()

            with patch("montage_ai.encoder_router._run_ffmpeg_local") as mock_ffmpeg:
                mock_ffmpeg.return_value = (True, "/output/segment_001.mp4")

                result = asyncio.run(encode_segment(
                    router,
                    input_path="/input/raw_001.mp4",
                    output_path="/output/segment_001.mp4",
                ))

                assert result.success is True
                mock_ffmpeg.assert_called_once()

    def test_encode_task_remote_ssh(self):
        """Should dispatch to remote node via SSH."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode, encode_segment
        import asyncio

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            router = EncoderRouter()
            router.nodes = []

            jetson = EncoderNode(
                name="jetson",
                hostname="192.168.1.100",
                encoder_type="nvmpi",
                ssh_user="codeai",
                priority=10,
            )
            router.add_node(jetson)

            with patch("montage_ai.encoder_router._run_ffmpeg_ssh") as mock_ssh:
                mock_ssh.return_value = (True, "/output/segment_001.mp4")

                result = asyncio.run(encode_segment(
                    router,
                    input_path="/input/raw_001.mp4",
                    output_path="/output/segment_001.mp4",
                    prefer_gpu=True,
                ))

                assert result.success is True
                mock_ssh.assert_called_once()


class TestAdrenoV4L2Support:
    """Tests for Adreno GPU (Snapdragon) support via V4L2."""

    def test_detect_adreno_via_sysfs(self):
        """Should detect Adreno GPU via sysfs."""
        from montage_ai.core.hardware import _has_adreno

        with patch("os.path.exists") as mock_exists:
            # Adreno GPU in sysfs
            mock_exists.side_effect = lambda p: p in [
                "/sys/class/drm/card0/device/vendor",
                "/dev/video0",
            ]

            with patch("builtins.open", create=True) as mock_open:
                # Qualcomm vendor ID
                mock_open.return_value.__enter__.return_value.read.return_value = "0x5143"

                assert _has_adreno() is True

    def test_adreno_encoder_v4l2(self):
        """Adreno should use V4L2 H.264 encoder."""
        from montage_ai.core.hardware import get_hwaccel_by_type

        with patch("montage_ai.core.hardware._has_adreno") as mock_adreno:
            mock_adreno.return_value = True

            with patch("montage_ai.core.hardware._check_ffmpeg_encoder") as mock_enc:
                mock_enc.return_value = True

                config = get_hwaccel_by_type("adreno")

                assert config is not None
                assert config.type == "adreno"
                assert "v4l2" in config.encoder.lower() or "h264" in config.encoder.lower()


class TestROCmSupport:
    """Tests for AMD ROCm GPU support."""

    def test_detect_rocm_gpu(self):
        """Should detect AMD ROCm GPU via rocm-smi or sysfs."""
        from montage_ai.core.hardware import _has_rocm

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/opt/rocm/bin/rocm-smi"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="GPU[0]")

                assert _has_rocm() is True

    def test_rocm_encoder_uses_vaapi(self):
        """ROCm should use VAAPI encoder on Linux (AMF is Windows-only)."""
        from montage_ai.core.hardware import get_hwaccel_by_type

        with patch("montage_ai.core.hardware._has_rocm") as mock_rocm:
            mock_rocm.return_value = True

            with patch("montage_ai.core.hardware._check_ffmpeg_encoder") as mock_enc:
                # VAAPI encoder available (standard for Linux ROCm)
                mock_enc.side_effect = lambda e: e in ["h264_vaapi", "hevc_vaapi"]

                config = get_hwaccel_by_type("rocm")

                assert config is not None
                assert config.type == "rocm"
                assert "vaapi" in config.encoder.lower()

    def test_rocm_fallback_to_vaapi(self):
        """ROCm should fallback to VAAPI if AMF not available."""
        from montage_ai.core.hardware import get_hwaccel_by_type

        with patch("montage_ai.core.hardware._has_rocm") as mock_rocm:
            mock_rocm.return_value = True

            with patch("montage_ai.core.hardware._has_vaapi") as mock_vaapi:
                mock_vaapi.return_value = True

            with patch("montage_ai.core.hardware._check_ffmpeg_encoder") as mock_enc:
                # AMF not available, but VAAPI works
                mock_enc.side_effect = lambda e: e in ["h264_vaapi", "hevc_vaapi"]

                config = get_hwaccel_by_type("rocm")

                # Falls back to vaapi with rocm type for tracking
                assert config is not None
                assert "vaapi" in config.encoder.lower()

    def test_router_includes_rocm_node(self):
        """Router should add ROCm node when detected."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="rocm", is_gpu=True, encoder="h264_vaapi")

            router = EncoderRouter()

            assert router.local_node is not None
            assert router.local_node.encoder_type in ("rocm", "vaapi")


class TestCGPUIntegration:
    """Tests for Cloud GPU (cgpu) integration."""

    def test_router_detects_cgpu_available(self):
        """Router should detect CGPU availability."""
        from montage_ai.encoder_router import EncoderRouter

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            with patch("montage_ai.encoder_router.is_cgpu_available") as mock_cgpu:
                mock_cgpu.return_value = True

                router = EncoderRouter(enable_cgpu=True)

                assert router.cgpu_available is True

    def test_route_heavy_task_to_cgpu(self):
        """Heavy tasks (4K upscale, long duration) should route to CGPU."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode, route_task

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="cpu", is_gpu=False)

            with patch("montage_ai.encoder_router.is_cgpu_available") as mock_cgpu:
                mock_cgpu.return_value = True

                router = EncoderRouter(enable_cgpu=True)

                # Heavy task: 4K upscale
                destination = route_task(
                    router,
                    task_type="upscale",
                    input_resolution=(1920, 1080),
                    output_resolution=(3840, 2160),
                    duration_seconds=300,
                )

                assert destination == "cgpu"

    def test_route_light_task_locally(self):
        """Light tasks should stay local."""
        from montage_ai.encoder_router import EncoderRouter, route_task

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="vaapi", is_gpu=True)

            router = EncoderRouter(enable_cgpu=True)

            # Light task: Preview encode
            destination = route_task(
                router,
                task_type="encode",
                input_resolution=(1920, 1080),
                output_resolution=(640, 360),
                duration_seconds=30,
            )

            assert destination == "local"


class TestClusterDiscovery:
    """Tests for automatic cluster node discovery."""

    def test_discover_k8s_nodes(self):
        """Should discover encoding nodes from K8s cluster."""
        from montage_ai.encoder_router import discover_cluster_nodes

        with patch("subprocess.run") as mock_run:
            # Simulate kubectl output (wide format with 6+ columns)
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="""codeai-fluxibriserver   Ready    worker    45d   v1.34.3   192.168.1.16   <none>
codeaijetson-desktop    Ready    worker    45d   v1.34.3   192.168.1.15   <none>
codeai-thinkpad-t14s    Ready    <none>    29h   v1.34.3   192.168.1.237  <none>
"""
            )

            nodes = discover_cluster_nodes()

            assert len(nodes) >= 2
            assert any("jetson" in n.name.lower() for n in nodes)

    def test_probe_node_capabilities(self):
        """Should probe node for GPU capabilities via SSH."""
        from montage_ai.encoder_router import probe_node_capabilities
        import asyncio

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(
                b"h264_nvmpi\nhevc_nvmpi",
                b""
            ))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            caps = asyncio.run(probe_node_capabilities("192.168.1.100", "codeai"))

            assert "nvmpi" in caps["encoders"]


class TestEncoderRouterE2E:
    """End-to-end tests for encoder routing."""

    def test_parallel_encoding_multiple_segments(self):
        """Should encode multiple segments in parallel across nodes."""
        from montage_ai.encoder_router import EncoderRouter, EncoderNode, encode_segments_parallel
        import asyncio

        with patch("montage_ai.encoder_router.get_best_hwaccel") as mock_hw:
            mock_hw.return_value = MagicMock(type="vaapi", is_gpu=True)

            router = EncoderRouter()
            router.nodes = []

            # Add two GPU nodes
            router.add_node(EncoderNode(name="amd", hostname="localhost", encoder_type="vaapi", max_concurrent=2))
            router.add_node(EncoderNode(name="jetson", hostname="192.168.1.100", encoder_type="nvmpi", max_concurrent=1, ssh_user="codeai"))

            segments = [
                {"input": "/in/seg1.mp4", "output": "/out/seg1.mp4"},
                {"input": "/in/seg2.mp4", "output": "/out/seg2.mp4"},
                {"input": "/in/seg3.mp4", "output": "/out/seg3.mp4"},
            ]

            with patch("montage_ai.encoder_router._run_ffmpeg_local") as mock_local:
                with patch("montage_ai.encoder_router._run_ffmpeg_ssh") as mock_ssh:
                    mock_local.return_value = (True, "")
                    mock_ssh.return_value = (True, "")

                    results = asyncio.run(encode_segments_parallel(router, segments))

                    assert len(results) == 3
                    assert all(r.success for r in results)
