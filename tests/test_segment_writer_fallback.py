import subprocess
import os
from pathlib import Path
import pytest

from montage_ai.segment_writer import SegmentWriter, SegmentInfo


def make_completed(cmd, returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(cmd, returncode=returncode, stdout=stdout, stderr=stderr)


def test_concat_fallback_success(tmp_path, monkeypatch):
    outdir = tmp_path / "segments"
    outdir.mkdir()

    # Create dummy segment files
    seg1 = outdir / "segment_0000.mp4"
    seg2 = outdir / "segment_0001.mp4"
    seg1.write_bytes(b"dummy")
    seg2.write_bytes(b"dummy")

    writer = SegmentWriter(output_dir=str(outdir))
    writer.segments = [
        SegmentInfo(path=str(seg1), index=0, clip_count=1, duration=1.0),
        SegmentInfo(path=str(seg2), index=1, clip_count=1, duration=1.0),
    ]

    # First run_command simulates hwupload failure
    first = make_completed(["ffmpeg"], returncode=1, stderr="hwupload: A hardware device reference is required to upload frames to.")
    # Second (fallback) succeeds
    second = make_completed(["ffmpeg"], returncode=0, stderr="")

    calls = {"n": 0}

    def fake_run_command(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return first
        # On fallback success, simulate creation of actual output file (last arg)
        out = cmd[-1] if isinstance(cmd, (list, tuple)) else None
        if out and isinstance(out, str) and out.endswith('.mp4'):
            try:
                Path(out).write_bytes(b"")
            except Exception:
                pass
        return second

    monkeypatch.setattr("montage_ai.segment_writer.run_command", fake_run_command)

    output_path = str(outdir / "final.mp4")
    success = writer.concatenate_segments(output_path)
    assert success
    # Fallback was invoked (two calls)
    assert calls["n"] >= 2


def test_concat_fallback_failure(tmp_path, monkeypatch):
    outdir = tmp_path / "segments"
    outdir.mkdir()

    seg1 = outdir / "segment_0000.mp4"
    seg1.write_bytes(b"dummy")

    writer = SegmentWriter(output_dir=str(outdir))
    writer.segments = [SegmentInfo(path=str(seg1), index=0, clip_count=1, duration=1.0)]

    first = make_completed(["ffmpeg"], returncode=1, stderr="error reinitializing filters")
    second = make_completed(["ffmpeg"], returncode=2, stderr="conversion failed")

    calls = {"n": 0}

    def fake_run_command(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return first
        # Simulate fallback attempt (but failure)
        return second

    monkeypatch.setattr("montage_ai.segment_writer.run_command", fake_run_command)

    output_path = str(outdir / "final_fail.mp4")
    success = writer.concatenate_segments(output_path)
    assert not success
    assert calls["n"] >= 2
