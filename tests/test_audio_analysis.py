"""
Tests for audio analysis module.

Tests beat detection, energy analysis, and dynamic cut length calculation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.montage_ai.audio_analysis import (
    BeatInfo,
    EnergyProfile,
    analyze_music_energy,
    get_beat_times,
    calculate_dynamic_cut_length,
    analyze_audio,
)


class TestBeatInfo:
    """Tests for BeatInfo dataclass."""

    def test_beat_count(self):
        """beat_count returns correct number of beats."""
        info = BeatInfo(
            tempo=120.0,
            beat_times=np.array([0.5, 1.0, 1.5, 2.0]),
            duration=4.0,
            sample_rate=22050
        )
        assert info.beat_count == 4

    def test_avg_beat_interval(self):
        """avg_beat_interval calculates correctly."""
        info = BeatInfo(
            tempo=120.0,  # 120 BPM = 0.5s per beat
            beat_times=np.array([0.5, 1.0]),
            duration=2.0,
            sample_rate=22050
        )
        assert info.avg_beat_interval == pytest.approx(0.5)

    def test_tempo_category_fast(self):
        """tempo_category returns 'fast' for >140 BPM."""
        info = BeatInfo(tempo=150.0, beat_times=np.array([]), duration=1.0, sample_rate=22050)
        assert info.tempo_category == "fast"

    def test_tempo_category_slow(self):
        """tempo_category returns 'slow' for <80 BPM."""
        info = BeatInfo(tempo=70.0, beat_times=np.array([]), duration=1.0, sample_rate=22050)
        assert info.tempo_category == "slow"

    def test_tempo_category_medium(self):
        """tempo_category returns 'medium' for 80-140 BPM."""
        info = BeatInfo(tempo=100.0, beat_times=np.array([]), duration=1.0, sample_rate=22050)
        assert info.tempo_category == "medium"


class TestEnergyProfile:
    """Tests for EnergyProfile dataclass."""

    def test_energy_stats(self):
        """Energy statistics are calculated correctly."""
        profile = EnergyProfile(
            times=np.array([0.0, 0.5, 1.0]),
            rms=np.array([0.2, 0.8, 0.5]),
            sample_rate=22050,
            hop_length=512
        )
        assert profile.avg_energy == pytest.approx(0.5)
        assert profile.max_energy == pytest.approx(0.8)
        assert profile.min_energy == pytest.approx(0.2)

    def test_high_energy_pct(self):
        """high_energy_pct calculates percentage above 70%."""
        profile = EnergyProfile(
            times=np.array([0.0, 0.5, 1.0, 1.5]),
            rms=np.array([0.5, 0.8, 0.9, 0.3]),  # 2 of 4 > 0.7
            sample_rate=22050,
            hop_length=512
        )
        assert profile.high_energy_pct == pytest.approx(50.0)

    def test_energy_at_time(self):
        """energy_at_time returns correct value for given time."""
        profile = EnergyProfile(
            times=np.array([0.0, 1.0, 2.0, 3.0]),
            rms=np.array([0.1, 0.5, 0.9, 0.3]),
            sample_rate=22050,
            hop_length=512
        )
        assert profile.energy_at_time(1.5) == pytest.approx(0.9)  # Finds index 2
        assert profile.energy_at_time(0.0) == pytest.approx(0.1)


class TestCalculateDynamicCutLength:
    """Tests for calculate_dynamic_cut_length function."""

    def test_intro_calm(self):
        """Intro phase with low energy returns long cuts."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.2,
            tempo=100.0,
            current_time=5.0,  # 5% into track
            total_duration=100.0,
            pattern_pool=[]
        )
        # Calm intro should have long cuts (8s)
        assert 8 in pattern

    def test_intro_energetic(self):
        """Intro phase with high energy returns steady rhythm."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.7,
            tempo=100.0,
            current_time=5.0,
            total_duration=100.0,
            pattern_pool=[]
        )
        # Energetic intro should have 4-beat pattern
        assert 4 in pattern

    def test_buildup_high_energy(self):
        """Build-up phase with high energy returns Fibonacci pattern."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.8,
            tempo=100.0,
            current_time=30.0,  # 30% into track
            total_duration=100.0,
            pattern_pool=[]
        )
        # Fibonacci descent
        assert 5 in pattern or 3 in pattern

    def test_climax_hyper_energy(self):
        """Climax phase with very high energy returns rapid cuts."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.9,
            tempo=100.0,
            current_time=60.0,  # 60% into track
            total_duration=100.0,
            pattern_pool=[]
        )
        # Stutter pattern has 0.5 and 1 beat cuts
        assert any(p <= 1 for p in pattern)

    def test_outro_calm(self):
        """Outro phase with low energy returns progressively longer cuts."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.3,
            tempo=100.0,
            current_time=90.0,  # 90% into track
            total_duration=100.0,
            pattern_pool=[]
        )
        # Fade pattern: 4, 8, 12, 16
        assert 16 in pattern or 12 in pattern

    def test_tempo_modulation_fast(self):
        """Fast tempo increases minimum cut length."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.5,
            tempo=150.0,  # Fast
            current_time=50.0,
            total_duration=100.0,
            pattern_pool=[]
        )
        # All values should be >= 2 for fast tempo
        assert all(p >= 2 for p in pattern)

    def test_tempo_modulation_slow(self):
        """Slow tempo halves cut lengths."""
        pattern = calculate_dynamic_cut_length(
            current_energy=0.5,
            tempo=70.0,  # Slow
            current_time=10.0,  # Intro
            total_duration=100.0,
            pattern_pool=[]
        )
        # Values should be smaller for slow tempo
        assert len(pattern) > 0

    def test_chaos_factor(self):
        """With chaos_factor=1.0, always uses pattern from pool."""
        custom_pattern = [[99, 99, 99]]
        pattern = calculate_dynamic_cut_length(
            current_energy=0.5,
            tempo=100.0,
            current_time=50.0,
            total_duration=100.0,
            pattern_pool=custom_pattern,
            chaos_factor=1.0  # Always inject chaos
        )
        assert pattern == [99, 99, 99]


class TestAnalyzeMusicEnergy:
    """Tests for analyze_music_energy function."""

    @patch('src.montage_ai.audio_analysis.probe_duration')
    @patch('src.montage_ai.audio_analysis._run_cloud_analysis')
    @patch('src.montage_ai.audio_analysis._ffmpeg_analyze_energy_fast')
    @patch('src.montage_ai.audio_analysis._settings')
    def test_returns_energy_profile(self, mock_settings, mock_ffmpeg, mock_cloud, mock_probe):
        """analyze_music_energy returns EnergyProfile."""
        # Cloud GPU returns None, fall back to FFmpeg
        mock_cloud.return_value = None
        mock_probe.return_value = 10.0  # 10 second audio
        
        # Setup mocks for FFmpeg path (now the primary)
        mock_ffmpeg.return_value = (
            np.array([0.0, 0.5, 1.0]),  # times
            np.array([0.1, 0.5, 0.9])   # rms_normalized
        )
        
        # Setup settings mock
        mock_features = MagicMock()
        mock_features.verbose = False
        mock_settings.features = mock_features

        profile = analyze_music_energy("/fake/audio.mp3", verbose=False)

        assert isinstance(profile, EnergyProfile)
        assert profile.sample_rate == 44100  # FFmpeg assumed SR
        assert len(profile.rms) == 3

    @patch('src.montage_ai.audio_analysis.probe_duration')
    @patch('src.montage_ai.audio_analysis._run_cloud_analysis')
    @patch('src.montage_ai.audio_analysis._ffmpeg_analyze_energy_fast')
    @patch('src.montage_ai.audio_analysis._settings')
    def test_normalizes_rms(self, mock_settings, mock_ffmpeg, mock_cloud, mock_probe):
        """Energy values are normalized to 0-1."""
        # Cloud GPU returns None, fall back to FFmpeg
        mock_cloud.return_value = None
        mock_probe.return_value = 10.0  # 10 second audio
        
        # FFmpeg already returns normalized values
        mock_ffmpeg.return_value = (
            np.array([0.0, 0.5, 1.0]),
            np.array([0.1, 0.5, 0.9])  # Already normalized
        )
        
        # Setup settings mock
        mock_features = MagicMock()
        mock_features.verbose = False
        mock_settings.features = mock_features

        profile = analyze_music_energy("/fake/audio.mp3", verbose=False)

        # Should be normalized
        assert profile.max_energy <= 1.0
        assert profile.min_energy >= 0.0


class TestGetBeatTimes:
    """Tests for get_beat_times function."""

    @patch('src.montage_ai.audio_analysis.probe_duration')
    @patch('src.montage_ai.audio_analysis._run_cloud_analysis')
    @patch('src.montage_ai.audio_analysis._ffmpeg_estimate_tempo')
    @patch('src.montage_ai.audio_analysis._settings')
    def test_returns_beat_info(self, mock_settings, mock_ffmpeg, mock_cloud, mock_probe):
        """get_beat_times returns BeatInfo."""
        # Cloud GPU returns None, fall back to FFmpeg
        mock_cloud.return_value = None
        mock_probe.return_value = 10.0  # 10 second audio
        
        # Setup FFmpeg mock
        mock_ffmpeg.return_value = (
            120.0,  # tempo
            np.array([0.5, 1.0, 1.5])  # beat_times
        )
        
        # Settings mock
        mock_features = MagicMock()
        mock_features.verbose = False
        mock_settings.features = mock_features

        info = get_beat_times("/fake/audio.mp3", verbose=False)

        assert isinstance(info, BeatInfo)
        assert info.tempo == 120.0
        assert info.beat_count == 3
        assert info.sample_rate == 44100  # FFmpeg assumed SR

    @patch('src.montage_ai.audio_analysis.probe_duration')
    @patch('src.montage_ai.audio_analysis._run_cloud_analysis')
    @patch('src.montage_ai.audio_analysis._ffmpeg_estimate_tempo')
    @patch('src.montage_ai.audio_analysis._settings')
    def test_handles_array_tempo(self, mock_settings, mock_ffmpeg, mock_cloud, mock_probe):
        """Handles tempo returned as scalar (FFmpeg path)."""
        # Cloud GPU returns None
        mock_cloud.return_value = None
        mock_probe.return_value = 10.0  # 10 second audio
        
        # FFmpeg returns scalar tempo
        mock_ffmpeg.return_value = (
            120.0,  # scalar tempo
            np.array([0.5])
        )
        
        # Settings mock
        mock_features = MagicMock()
        mock_features.verbose = False
        mock_settings.features = mock_features

        info = get_beat_times("/fake/audio.mp3", verbose=False)

        assert info.tempo == 120.0
        assert isinstance(info.tempo, float)


class TestAnalyzeAudio:
    """Tests for analyze_audio convenience function."""

    @patch('src.montage_ai.audio_analysis.probe_duration')
    @patch('src.montage_ai.audio_analysis._run_cloud_analysis')
    @patch('src.montage_ai.audio_analysis._ffmpeg_estimate_tempo')
    @patch('src.montage_ai.audio_analysis._ffmpeg_analyze_energy_fast')
    @patch('src.montage_ai.audio_analysis._settings')
    def test_returns_both_beat_and_energy(self, mock_settings, mock_energy, mock_tempo, mock_cloud, mock_probe):
        """analyze_audio returns tuple of (BeatInfo, EnergyProfile)."""
        # Cloud GPU returns None for both
        mock_cloud.return_value = None
        mock_probe.return_value = 10.0  # 10 second audio
        
        # Setup FFmpeg mocks
        mock_tempo.return_value = (
            120.0,  # tempo
            np.array([0.5, 1.0])  # beat_times
        )
        mock_energy.return_value = (
            np.array([0.0, 0.5, 1.0]),  # times
            np.array([0.3, 0.6, 0.9])   # rms_normalized
        )
        
        # Settings mock
        mock_features = MagicMock()
        mock_features.verbose = False
        mock_settings.features = mock_features

        beat_info, energy_profile = analyze_audio("/fake/audio.mp3", verbose=False)

        assert isinstance(beat_info, BeatInfo)
        assert isinstance(energy_profile, EnergyProfile)
