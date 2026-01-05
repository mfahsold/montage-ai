
import pytest
from src.montage_ai.audio_analysis import MusicSection, fit_story_arc_to_sections

class TestStoryArc:
    def test_classic_edm_structure(self):
        # Intro -> Build -> Drop -> Outro
        total_duration = 100.0
        sections = [
            MusicSection(0.0, 20.0, "low", 0.2),      # 0: Intro
            MusicSection(20.0, 40.0, "medium", 0.5),  # 1: Build (because next is high)
            MusicSection(40.0, 60.0, "high", 0.9),    # 2: Drop
            MusicSection(60.0, 80.0, "medium", 0.6),  # 3: Verse/Chorus (after drop)
            MusicSection(80.0, 100.0, "low", 0.2)     # 4: Outro
        ]
        
        labeled = fit_story_arc_to_sections(sections, total_duration)
        
        assert labeled[0].label == "intro"
        assert labeled[1].label == "build"
        assert labeled[2].label == "drop"
        assert labeled[4].label == "outro"

    def test_short_clip_intro_outro(self):
        # Just Intro -> Outro
        total_duration = 20.0
        sections = [
            MusicSection(0.0, 10.0, "low", 0.2),
            MusicSection(10.0, 20.0, "low", 0.2)
        ]
        labeled = fit_story_arc_to_sections(sections, total_duration)
        
        assert labeled[0].label == "intro"
        assert labeled[1].label == "outro"

    def test_constant_high_energy(self):
        # All High (e.g. Punk Rock)
        total_duration = 60.0
        sections = [
            MusicSection(0.0, 20.0, "high", 0.8),
            MusicSection(20.0, 40.0, "high", 0.8),
            MusicSection(40.0, 60.0, "high", 0.8)
        ]
        labeled = fit_story_arc_to_sections(sections, total_duration)
        
        # First might be Intro or Drop depending on logic.
        # Logic says: if i==0 -> Intro.
        assert labeled[0].label == "intro" 
        # Logic says: High energy -> Drop.
        assert labeled[1].label == "drop"
        # Logic says: Last -> Outro.
        assert labeled[2].label == "outro"
