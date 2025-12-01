# Professional Video Editing Concepts

Research-backed implementation of professional editing workflows for Montage AI.

## Core Principle: Footage Consumption

**"Das Schnittmaterial wird möglichst immer nur einmal genutzt"**
(Footage should ideally be used only once)

This is a fundamental principle from professional film editing. Once a clip is used,
it's "consumed" and removed from the available pool to ensure variety and prevent
repetitive content.

## Story Arc Structure

Professional edits follow a narrative arc:

1. **INTRO (0-15%)** - Establishing shots, calm energy
2. **BUILD (15-40%)** - Rising tension, increasing variety
3. **CLIMAX (40-70%)** - Peak intensity, fast cuts
4. **SUSTAIN (70-90%)** - Maintain interest, varied pacing
5. **OUTRO (90-100%)** - Resolution, return to calm

## Implementation

See `src/montage_ai/footage_manager.py` for the implementation:
- `FootagePoolManager` - Tracks clip usage
- `StoryArcController` - Maps timeline position to editing requirements
- `select_for_position()` - Intelligent clip selection based on story phase

## References

- Wikipedia: Film Editing, Montage (film editing)
- Continuity editing principles
- Soviet Montage Theory (Eisenstein, Kuleshov)
- Hollywood B-roll conventions
- Shot logging and EDL workflows
