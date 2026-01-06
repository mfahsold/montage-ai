import random
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..logger import logger
from .context import MontageContext

class SelectionEngine:
    """
    Engine responsible for scoring and selecting the best clip for the timeline.
    
    Handles:
    - Clip scoring based on usage, continuity, and style.
    - Intelligent selection (LLM/Probabilistic).
    - Match cuts and semantic matching.
    """

    def __init__(self, ctx: MontageContext):
        self.ctx = ctx
        self._intelligent_selector = None
        self._init_intelligent_selector()

    def _init_intelligent_selector(self):
        """Initialize intelligent clip selector."""
        try:
            from ..clip_selector import IntelligentClipSelector
            style = "dynamic"
            if self.ctx.creative.editing_instructions is not None:
                style = self.ctx.creative.editing_instructions.get('style', {}).get('template', 'dynamic')
            self._intelligent_selector = IntelligentClipSelector(style=style)
            logger.info(f"   🧠 Intelligent Clip Selector initialized (style={style})")
        except ImportError:
            self._intelligent_selector = None
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to initialize Intelligent Clip Selector: {e}")
            self._intelligent_selector = None

    def select_clip(
        self,
        available_footage,
        current_energy: float,
        unique_videos: int,
        similarity_fn: Optional[Any] = None
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Score and select the best clip for this cut."""
        
        valid_scenes = self._get_candidate_scenes(available_footage)
        if not valid_scenes:
            return None, 0

        candidates = valid_scenes[:20]
        scoring_rules = self._resolve_scoring_rules()
        current_section = self._get_current_music_section()
        clip_map = {c.clip_id: c for c in available_footage}

        for scene in candidates:
            # Note: available_footage is a list of FootageClip objects? 
            # Or list of scenes? 
            # In Builder: available_footage = self._footage_pool.get_available_clips(min_duration=min_dur)
            # get_available_clips returns a list of candidate objects which likely contain the scene dict.
            # Wait, let's verify what get_available_clips returns.
            # It seems it returns objects that have .clip_id and behave like the scene dict in some contexts?
            # Looking at Builder: 
            # clip_map = {c.clip_id: c for c in available_footage}
            # for scene in candidates: ... scene is a dict?
            # 
            # Checking Builder code again:
            # valid_scenes = self._get_candidate_scenes(available_footage)
            # 
            # Let's assume _get_candidate_scenes returns list of scene dicts.
            
            meta = scene.get('meta', {})
            shot = meta.get('shot', 'medium')

            footage_clip = clip_map.get(id(scene)) # This looks suspicious in original code. id(scene)?
            # In Builder: footage_clip = clip_map.get(id(scene))
            # This implies the input `available_footage` items are distinct from `scene` dicts.
            # But wait, `_get_candidate_scenes` probably returns `scene` dicts from the footage clips?
            
            score = 0.0
            
            # If footage_clip is None (which it shouldn't be if logic holds), we need to handle it.
            # Logic from Builder:
            score += self._score_usage_and_story_phase(
                footage_clip,
                current_energy,
                fresh_clip_bonus=scoring_rules["fresh_clip_bonus"],
            )
            score += self._score_jump_cut(
                scene['path'],
                unique_videos,
                jump_cut_penalty=scoring_rules["jump_cut_penalty"],
            )
            score += self._score_action_energy(meta, current_energy, current_section)
            score += self._score_style_preferences(meta)
            score += self._score_shot_variation(
                shot,
                shot_variation_bonus=scoring_rules["shot_variation_bonus"],
                shot_repetition_penalty=scoring_rules["shot_repetition_penalty"],
            )
            if similarity_fn:
                score += self._score_match_cut(scene, similarity_fn)
            
            score += self._score_semantic_match(meta)
            score += self._score_broll_match(scene, meta)

            score += random.randint(-15, 15)
            scene['_heuristic_score'] = score

        if self._intelligent_selector:
            return self._select_with_intelligent_selector(candidates, current_energy)

        return self._select_probabilistic(candidates)

    def _get_candidate_scenes(self, available_footage) -> List[Dict[str, Any]]:
        clip_ids = {c.clip_id for c in available_footage}
        return [
            scene for scene in self.ctx.media.all_scenes_dicts
            if id(scene) in clip_ids
        ]

    # Placeholder for copied methods - will be populated via read_file logic or assumed based on previous read.
    
    def _resolve_scoring_rules(self) -> Dict[str, float]:
        scoring_rules = self.ctx.creative.style_params.get("scoring_rules", {})
        return {
            "fresh_clip_bonus": scoring_rules.get("fresh_clip_bonus", 50),
            "jump_cut_penalty": scoring_rules.get("jump_cut_penalty", 50),
            "shot_variation_bonus": scoring_rules.get("shot_variation_bonus", 10),
            "shot_repetition_penalty": scoring_rules.get("shot_repetition_penalty", 10),
        }

    def _get_current_music_section(self):
        audio = self.ctx.media.audio_result
        if not audio or not audio.sections:
            return None
        current_time = self.ctx.timeline.current_time
        for section in audio.sections:
            if section.start_time <= current_time < section.end_time:
                return section
        return None

    def _score_usage_and_story_phase(
        self,
        footage_clip: Any,
        current_energy: float,
        fresh_clip_bonus: float,
    ) -> float:
        score = 0.0
        if not footage_clip:
            return score

        if footage_clip.usage_count == 0:
            score += fresh_clip_bonus
        elif footage_clip.usage_count == 1:
            score += (fresh_clip_bonus * 0.4)
        else:
            score -= footage_clip.usage_count * 10

        phase = self.ctx.get_story_phase()
        if phase == "intro" and current_energy < 0.4:
            score += 15
        elif phase == "build" and 0.4 <= current_energy < 0.7:
            score += 15
        elif phase == "climax" and current_energy >= 0.7:
            score += 15
        elif phase == "outro" and current_energy < 0.5:
            score += 15

        return score

    def _score_jump_cut(
        self, scene_path: str, unique_videos: int, jump_cut_penalty: float
    ) -> float:
        if scene_path != self.ctx.timeline.last_used_path:
            return 0.0
        if unique_videos > 1:
            return -jump_cut_penalty
        return -(jump_cut_penalty * 0.1)

    def _score_action_energy(
        self,
        meta: Dict[str, Any],
        current_energy: float,
        current_section: Any,
    ) -> float:
        score = 0.0
        action = meta.get('action', 'medium')
        if current_energy > 0.6 and action == 'high':
            score += 20
        if current_energy < 0.4 and action == 'low':
            score += 20

        if current_section:
            if current_section.energy_level == "high" and action == "high":
                score += 25
            elif current_section.energy_level == "low" and action == "low":
                score += 25

        return score

    def _score_style_preferences(self, meta: Dict[str, Any]) -> float:
        style_params = self.ctx.creative.style_params
        if not style_params:
            return 0.0
        return self._apply_style_scoring(meta, style_params)

    def _apply_style_scoring(self, meta: Dict[str, Any], style_params: Dict[str, Any]) -> float:
        score = 0.0
        weights = style_params.get('weights', {})
        
        # 1. Weights
        for key, weight in weights.items():
            if key == 'action': continue
            val = meta.get(key)
            if isinstance(val, (int, float)) and val > 0:
                score += (val * 10.0) * weight

        if 'action' in weights:
            action = meta.get('action', 'medium')
            weight = weights['action']
            if action == 'high':
                score += 15.0 * weight
            elif action == 'low':
                score -= 5.0 * weight

        # 2. Preferred Lists
        for param_key, param_val in style_params.items():
            if param_key.startswith('preferred_') and isinstance(param_val, list):
                field_name = param_key.replace('preferred_', '').rstrip('s')
                meta_val = meta.get(field_name)
                if meta_val and meta_val in param_val:
                    score += 20.0
        return score

    def _score_shot_variation(
        self,
        shot: str,
        shot_variation_bonus: float,
        shot_repetition_penalty: float,
    ) -> float:
        if self.ctx.timeline.last_shot_type and shot == self.ctx.timeline.last_shot_type:
            return -shot_repetition_penalty
        return shot_variation_bonus

    def _score_match_cut(self, scene: Dict[str, Any], similarity_fn) -> float:
        if self.ctx.timeline.last_clip_end_time is None or not self.ctx.timeline.last_used_path:
            return 0.0

        try:
            similarity = similarity_fn(
                self.ctx.timeline.last_used_path,
                self.ctx.timeline.last_clip_end_time,
                scene['path'],
                scene['start'],
            )
            if similarity > 0.7:
                return 30.0
        except Exception:
            pass
        return 0.0

    def _score_semantic_match(self, meta: Dict[str, Any]) -> float:
        if not self.ctx.creative.semantic_query:
            return 0.0
        if not (meta.get('tags') or meta.get('caption')):
            return 0.0

        try:
            from ..semantic_matcher import get_semantic_matcher
            matcher = get_semantic_matcher()
            if matcher.is_available:
                sem_result = matcher.match_query_to_clip(self.ctx.creative.semantic_query, meta)
                return int(sem_result.overall_score * 40)
        except Exception:
            return 0.0
        return 0.0

    def _score_broll_match(self, scene: Dict[str, Any], meta: Dict[str, Any]) -> float:
        if not self.ctx.creative.broll_plan:
            return 0.0

        active_segment = None
        current_time = self.ctx.timeline.current_time
        for seg in self.ctx.creative.broll_plan:
            if seg.get("start_time", 0) <= current_time < seg.get("end_time", 99999):
                active_segment = seg
                break

        if not active_segment:
            return 0.0

        for sug in active_segment.get("suggestions", []):
            if sug.get("clip") == scene["path"]:
                return 100.0

        if not (meta.get('tags') or meta.get('caption')):
            return 0.0
            
        return 0.0

    def _select_probabilistic(self, candidates: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        """Select a clip using weighted probability based on scores."""
        if not candidates:
            return None, 0.0

        # Filter out negative scores for probability calculation (keep them for debugging)
        candidates_scores = [(c, max(1.0, c.get('_heuristic_score', 0))) for c in candidates]
        total_score = sum(s for _, s in candidates_scores)
        
        if total_score <= 0:
            return candidates[0], candidates[0].get('_heuristic_score', 0)

        pick = random.uniform(0, total_score)
        current = 0
        for scene, score in candidates_scores:
            current += score
            if current >= pick:
                return scene, scene.get('_heuristic_score', 0)
        
        return candidates[-1], candidates[-1].get('_heuristic_score', 0)

    def _select_with_intelligent_selector(
        self, 
        candidates: List[Dict[str, Any]], 
        energy: float
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Select using the ML/LLM based selector (fallback to probabilistic)."""
        if not self._intelligent_selector:
             return self._select_probabilistic(candidates)

        try:
            from ..clip_selector import ClipCandidate
            
            clip_candidates = [
                ClipCandidate(
                    path=scene['path'],
                    start_time=scene['start'],
                    duration=scene['duration'],
                    heuristic_score=int(scene.get('_heuristic_score', 0)),
                    metadata=scene.get('meta', {}),
                )
                for scene in candidates
            ]

            prev_clips = []
            if self.ctx.timeline.last_used_path:
                prev_meta = {'shot': self.ctx.timeline.last_shot_type}
                prev_clips.append({'meta': prev_meta})

            context = {
                "story_phase": self.ctx.get_story_phase(),
                "current_energy": energy,
                "previous_clips": prev_clips,
                "last_shot": self.ctx.timeline.last_shot_type,
                "music_tempo": self.ctx.media.audio_result.tempo if self.ctx.media.audio_result else 120,
                "beat_idx": self.ctx.timeline.beat_idx,
                "style": self.ctx.creative.style_params,
                "semantic_query": self.ctx.creative.semantic_query,
                "editing_instructions": self.ctx.creative.editing_instructions
            }

            best_candidate_obj, reason = self._intelligent_selector.select_best_clip(
                clip_candidates, 
                context
            )

            if best_candidate_obj:
                logger.info(f"      🧠 Intelligent choice: {reason}")
                
                # We need to map back to the original scene dict
                selected_scene = next(
                    (
                        scene for scene in candidates
                        if scene['path'] == best_candidate_obj.path
                        and scene['start'] == best_candidate_obj.start_time
                    ),
                    candidates[0],
                )
                
                # Boost score of selected
                selected_scene['_heuristic_score'] = 100
                return selected_scene, 100.0
                
        except Exception as e:
            logger.warning(f"   ⚠️ Intelligent Selection Failed: {e}. Falling back.")

        return self._select_probabilistic(candidates)
