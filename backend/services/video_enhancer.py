"""
Video Enhancer: Enhances manual steps with detailed sub-steps from video analysis.

This service takes video analysis results and generates video-enhanced.json with:
- Hierarchical sub-steps (1.1, 1.2, 1.3) breaking down each manual step
- Natural language spatial descriptions ("bottom left corner, 3 studs from left edge")
- Error corrections when video contradicts the 2D image-based manual
"""

import io
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from PIL import Image as PILImage

from ingestion.vlm_extractor import VLMExtractor, _parse_json
from backend.services.data_service import DataService
from config.settings import Settings


def _resize_image(image_path: str, width: int = 800) -> PILImage.Image:
    """Resize image to specified width, preserving aspect ratio."""
    with PILImage.open(image_path) as img:
        orig_w, orig_h = img.size
        new_h = int(width * orig_h / orig_w)
        return img.resize((width, new_h), PILImage.Resampling.LANCZOS)


def _image_to_b64(img: PILImage.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VideoEnhancer:
    """Enhances assembly instructions with detailed sub-steps from video analysis."""

    def __init__(
        self,
        vlm_extractor: VLMExtractor,
        data_service: DataService,
        settings: Settings
    ):
        """
        Initialize video enhancer.

        Args:
            vlm_extractor: VLM extractor for making VLM calls
            data_service: Data service for loading manual and video data
            settings: Application settings
        """
        self.vlm = vlm_extractor
        self.data_service = data_service
        self.settings = settings

        # Load prompt templates
        prompts_dir = Path("prompts")
        with open(prompts_dir / "video_action_segmentation.txt", "r") as f:
            self.action_segmentation_template = f.read()
        with open(prompts_dir / "video_spatial_extraction.txt", "r") as f:
            self.spatial_extraction_template = f.read()
        with open(prompts_dir / "video_action_description.txt", "r") as f:
            self.action_description_template = f.read()
        with open(prompts_dir / "video_conflict_resolution.txt", "r") as f:
            self.conflict_resolution_template = f.read()

    async def enhance_manual_with_video(
        self,
        manual_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """
        Generate video-enhanced manual from video analysis.

        This is the main orchestration method that:
        1. Loads enhanced.json and video_analysis.json
        2. For each manual step:
           - Gets frame sequence
           - Segments into micro-actions
           - Extracts spatial info for placements
           - Generates descriptions
           - Detects conflicts
        3. Builds video_enhanced.json structure
        4. Returns the enhanced data

        Args:
            manual_id: Manual identifier
            video_id: Video identifier (must have completed analysis)

        Returns:
            Complete video_enhanced.json structure
        """
        logger.info(f"Starting video enhancement for manual {manual_id}, video {video_id}")

        # Load source data
        try:
            manual_data = self.data_service.get_steps(manual_id)
            logger.info(f"Loaded manual data: {len(manual_data['steps'])} steps")
        except Exception as e:
            logger.error(f"Failed to load manual data: {e}")
            raise ValueError(f"Manual {manual_id} not found: {e}")

        try:
            video_analysis = self.data_service.get_video_analysis(manual_id, video_id)
            logger.info(f"Loaded video analysis: {video_analysis.get('total_frames_extracted', 0)} frames")
        except Exception as e:
            logger.error(f"Failed to load video analysis: {e}")
            raise ValueError(f"Video analysis for {video_id} not found: {e}")

        # Build enhanced steps
        enhanced_steps = []

        for step in manual_data["steps"]:
            step_num = step["step_number"]
            logger.info(f"Processing step {step_num}")

            try:
                # Get frame sequence for this step
                frame_sequence = self._get_frames_for_step(step_num, video_analysis)

                if not frame_sequence:
                    logger.warning(f"No frames found for step {step_num}, using original step")
                    # Add step without sub-steps
                    enhanced_step = {**step, "sub_steps": [], "corrections": []}
                    enhanced_steps.append(enhanced_step)
                    continue

                logger.debug(f"  Found {len(frame_sequence)} frames for step {step_num}")

                # Segment into micro-actions
                micro_actions = self._segment_step_actions(step, frame_sequence, video_analysis)
                logger.debug(f"  Segmented into {len(micro_actions)} micro-actions")

                # Generate sub-steps from micro-actions
                sub_steps = []
                for i, action in enumerate(micro_actions):
                    sub_step_num = f"{step_num}.{i + 1}"
                    logger.debug(f"  Generating sub-step {sub_step_num}: {action['action_type']}")

                    # Extract spatial info if placement action
                    spatial_desc = None
                    if action["action_type"] in ["PLACE", "ATTACH"]:
                        spatial_desc = self._extract_spatial_placement(
                            action,
                            frame_sequence,
                            video_analysis
                        )

                    # Generate detailed description
                    description = self._generate_action_description(
                        action,
                        spatial_desc
                    )

                    sub_step = {
                        "sub_step_number": sub_step_num,
                        "action_type": action["action_type"].lower(),
                        "description": description,
                        "parts_involved": action["parts_involved"],
                        "spatial_description": spatial_desc,
                        "frame_range": {
                            "start_frame": action["start_frame"],
                            "end_frame": action["end_frame"],
                            "start_time": frame_sequence[0]["timestamp_seconds"] if frame_sequence else 0.0,
                            "end_time": frame_sequence[-1]["timestamp_seconds"] if frame_sequence else 0.0,
                        },
                        "confidence": action["confidence"]
                    }
                    sub_steps.append(sub_step)

                # Detect conflicts with manual
                corrections = self._detect_conflicts(step, frame_sequence, video_analysis)
                if corrections:
                    logger.info(f"  Detected {len(corrections)} corrections for step {step_num}")

                # Build enhanced step
                enhanced_step = {
                    **step,
                    "original_manual_step": step_num,
                    "sub_steps": sub_steps,
                    "corrections": corrections
                }
                enhanced_steps.append(enhanced_step)

            except Exception as e:
                logger.error(f"Failed to enhance step {step_num}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Add step without enhancements on error
                enhanced_step = {**step, "sub_steps": [], "corrections": []}
                enhanced_steps.append(enhanced_step)

        # Build video-enhanced manual structure
        video_meta = video_analysis.get("video_metadata", {})
        enhanced_manual = {
            "manual_id": manual_id,
            "source_video_id": video_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "video_metadata": {
                "duration_seconds": video_meta.get("duration_seconds", 0.0),
                "frame_count": video_analysis.get("total_frames_extracted", 0),
                "filename": video_meta.get("filename", "unknown.mp4")
            },
            "steps": enhanced_steps,
            "manual_step_mapping": self._build_step_mapping(manual_data["steps"], enhanced_steps)
        }

        logger.info(f"Video enhancement complete: {len(enhanced_steps)} steps enhanced")
        return enhanced_manual

    def _get_frames_for_step(
        self,
        step_number: int,
        video_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract frame sequence for a specific step.

        Uses step_timeline to find frame range, then returns those frames
        from frame_analyses array.

        Args:
            step_number: Step number to get frames for
            video_analysis: Complete video analysis data

        Returns:
            List of frame analysis dicts for this step
        """
        # Find this step in timeline
        step_timeline = video_analysis.get("step_timeline", [])
        timeline_entry = None
        for entry in step_timeline:
            if entry["step_number"] == step_number:
                timeline_entry = entry
                break

        if not timeline_entry:
            logger.debug(f"Step {step_number} not found in video timeline")
            return []

        # Get frame numbers for this step
        frame_numbers = set(timeline_entry["frame_numbers"])

        # Extract those frames from frame_analyses
        frame_analyses = video_analysis.get("frame_analyses", [])
        frames = [
            frame for frame in frame_analyses
            if frame["frame_number"] in frame_numbers
        ]

        return frames

    def _segment_step_actions(
        self,
        step: Dict[str, Any],
        frame_sequence: List[Dict[str, Any]],
        video_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Segment a manual step into micro-actions.

        Uses VLM with video_action_segmentation.txt prompt to identify
        action boundaries (pick, place, attach, etc.) in the frame sequence.

        Args:
            step: Manual step data
            frame_sequence: Frames for this step
            video_analysis: Complete video analysis (for frame paths)

        Returns:
            List of MicroAction dicts with:
                - action_type: PICK, PLACE, ATTACH, ADJUST, VERIFY
                - start_frame, end_frame: Frame numbers
                - parts_involved: List of part descriptions
                - description: Brief description
                - confidence: 0.0-1.0
        """
        if not frame_sequence:
            return []

        # Build step description for prompt
        step_desc = ", ".join(step.get("actions", []))
        parts_list = [p["description"] for p in step.get("parts_required", [])]

        # Build prompt
        prompt = self.action_segmentation_template.replace(
            "{step_number}", str(step["step_number"])
        ).replace(
            "{step_description}", step_desc
        ).replace(
            "{parts_list}", ", ".join(parts_list)
        ).replace(
            "{n_frames}", str(len(frame_sequence))
        )

        # Load frame images
        content = [{"type": "text", "text": "FRAME SEQUENCE:"}]

        # Get video directory from analysis
        video_id = video_analysis.get("video_id", "")
        manual_id = video_analysis.get("manual_id", "")
        videos_dir = self.settings.data_dir / "videos" / manual_id

        # Add frame images
        for frame in frame_sequence:
            frame_num = frame["frame_number"]
            timestamp = frame["timestamp_seconds"]

            # Construct frame path
            frame_path = videos_dir / f"{video_id}_frames" / f"frame_{frame_num:06d}.jpg"

            if frame_path.exists():
                frame_img = _resize_image(str(frame_path))
                frame_b64 = _image_to_b64(frame_img)
                content.append({
                    "type": "text",
                    "text": f"Frame {frame_num} (t={timestamp:.2f}s):"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame_b64}"}
                })

        content.append({"type": "text", "text": prompt})

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        try:
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)
            micro_actions = result.get("micro_actions", [])
            logger.debug(f"Segmentation found {len(micro_actions)} micro-actions")
            return micro_actions
        except Exception as e:
            logger.error(f"Action segmentation failed: {e}")
            return []

    def _extract_spatial_placement(
        self,
        action: Dict[str, Any],
        frame_sequence: List[Dict[str, Any]],
        video_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract spatial placement description from placement action.

        Uses VLM with video_spatial_extraction.txt prompt to analyze
        placement frames and extract stud-level spatial information.

        Args:
            action: MicroAction dict (PLACE or ATTACH type)
            frame_sequence: All frames for this step
            video_analysis: Complete video analysis (for frame paths)

        Returns:
            SpatialDescription dict with:
                - target_part: What you're placing onto
                - placement_part: What you're placing
                - location: General area
                - position_detail: Stud-level precision
                - orientation: horizontal/vertical/null
                - relative_to: Nearby part reference or null
            Or None if spatial info cannot be extracted
        """
        parts = action.get("parts_involved", [])
        if len(parts) < 2:
            # Need at least placement part and target part
            return None

        placement_part = parts[0]
        target_part = parts[1] if len(parts) > 1 else "baseplate"

        # Build prompt
        prompt = self.spatial_extraction_template.replace(
            "{placement_part}", placement_part
        ).replace(
            "{target_part}", target_part
        )

        # Get frame range for this action
        start_frame = action.get("start_frame", 0)
        end_frame = action.get("end_frame", 0)

        # Get middle frame (clearest view of placement)
        mid_frame_num = (start_frame + end_frame) // 2
        placement_frame = None
        for frame in frame_sequence:
            if frame["frame_number"] == mid_frame_num:
                placement_frame = frame
                break

        if not placement_frame:
            logger.debug("No placement frame found for spatial extraction")
            return None

        # Load frame image
        video_id = video_analysis.get("video_id", "")
        manual_id = video_analysis.get("manual_id", "")
        videos_dir = self.settings.data_dir / "videos" / manual_id
        frame_path = videos_dir / f"{video_id}_frames" / f"frame_{placement_frame['frame_number']:06d}.jpg"

        if not frame_path.exists():
            logger.debug(f"Frame image not found: {frame_path}")
            return None

        # Build multi-image content
        frame_img = _resize_image(str(frame_path))
        frame_b64 = _image_to_b64(frame_img)

        content = [
            {"type": "text", "text": f"Placement frame (t={placement_frame['timestamp_seconds']:.2f}s):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
            {"type": "text", "text": prompt}
        ]

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        try:
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            # Check confidence
            if result.get("confidence", 0.0) < 0.5:
                logger.debug("Low confidence spatial extraction, skipping")
                return None

            # Return spatial description if location was extracted
            if result.get("location"):
                return {
                    "target_part": target_part,
                    "placement_part": placement_part,
                    "location": result["location"],
                    "position_detail": result.get("position_detail", ""),
                    "orientation": result.get("orientation"),
                    "relative_to": result.get("relative_to")
                }

            return None

        except Exception as e:
            logger.error(f"Spatial extraction failed: {e}")
            return None

    def _generate_action_description(
        self,
        action: Dict[str, Any],
        spatial: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate natural language description from action + spatial data.

        Uses VLM with video_action_description.txt prompt to create
        a clear, concise instruction sentence.

        Args:
            action: MicroAction dict
            spatial: SpatialDescription dict or None

        Returns:
            Description string like "Place red round tile on grey baseplate at bottom left corner"
        """
        action_type = action.get("action_type", "PLACE")
        parts = action.get("parts_involved", [])

        # Build prompt
        prompt = self.action_description_template.replace(
            "{action_type}", action_type
        ).replace(
            "{parts_list}", ", ".join(parts)
        ).replace(
            "{spatial_json}", json.dumps(spatial) if spatial else "null"
        )

        # Make VLM call
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)
            description = result.get("description", action.get("description", ""))
            return description
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            # Fallback to simple description
            return action.get("description", f"{action_type} {parts[0] if parts else 'part'}")

    def _detect_conflicts(
        self,
        manual_step: Dict[str, Any],
        frame_sequence: List[Dict[str, Any]],
        video_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between manual and video evidence.

        Compares manual parts_required vs video-detected parts. Uses VLM
        with video_conflict_resolution.txt prompt to resolve discrepancies.

        Args:
            manual_step: Step from enhanced.json
            frame_sequence: Frames for this step
            video_analysis: Complete video analysis

        Returns:
            List of Correction dicts with:
                - field: Field name (e.g., "parts_required")
                - original_value: Value from manual
                - corrected_value: Correct value from video
                - reason: Explanation
                - confidence: 0.0-1.0
        """
        # Get manual parts
        manual_parts = {p["description"] for p in manual_step.get("parts_required", [])}

        # Get video-detected parts for this step
        video_parts = set()
        for frame in frame_sequence:
            video_parts.update(frame.get("detected_parts", []))

        # Check for conflicts
        if manual_parts == video_parts:
            logger.debug("No conflicts detected between manual and video")
            return []

        missing_in_video = manual_parts - video_parts
        extra_in_video = video_parts - manual_parts

        if not missing_in_video and not extra_in_video:
            return []

        logger.debug(f"Potential conflict: manual={manual_parts}, video={video_parts}")

        # Build prompt for conflict resolution
        step_num = manual_step["step_number"]
        frame_range = f"{frame_sequence[0]['frame_number']}-{frame_sequence[-1]['frame_number']}"
        time_range = f"{frame_sequence[0]['timestamp_seconds']:.1f}s-{frame_sequence[-1]['timestamp_seconds']:.1f}s"

        # Calculate confidence scores
        confidence_scores = {}
        for frame in frame_sequence:
            for part in frame.get("detected_parts", []):
                if part in video_parts:
                    confidence_scores[part] = max(
                        confidence_scores.get(part, 0.0),
                        frame.get("parts_confidence", 0.0)
                    )

        prompt = self.conflict_resolution_template.replace(
            "{step_number}", str(step_num)
        ).replace(
            "{manual_parts}", ", ".join(manual_parts)
        ).replace(
            "{video_parts}", ", ".join(video_parts)
        ).replace(
            "{frame_range}", frame_range
        ).replace(
            "{time_range}", time_range
        ).replace(
            "{confidence_scores}", json.dumps(confidence_scores)
        )

        # Make VLM call
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            if result.get("conflict_exists") and result.get("correct_source") == "video":
                # Build correction
                corrections = []
                for incorrect_part in result.get("incorrect_parts", []):
                    if incorrect_part in missing_in_video:
                        # Part in manual but not in video
                        correct_part = list(extra_in_video)[0] if extra_in_video else None
                        if correct_part:
                            corrections.append({
                                "field": "parts_required",
                                "original_value": incorrect_part,
                                "corrected_value": correct_part,
                                "reason": result.get("reasoning", "Video evidence contradicts manual"),
                                "confidence": result.get("confidence", 0.0)
                            })

                return corrections

        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")

        return []

    def _build_step_mapping(
        self,
        manual_steps: List[Dict[str, Any]],
        enhanced_steps: List[Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """
        Build mapping from manual step numbers to enhanced step numbers.

        Currently 1:1 mapping since we're enhancing each step individually.
        Future: Could split manual steps if video shows more detail.

        Args:
            manual_steps: Original manual steps
            enhanced_steps: Enhanced steps with sub-steps

        Returns:
            Dict mapping manual step number (str) to list of enhanced step numbers
        """
        mapping = {}
        for manual_step, enhanced_step in zip(manual_steps, enhanced_steps):
            manual_num = str(manual_step["step_number"])
            enhanced_num = enhanced_step["step_number"]
            mapping[manual_num] = [enhanced_num]

        return mapping
