"""
Video Enhancer: Enhances manual steps with detailed sub-steps from video analysis.

This service uses exactly 3 VLM call types:
1. Action Detection: For each frame, detect action type and parts involved
2. Spatial Extraction: For placement frames, extract spatial brick placement info
3. Reconciliation: Reconcile data with manual and generate corrections

Outputs video_enhanced.json with hierarchical sub-steps and spatial descriptions.
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

        # VLM Call Type 1: Action Detection
        action_detection_prompt = prompts_dir / "video_action_detection.txt"
        if action_detection_prompt.exists():
            with open(action_detection_prompt, "r") as f:
                self.action_detection_template = f.read()
        else:
            self.action_detection_template = self._get_default_action_detection_prompt()

        # VLM Call Type 2: Spatial Extraction
        spatial_extraction_prompt = prompts_dir / "video_spatial_extraction.txt"
        if spatial_extraction_prompt.exists():
            with open(spatial_extraction_prompt, "r") as f:
                self.spatial_extraction_template = f.read()
        else:
            self.spatial_extraction_template = self._get_default_spatial_extraction_prompt()

        # VLM Call Type 3: Reconciliation
        reconciliation_prompt = prompts_dir / "video_reconciliation.txt"
        if reconciliation_prompt.exists():
            with open(reconciliation_prompt, "r") as f:
                self.reconciliation_template = f.read()
        else:
            self.reconciliation_template = self._get_default_reconciliation_prompt()

    async def enhance_manual_with_video(
        self,
        manual_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """
        Generate video-enhanced manual from entire video using 3 VLM call types.

        Flow:
        1. Extract ALL frames from video (every 30 frames, no limit)
        2. VLM Call Type 1: For each frame, detect action and parts
        3. VLM Call Type 2: For placement frames, extract spatial info
        4. VLM Call Type 3: Reconcile with manual and generate corrections
        5. Build video_enhanced.json structure

        Args:
            manual_id: Manual identifier
            video_id: Video identifier

        Returns:
            Complete video_enhanced.json structure
        """
        logger.info(f"Starting video enhancement for manual {manual_id}, video {video_id}")

        # Load manual data
        try:
            manual_data = self.data_service.get_steps(manual_id)
            logger.info(f"Loaded manual data: {len(manual_data['steps'])} steps")
        except Exception as e:
            logger.error(f"Failed to load manual data: {e}")
            raise ValueError(f"Manual {manual_id} not found: {e}")

        # Extract all frames from video
        logger.info("Extracting all frames from video...")
        frames = await self._extract_all_frames(manual_id, video_id)
        logger.info(f"Extracted {len(frames)} frames for enhancement")

        # === VLM CALL TYPE 1: Action Detection ===
        # For each frame, detect what action is happening
        logger.info("Running VLM Call Type 1: Action Detection for all frames...")
        frame_actions = await self._detect_actions_for_all_frames(frames, manual_id, video_id)
        logger.info(f"Detected actions for {len(frame_actions)} frames")

        # Group frames by manual step
        frames_by_step = self._group_frames_by_step(frame_actions, manual_data)

        # Build enhanced steps
        enhanced_steps = []

        for step in manual_data["steps"]:
            step_num = step["step_number"]
            logger.info(f"Processing step {step_num}")

            step_frames = frames_by_step.get(step_num, [])

            if not step_frames:
                logger.warning(f"No frames found for step {step_num}, using original step")
                enhanced_step = {**step, "sub_steps": [], "corrections": []}
                enhanced_steps.append(enhanced_step)
                continue

            # Segment frames into sub-steps based on action changes
            sub_step_segments = self._segment_into_substeps(step_frames)
            logger.info(f"  Segmented into {len(sub_step_segments)} sub-steps")
            for i, seg in enumerate(sub_step_segments):
                logger.info(
                    f"    Sub-step {step_num}.{i+1}: "
                    f"action={seg['action_type']}, "
                    f"frames={seg['start_frame']}-{seg['end_frame']}, "
                    f"parts={seg['parts_involved']}"
                )

            # === VLM CALL TYPE 2: Spatial Extraction ===
            # For each placement sub-step, extract spatial information
            sub_steps = []
            for i, segment in enumerate(sub_step_segments):
                sub_step_num = f"{step_num}.{i + 1}"
                logger.info(f"  Processing sub-step {sub_step_num}: {segment['action_type']}")

                spatial_desc = None
                if segment["action_type"] in ["place", "attach"]:
                    logger.info(f"    🎯 Placement sub-step detected! Running spatial extraction...")
                    # Run VLM Call Type 2: Spatial Extraction
                    spatial_desc = await self._extract_spatial_info(
                        segment, manual_id, video_id
                    )

                # Build description from action and spatial info
                description = self._build_description(segment, spatial_desc)

                sub_step = {
                    "sub_step_number": sub_step_num,
                    "action_type": segment["action_type"],
                    "description": description,
                    "parts_involved": segment["parts_involved"],
                    "spatial_description": spatial_desc,
                    "frame_range": {
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"]
                    },
                    "confidence": segment["confidence"]
                }
                sub_steps.append(sub_step)

            # === VLM CALL TYPE 3: Reconciliation ===
            # Reconcile video data with manual and detect corrections
            logger.info(f"  Running VLM Call Type 3: Reconciliation for step {step_num}")
            corrections = await self._reconcile_with_manual(
                step, step_frames, manual_id, video_id
            )
            if corrections:
                logger.info(f"  Found {len(corrections)} corrections for step {step_num}")

            # Build enhanced step
            enhanced_step = {
                **step,
                "original_manual_step": step_num,
                "sub_steps": sub_steps,
                "corrections": corrections
            }
            enhanced_steps.append(enhanced_step)

        # Build video-enhanced manual structure
        enhanced_manual = {
            "manual_id": manual_id,
            "source_video_id": video_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "video_metadata": {
                "duration_seconds": 0.0,  # Will be filled from metadata
                "frame_count": len(frames),
                "filename": f"{video_id}.mp4"
            },
            "steps": enhanced_steps,
            "manual_step_mapping": self._build_step_mapping(manual_data["steps"], enhanced_steps)
        }

        logger.info(f"Video enhancement complete: {len(enhanced_steps)} steps enhanced")
        return enhanced_manual

    async def _extract_all_frames(
        self,
        manual_id: str,
        video_id: str
    ) -> List[Path]:
        """Extract all frames from video (no limit)."""
        from backend.services.video_processor import VideoProcessor

        video_path = self.settings.data_dir / "videos" / manual_id / f"{video_id}.mp4"

        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        video_processor = VideoProcessor(self.settings)
        frames_dir = video_path.parent / f"{video_id}_enhancement_frames"
        frames_dir.mkdir(exist_ok=True)

        logger.info("Extracting frames from entire video (no frame limit)...")
        frames = video_processor.extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            frame_interval=30,  # Extract every 30th frame for detailed enhancement
            max_frames=None  # NO LIMIT - process entire video
        )

        return [Path(f["frame_path"]) for f in frames]

    async def _detect_actions_for_all_frames(
        self,
        frames: List[Path],
        manual_id: str,
        video_id: str
    ) -> List[Dict[str, Any]]:
        """
        VLM CALL TYPE 1: Action Detection

        For each frame, detect what action is happening and what parts are involved.

        Returns list of frame action dicts:
        {
            "frame_number": int,
            "frame_path": Path,
            "timestamp": float,
            "action_type": "pick" | "place" | "attach" | "adjust" | "verify" | "none",
            "parts_involved": [str],
            "confidence": float,
            "is_relevant": bool  # False for title screens, etc.
        }
        """
        logger.info(f"Running action detection on {len(frames)} frames...")

        frame_actions = []

        for i, frame_path in enumerate(frames):
            # Extract frame number from filename (e.g., "frame_000123.jpg" -> 123)
            frame_num = int(frame_path.stem.split('_')[-1])

            # Calculate timestamp (assuming 30 fps and frame_interval=30, so 1 frame per second)
            timestamp = frame_num / 30.0

            try:
                # Load and resize frame
                frame_img = _resize_image(str(frame_path), width=600)
                frame_b64 = _image_to_b64(frame_img)

                # Build prompt for action detection
                prompt = self.action_detection_template.replace(
                    "{frame_number}", str(frame_num)
                ).replace(
                    "{timestamp}", f"{timestamp:.2f}"
                )

                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
                    {"type": "text", "text": prompt}
                ]

                messages = [{"role": "user", "content": content}]

                # Make VLM call
                raw = self.vlm._litellm_with_retry(messages)
                result = _parse_json(raw)

                frame_action = {
                    "frame_number": frame_num,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "action_type": result.get("action_type", "none").lower(),
                    "parts_involved": result.get("parts_involved", []),
                    "confidence": result.get("confidence", 0.0),
                    "is_relevant": result.get("is_relevant", True)
                }

                # Log detailed VLM output for action detection
                logger.info(
                    f"  Frame {frame_num} (t={timestamp:.1f}s): "
                    f"action={frame_action['action_type']}, "
                    f"parts={frame_action['parts_involved']}, "
                    f"confidence={frame_action['confidence']:.2f}, "
                    f"relevant={frame_action['is_relevant']}"
                )
                if frame_action['action_type'] in ['place', 'attach']:
                    logger.warning(f"    ⭐ PLACEMENT FRAME DETECTED: Frame {frame_num}")

                frame_actions.append(frame_action)

                if (i + 1) % 50 == 0:
                    logger.info(f"  === Processed {i + 1}/{len(frames)} frames ===")

            except Exception as e:
                logger.error(f"Failed to detect action for frame {frame_num}: {e}")
                # Add placeholder for failed frame
                frame_actions.append({
                    "frame_number": frame_num,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "action_type": "none",
                    "parts_involved": [],
                    "confidence": 0.0,
                    "is_relevant": True
                })

        # Filter out irrelevant frames
        relevant_frames = [f for f in frame_actions if f["is_relevant"]]
        irrelevant_count = len(frame_actions) - len(relevant_frames)
        logger.info(f"Filtered to {len(relevant_frames)} relevant frames ({irrelevant_count} irrelevant frames removed)")

        # Log summary of placement frames
        placement_frames = [f for f in relevant_frames if f["action_type"] in ["place", "attach"]]
        logger.info(f"📊 SUMMARY: {len(placement_frames)} PLACEMENT FRAMES detected out of {len(relevant_frames)} relevant frames")
        for pf in placement_frames[:10]:  # Show first 10
            logger.info(f"   Frame {pf['frame_number']} (t={pf['timestamp']:.1f}s): {pf['action_type']} - {pf['parts_involved']}")
        if len(placement_frames) > 10:
            logger.info(f"   ... and {len(placement_frames) - 10} more placement frames")

        return relevant_frames

    def _group_frames_by_step(
        self,
        frame_actions: List[Dict[str, Any]],
        manual_data: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group frames by manual step number based on parts involved.

        Matches parts detected in frames to parts required in manual steps.
        """
        frames_by_step = {}

        for step in manual_data["steps"]:
            step_num = step["step_number"]
            step_parts = {p["description"].lower() for p in step.get("parts_required", [])}

            # Find frames that involve parts from this step
            step_frames = []
            for frame_action in frame_actions:
                frame_parts = {p.lower() for p in frame_action["parts_involved"]}

                # Check if any frame parts match step parts
                if frame_parts & step_parts or not frame_action["parts_involved"]:
                    step_frames.append(frame_action)

            frames_by_step[step_num] = step_frames

        return frames_by_step

    def _segment_into_substeps(
        self,
        step_frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Segment frames into sub-steps based on action type changes.

        Returns list of sub-step segments:
        {
            "action_type": str,
            "parts_involved": [str],
            "start_frame": int,
            "end_frame": int,
            "start_time": float,
            "end_time": float,
            "frames": [Dict],
            "confidence": float
        }
        """
        if not step_frames:
            return []

        segments = []
        current_segment = None

        for frame in step_frames:
            action_type = frame["action_type"]

            # Start new segment if action type changes or no current segment
            if current_segment is None or current_segment["action_type"] != action_type:
                if current_segment is not None:
                    segments.append(current_segment)

                current_segment = {
                    "action_type": action_type,
                    "parts_involved": frame["parts_involved"],
                    "start_frame": frame["frame_number"],
                    "end_frame": frame["frame_number"],
                    "start_time": frame["timestamp"],
                    "end_time": frame["timestamp"],
                    "frames": [frame],
                    "confidence": frame["confidence"]
                }
            else:
                # Continue current segment
                current_segment["end_frame"] = frame["frame_number"]
                current_segment["end_time"] = frame["timestamp"]
                current_segment["frames"].append(frame)
                # Average confidence
                current_segment["confidence"] = (
                    current_segment["confidence"] + frame["confidence"]
                ) / 2
                # Accumulate unique parts
                current_segment["parts_involved"] = list(set(
                    current_segment["parts_involved"] + frame["parts_involved"]
                ))

        # Add last segment
        if current_segment is not None:
            segments.append(current_segment)

        return segments

    async def _extract_spatial_info(
        self,
        segment: Dict[str, Any],
        manual_id: str,
        video_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        VLM CALL TYPE 2: Spatial Extraction

        For placement actions, extract spatial information about brick placement.

        Returns:
        {
            "target_part": str,  # What you're placing onto
            "placement_part": str,  # What you're placing
            "location": str,  # General area (e.g., "top left corner")
            "position_detail": str,  # Stud-level precision (e.g., "2 studs from left edge")
            "orientation": str | null,  # "horizontal", "vertical", null
            "relative_to": str | null  # Nearby part reference
        }
        """
        parts = segment.get("parts_involved", [])
        if not parts:
            return None

        placement_part = parts[0] if parts else "unknown part"
        target_part = parts[1] if len(parts) > 1 else "baseplate"

        # Get middle frame from segment for clearest placement view
        frames = segment.get("frames", [])
        if not frames:
            return None

        mid_frame = frames[len(frames) // 2]
        frame_path = mid_frame["frame_path"]

        try:
            # Load frame
            frame_img = _resize_image(str(frame_path), width=800)
            frame_b64 = _image_to_b64(frame_img)

            # Build prompt
            prompt = self.spatial_extraction_template.replace(
                "{placement_part}", placement_part
            ).replace(
                "{target_part}", target_part
            )

            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
                {"type": "text", "text": prompt}
            ]

            messages = [{"role": "user", "content": content}]

            # Make VLM call
            logger.info(f"    🔍 VLM CALL 2 (Spatial Extraction) for frame {placement_frame['frame_number']}")
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            # Log detailed spatial extraction result
            logger.info(f"    📍 Spatial Extraction Result:")
            logger.info(f"       Location: {result.get('location', 'N/A')}")
            logger.info(f"       Position Detail: {result.get('position_detail', 'N/A')}")
            logger.info(f"       Orientation: {result.get('orientation', 'N/A')}")
            logger.info(f"       Relative To: {result.get('relative_to', 'N/A')}")
            logger.info(f"       Confidence: {result.get('confidence', 0.0):.2f}")

            # Check confidence
            if result.get("confidence", 0.0) < 0.5:
                logger.warning(f"    ⚠️  Low confidence spatial extraction ({result.get('confidence', 0.0):.2f}), skipping")
                return None

            spatial_info = {
                "target_part": target_part,
                "placement_part": placement_part,
                "location": result.get("location", ""),
                "position_detail": result.get("position_detail", ""),
                "orientation": result.get("orientation"),
                "relative_to": result.get("relative_to")
            }

            logger.info(f"    ✅ Spatial extraction successful")
            return spatial_info

        except Exception as e:
            logger.error(f"Spatial extraction failed: {e}")
            return None

    async def _reconcile_with_manual(
        self,
        manual_step: Dict[str, Any],
        step_frames: List[Dict[str, Any]],
        manual_id: str,
        video_id: str
    ) -> List[Dict[str, Any]]:
        """
        VLM CALL TYPE 3: Reconciliation

        Reconcile video evidence with manual data and detect corrections.

        Compares:
        - Parts required in manual vs parts detected in video
        - Actions described in manual vs actions seen in video

        Returns list of corrections:
        {
            "field": str,  # e.g., "parts_required"
            "original_value": any,  # Value from manual
            "corrected_value": any,  # Correct value from video
            "reason": str,  # Explanation
            "confidence": float
        }
        """
        if not step_frames:
            return []

        # Get manual parts
        manual_parts = {p["description"] for p in manual_step.get("parts_required", [])}

        # Get video parts
        video_parts = set()
        for frame in step_frames:
            video_parts.update(frame.get("parts_involved", []))

        # Check if reconciliation is needed
        if manual_parts == video_parts:
            return []

        missing_in_video = manual_parts - video_parts
        extra_in_video = video_parts - manual_parts

        if not missing_in_video and not extra_in_video:
            return []

        logger.info(f"Detected discrepancy: manual={manual_parts}, video={video_parts}")

        try:
            # Get representative frames
            sample_frames = step_frames[::max(1, len(step_frames) // 3)][:3]

            # Build multi-frame content
            content = [{"type": "text", "text": "FRAMES FROM VIDEO:"}]

            for frame in sample_frames:
                frame_path = frame["frame_path"]
                frame_img = _resize_image(str(frame_path), width=600)
                frame_b64 = _image_to_b64(frame_img)

                content.append({
                    "type": "text",
                    "text": f"Frame {frame['frame_number']} (t={frame['timestamp']:.1f}s):"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame_b64}"}
                })

            # Build prompt
            prompt = self.reconciliation_template.replace(
                "{step_number}", str(manual_step["step_number"])
            ).replace(
                "{manual_parts}", ", ".join(manual_parts) if manual_parts else "none"
            ).replace(
                "{video_parts}", ", ".join(video_parts) if video_parts else "none"
            ).replace(
                "{manual_actions}", ", ".join(manual_step.get("actions", []))
            )

            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            # Make VLM call
            logger.info(f"    🔍 VLM CALL 3 (Reconciliation) for step {manual_step['step_number']}")
            logger.info(f"       Manual parts: {manual_parts}")
            logger.info(f"       Video parts: {video_parts}")
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            # Log reconciliation result
            logger.info(f"    🔄 Reconciliation Result:")
            logger.info(f"       Conflict exists: {result.get('conflict_exists', False)}")
            logger.info(f"       Video is correct: {result.get('video_is_correct', False)}")
            logger.info(f"       Incorrect parts: {result.get('incorrect_parts', [])}")
            logger.info(f"       Correct parts: {result.get('correct_parts', [])}")
            logger.info(f"       Reason: {result.get('reason', 'N/A')}")
            logger.info(f"       Confidence: {result.get('confidence', 0.0):.2f}")

            corrections = []

            if result.get("conflict_exists") and result.get("video_is_correct"):
                logger.warning(f"    ⚠️  CORRECTION DETECTED for step {manual_step['step_number']}")
                # Video shows the correct parts
                for incorrect_part in result.get("incorrect_parts", []):
                    if incorrect_part in missing_in_video:
                        # Find corresponding correct part
                        correct_parts = result.get("correct_parts", [])
                        if correct_parts:
                            correction = {
                                "field": "parts_required",
                                "original_value": incorrect_part,
                                "corrected_value": correct_parts[0],
                                "reason": result.get("reason", "Video evidence contradicts manual"),
                                "confidence": result.get("confidence", 0.8)
                            }
                            corrections.append(correction)
                            logger.info(f"       ✏️  Correction: {incorrect_part} → {correct_parts[0]}")

            if not corrections:
                logger.info(f"    ✅ No corrections needed for step {manual_step['step_number']}")

            return corrections

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            return []

    def _build_description(
        self,
        segment: Dict[str, Any],
        spatial: Optional[Dict[str, Any]]
    ) -> str:
        """Build natural language description from action and spatial info."""
        action_type = segment["action_type"]
        parts = segment.get("parts_involved", [])

        if not parts:
            return f"{action_type.capitalize()} action"

        part_desc = parts[0]

        if spatial and action_type in ["place", "attach"]:
            location = spatial.get("location", "")
            position_detail = spatial.get("position_detail", "")
            target = spatial.get("target_part", "baseplate")

            desc = f"{action_type.capitalize()} {part_desc} on {target}"
            if location:
                desc += f" at {location}"
            if position_detail:
                desc += f" ({position_detail})"
            return desc
        else:
            return f"{action_type.capitalize()} {part_desc}"

    def _build_step_mapping(
        self,
        manual_steps: List[Dict[str, Any]],
        enhanced_steps: List[Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """Build mapping from manual step numbers to enhanced step numbers."""
        mapping = {}
        for manual_step, enhanced_step in zip(manual_steps, enhanced_steps):
            manual_num = str(manual_step["step_number"])
            enhanced_num = enhanced_step["step_number"]
            mapping[manual_num] = [enhanced_num]
        return mapping

    def _get_default_action_detection_prompt(self) -> str:
        """Default prompt for action detection."""
        return """Analyze this frame and detect what assembly action is happening.

Return JSON:
{
  "action_type": "pick" | "place" | "attach" | "adjust" | "verify" | "none",
  "parts_involved": ["part description 1", "part description 2"],
  "confidence": 0.0-1.0,
  "is_relevant": true/false,
  "reasoning": "brief explanation"
}

Action types:
- "pick": Picking up a LEGO piece
- "place": Placing a piece on the assembly
- "attach": Attaching/pressing pieces together
- "adjust": Adjusting position of placed pieces
- "verify": Checking or showing the assembly
- "none": No action or irrelevant frame (title screen, etc.)

is_relevant should be false for title screens, text overlays, black screens, etc.
"""

    def _get_default_spatial_extraction_prompt(self) -> str:
        """Default prompt for spatial extraction."""
        return """Analyze this frame showing a LEGO brick placement and extract spatial information.

The frame shows: {placement_part} being placed on {target_part}

Return JSON:
{
  "location": "general area (e.g., 'top left corner', 'center', 'right side')",
  "position_detail": "stud-level precision (e.g., '2 studs from left edge', '3 studs from top')",
  "orientation": "horizontal" | "vertical" | null,
  "relative_to": "nearby part reference" | null,
  "confidence": 0.0-1.0
}

Be as precise as possible about the placement location."""

    def _get_default_reconciliation_prompt(self) -> str:
        """Default prompt for reconciliation."""
        return """Compare the manual instructions with video evidence and detect discrepancies.

Step {step_number}:
- Manual says parts: {manual_parts}
- Video shows parts: {video_parts}
- Manual actions: {manual_actions}

Return JSON:
{
  "conflict_exists": true/false,
  "video_is_correct": true/false,
  "incorrect_parts": ["parts that are wrong in manual"],
  "correct_parts": ["correct parts from video"],
  "reason": "explanation of discrepancy",
  "confidence": 0.0-1.0
}

Only mark video_is_correct=true if you are confident the video shows the correct parts."""
