"""
EXPERIMENTAL FEATURE

Video Analyzer: Verifies user assembly videos against ground truth.

Uses 2 VLM call types:
1. Step Detection + Placement Detection: Identify current step and if this is a placement frame
2. Verification (conditional): Verify part and spatial placement against ground truth

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
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


class VideoAnalyzer:
    """Verifies user assembly videos against ground truth (video_enhanced.json or enhanced.json)."""

    def __init__(
        self,
        vlm_extractor: VLMExtractor,
        data_service: DataService,
        settings: Settings
    ):
        """
        Initialize video analyzer.

        Args:
            vlm_extractor: VLM extractor for making VLM calls
            data_service: Data service for loading manual data
            settings: Application settings
        """
        self.vlm = vlm_extractor
        self.data_service = data_service
        self.settings = settings

    def analyze_video(
        self,
        manual_id: str,
        frames: List[Dict[str, Any]],
        video_id: str
    ) -> Dict[str, Any]:
        """
        Analyze user assembly video for verification (2 VLM call types).

        VLM Call 1: For each frame, detect step and if it's a placement frame
        VLM Call 2: For placement frames, verify part and spatial placement

        Args:
            manual_id: Manual ID to verify against
            frames: List of frame metadata from VideoProcessor
            video_id: Video ID for tracking

        Returns:
            Complete video analysis with verification results:
            {
                "video_id": str,
                "manual_id": str,
                "total_frames_extracted": int,
                "frame_analyses": [...],
                "step_timeline": [...],
                "verification_summary": {...}
            }
        """
        logger.info(f"Analyzing {len(frames)} frames for user assembly verification (video {video_id})")

        # Load ground truth (prefer video_enhanced.json, fallback to enhanced.json)
        ground_truth = self._load_ground_truth(manual_id)
        logger.info(f"Loaded ground truth with {len(ground_truth['steps'])} steps")

        frame_analyses = []
        total_placements = 0
        correct_placements = 0
        incorrect_placements = 0

        for i, frame in enumerate(frames):
            logger.info(
                f"Analyzing frame {i+1}/{len(frames)} "
                f"(frame #{frame['frame_number']}, t={frame['timestamp_seconds']}s)"
            )

            # === VLM CALL 1: Step Detection + Placement Detection ===
            try:
                detection_result = self._detect_step_and_placement(
                    frame["frame_path"],
                    ground_truth
                )
                logger.info(
                    f"  VLM Call 1: step={detection_result.get('detected_step')}, "
                    f"is_placement={detection_result.get('is_placement_frame')}"
                )
            except Exception as e:
                logger.error(f"VLM Call 1 failed for frame {frame['frame_number']}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                detection_result = {
                    "detected_step": None,
                    "is_placement_frame": False,
                    "confidence": 0.0
                }

            # === VLM CALL 2: Verification (conditional) ===
            verification_result = None
            if detection_result.get("is_placement_frame") and detection_result.get("detected_step"):
                try:
                    total_placements += 1
                    verification_result = self._verify_placement(
                        frame["frame_path"],
                        detection_result["detected_step"],
                        ground_truth
                    )
                    logger.info(
                        f"  VLM Call 2: correct_part={verification_result.get('correct_part')}, "
                        f"correct_placement={verification_result.get('correct_spatial_placement')}"
                    )

                    # Track correctness
                    if verification_result.get("correct_part") and verification_result.get("correct_spatial_placement"):
                        correct_placements += 1
                    else:
                        incorrect_placements += 1

                except Exception as e:
                    logger.error(f"VLM Call 2 failed for frame {frame['frame_number']}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    verification_result = {
                        "correct_part": False,
                        "correct_spatial_placement": False,
                        "error": str(e)
                    }
                    incorrect_placements += 1

            # Combine results
            frame_analysis = {
                "frame_number": frame["frame_number"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "detected_step": detection_result.get("detected_step"),
                "step_confidence": detection_result.get("confidence", 0.0),
                "is_placement_frame": detection_result.get("is_placement_frame", False),
                "verification": verification_result
            }
            frame_analyses.append(frame_analysis)

        # Build timeline
        step_timeline = self._build_step_timeline(frame_analyses)

        # Verification summary
        verification_summary = {
            "total_placements_detected": total_placements,
            "correct_placements": correct_placements,
            "incorrect_placements": incorrect_placements,
            "accuracy": round(correct_placements / total_placements, 2) if total_placements > 0 else 0.0
        }

        logger.info(
            f"Analysis complete: {len(step_timeline)} steps detected, "
            f"{total_placements} placements verified "
            f"({correct_placements} correct, {incorrect_placements} incorrect)"
        )

        return {
            "video_id": video_id,
            "manual_id": manual_id,
            "total_frames_extracted": len(frames),
            "frame_analyses": frame_analyses,
            "step_timeline": step_timeline,
            "verification_summary": verification_summary
        }

    def _load_ground_truth(self, manual_id: str) -> Dict[str, Any]:
        """
        Load ground truth for verification.

        Priority: video_enhanced.json > enhanced.json

        Args:
            manual_id: Manual identifier

        Returns:
            Ground truth data with steps
        """
        # Try video_enhanced.json first
        try:
            video_enhanced = self.data_service.get_video_enhanced_steps(manual_id)
            logger.info(f"Loaded video_enhanced.json with {len(video_enhanced['steps'])} steps")
            return video_enhanced
        except Exception:
            logger.info("video_enhanced.json not found, falling back to enhanced.json")

        # Fallback to enhanced.json
        try:
            enhanced = self.data_service.get_steps(manual_id)
            logger.info(f"Loaded enhanced.json with {len(enhanced['steps'])} steps")
            return enhanced
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            raise ValueError(f"No ground truth found for manual {manual_id}: {e}")

    def _detect_step_and_placement(
        self,
        frame_path: str,
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        VLM CALL 1: Detect current step and if this is a placement frame.

        Args:
            frame_path: Path to video frame
            ground_truth: Ground truth data (video_enhanced.json or enhanced.json)

        Returns:
            {
                "detected_step": int | null,
                "is_placement_frame": bool,
                "confidence": float,
                "reasoning": str
            }
        """
        # Build steps summary
        steps_summary = []
        for step in ground_truth["steps"]:
            step_num = step["step_number"]

            # Get description from sub_steps if available (video_enhanced.json)
            if "sub_steps" in step and step["sub_steps"]:
                sub_descs = [s["description"] for s in step["sub_steps"][:3]]
                desc = ", ".join(sub_descs)
            # Otherwise use actions (enhanced.json)
            elif "actions" in step and step["actions"]:
                desc = ", ".join(step["actions"][:2])
            else:
                desc = "Assembly step"

            steps_summary.append(f"Step {step_num}: {desc}")

        # Build prompt
        prompt = f"""Analyze this video frame of a user assembling LEGO.

TASK:
1. Identify which step the user is currently on (or about to start)
2. Determine if this frame shows a COMPLETED PLACEMENT (user has finished placing a brick and hands are clear)

STEPS IN THIS MANUAL:
{chr(10).join(steps_summary)}

Return JSON:
{{
  "detected_step": <step number> | null,
  "is_placement_frame": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

A placement frame shows:
- Brick has been PLACED and placement is COMPLETE
- Hands have moved away or are no longer obstructing the view
- The final placement result is clearly visible
- You can see where the brick ended up on the assembly

NOT a placement frame:
- Mid-placement (hands still placing/pressing the brick)
- Hands obstructing the view of the placement
- Just holding a brick (not yet placed)
- Picking up a brick
- Looking at the assembly without placing
- Checking reference manual
- Empty hands"""

        # Load frame
        frame_img = _resize_image(frame_path)
        frame_b64 = _image_to_b64(frame_img)

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
            {"type": "text", "text": prompt}
        ]

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        return result

    def _verify_placement(
        self,
        frame_path: str,
        detected_step: int,
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        VLM CALL 2: Verify part and spatial placement against ground truth.

        Only called for placement frames.

        Args:
            frame_path: Path to video frame showing placement
            detected_step: Step number detected in VLM Call 1
            ground_truth: Ground truth data with spatial information

        Returns:
            {
                "correct_part": bool,
                "correct_spatial_placement": bool,
                "expected_part": str,
                "detected_part": str,
                "expected_spatial": str,
                "detected_spatial": str,
                "feedback": str,
                "confidence": float
            }
        """
        # Find the step in ground truth
        step_data = None
        for step in ground_truth["steps"]:
            if step["step_number"] == detected_step:
                step_data = step
                break

        if not step_data:
            return {
                "correct_part": False,
                "correct_spatial_placement": False,
                "error": f"Step {detected_step} not found in ground truth"
            }

        # Extract expected parts and spatial info
        expected_parts = []
        expected_spatial_info = []

        # If video_enhanced.json (has sub_steps with spatial_description)
        if "sub_steps" in step_data:
            for sub_step in step_data["sub_steps"]:
                if sub_step["action_type"] in ["place", "attach"]:
                    expected_parts.extend(sub_step.get("parts_involved", []))
                    if sub_step.get("spatial_description"):
                        spatial = sub_step["spatial_description"]
                        spatial_str = (
                            f"Place {spatial['placement_part']} on {spatial['target_part']} "
                            f"at {spatial['location']} ({spatial['position_detail']})"
                        )
                        expected_spatial_info.append(spatial_str)

        # If enhanced.json (has parts_required)
        elif "parts_required" in step_data:
            expected_parts = [p["description"] for p in step_data["parts_required"]]
            expected_spatial_info = step_data.get("actions", [])

        # Build prompt
        expected_parts_str = ", ".join(expected_parts) if expected_parts else "No specific parts listed"
        expected_spatial_str = "; ".join(expected_spatial_info) if expected_spatial_info else "No specific placement info"

        prompt = f"""Verify this LEGO placement against the ground truth instructions.

EXPECTED (Ground Truth for Step {detected_step}):
- Parts: {expected_parts_str}
- Spatial Placement: {expected_spatial_str}

TASK:
1. Identify what part the user is placing
2. Check if it's the correct part
3. Check if the spatial placement (location, orientation) is correct

Return JSON:
{{
  "correct_part": true/false,
  "correct_spatial_placement": true/false,
  "expected_part": "{expected_parts[0] if expected_parts else 'unknown'}",
  "detected_part": "description of part user is placing",
  "expected_spatial": "{expected_spatial_info[0] if expected_spatial_info else 'unknown'}",
  "detected_spatial": "description of where/how user is placing it",
  "feedback": "constructive feedback for user (if incorrect)",
  "confidence": 0.0-1.0
}}

Be precise about spatial placement:
- Location: left/right/center, top/bottom
- Stud count: "2 studs from edge" vs "3 studs from edge"
- Orientation: horizontal vs vertical"""

        # Load frame
        frame_img = _resize_image(frame_path)
        frame_b64 = _image_to_b64(frame_img)

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
            {"type": "text", "text": prompt}
        ]

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        return result

    def _build_step_timeline(
        self,
        frame_analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build step timeline from frame analyses.

        Groups consecutive frames by detected step.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            [
                {
                    "step_number": int,
                    "start_time": float,
                    "end_time": float,
                    "duration_seconds": float,
                    "confidence_avg": float,
                    "frame_numbers": [int, ...],
                    "placements_in_step": int,
                    "correct_placements": int
                }
            ]
        """
        if not frame_analyses:
            return []

        timeline = []
        current_step = None
        step_start = None
        step_frames = []
        step_confidences = []
        step_placements = 0
        step_correct = 0

        for analysis in frame_analyses:
            detected = analysis["detected_step"]
            confidence = analysis["step_confidence"]

            # Only accept step if confidence is high enough
            if detected is not None and confidence > 0.6:
                if detected != current_step:
                    # Step transition
                    if current_step is not None:
                        timeline.append({
                            "step_number": current_step,
                            "start_time": step_start,
                            "end_time": analysis["timestamp_seconds"],
                            "duration_seconds": round(
                                analysis["timestamp_seconds"] - step_start, 2
                            ),
                            "confidence_avg": round(
                                sum(step_confidences) / len(step_confidences), 2
                            ) if step_confidences else 0.0,
                            "frame_numbers": step_frames,
                            "placements_in_step": step_placements,
                            "correct_placements": step_correct
                        })

                    current_step = detected
                    step_start = analysis["timestamp_seconds"]
                    step_frames = [analysis["frame_number"]]
                    step_confidences = [confidence]
                    step_placements = 0
                    step_correct = 0
                else:
                    # Same step continues
                    step_frames.append(analysis["frame_number"])
                    step_confidences.append(confidence)

                # Track placements in this step
                if analysis.get("is_placement_frame") and analysis.get("verification"):
                    step_placements += 1
                    verification = analysis["verification"]
                    if verification.get("correct_part") and verification.get("correct_spatial_placement"):
                        step_correct += 1

        # Add final step
        if current_step is not None and frame_analyses:
            timeline.append({
                "step_number": current_step,
                "start_time": step_start,
                "end_time": frame_analyses[-1]["timestamp_seconds"],
                "duration_seconds": round(
                    frame_analyses[-1]["timestamp_seconds"] - step_start, 2
                ),
                "confidence_avg": round(
                    sum(step_confidences) / len(step_confidences), 2
                ) if step_confidences else 0.0,
                "frame_numbers": step_frames,
                "placements_in_step": step_placements,
                "correct_placements": step_correct
            })

        return timeline
