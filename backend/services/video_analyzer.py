"""
Video Analyzer: Analyzes assembly videos using VLM to detect steps and parts.
Uses multi-image VLM calls to compare video frames against reference images.
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
    """Analyzes assembly videos to detect steps and part usage."""

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

        # Load prompts
        prompts_dir = Path("prompts")
        with open(prompts_dir / "video_step_detection.txt", "r") as f:
            self.step_prompt_template = f.read()
        with open(prompts_dir / "video_part_detection.txt", "r") as f:
            self.part_prompt_template = f.read()

    def analyze_video(
        self,
        manual_id: str,
        frames: List[Dict[str, Any]],
        video_id: str
    ) -> Dict[str, Any]:
        """
        Analyze all frames for step detection and part tracking.

        Args:
            manual_id: Manual ID to analyze against
            frames: List of frame metadata from VideoProcessor
            video_id: Video ID for tracking

        Returns:
            Complete video analysis:
            {
                "video_id": str,
                "manual_id": str,
                "total_frames_extracted": int,
                "frame_analyses": [...],
                "step_timeline": [...],
                "parts_used": {...}
            }
        """
        logger.info(f"Analyzing {len(frames)} frames for video {video_id}")

        # Load reference data
        try:
            manual_data = self.data_service.get_steps(manual_id)
            logger.info(f"Loaded manual steps: {len(manual_data['steps'])} steps")
        except Exception as e:
            logger.error(f"Failed to load manual data for {manual_id}: {e}")
            raise ValueError(f"Manual {manual_id} not found or invalid: {e}")

        try:
            parts_catalog = self.data_service.get_parts_catalog(manual_id)
            logger.info(f"Loaded parts catalog: {parts_catalog['total_unique_parts']} unique parts")
        except Exception as e:
            logger.error(f"Failed to load parts catalog for {manual_id}: {e}")
            raise ValueError(f"Parts catalog for manual {manual_id} not found: {e}")

        logger.info(
            f"Loaded manual {manual_id}: {len(manual_data['steps'])} steps, "
            f"{parts_catalog['total_unique_parts']} unique parts"
        )

        frame_analyses = []

        for i, frame in enumerate(frames):
            logger.info(
                f"Analyzing frame {i+1}/{len(frames)} "
                f"(frame #{frame['frame_number']}, t={frame['timestamp_seconds']}s)"
            )

            # Step detection
            try:
                step_result = self._analyze_frame_for_step(
                    frame["frame_path"],
                    manual_data
                )
                logger.debug(
                    f"  Step detection: step={step_result.get('current_step')}, "
                    f"confidence={step_result.get('confidence', 0.0):.2f}"
                )
            except Exception as e:
                logger.error(f"Step detection failed for frame {frame['frame_number']}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                step_result = {"current_step": None, "confidence": 0.0, "reasoning": str(e)}

            # Part detection
            try:
                part_result = self._analyze_frame_for_parts(
                    frame["frame_path"],
                    parts_catalog
                )
                logger.debug(
                    f"  Part detection: {len(part_result.get('detected_parts', []))} parts found"
                )
            except Exception as e:
                logger.error(f"Part detection failed for frame {frame['frame_number']}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                part_result = {"detected_parts": []}

            frame_analyses.append({
                "frame_number": frame["frame_number"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "detected_step": step_result.get("current_step"),
                "step_confidence": step_result.get("confidence", 0.0),
                "step_reasoning": step_result.get("reasoning", ""),
                "detected_parts": [p["description"] for p in part_result.get("detected_parts", [])],
                "parts_confidence": max(
                    [p.get("confidence", 0.0) for p in part_result.get("detected_parts", [])],
                    default=0.0
                )
            })

        # Build timeline and aggregate parts
        step_timeline = self._build_step_timeline(frame_analyses)
        parts_used = self._aggregate_parts_usage(frame_analyses)

        logger.info(
            f"Analysis complete: {len(step_timeline)} steps detected, "
            f"{len(parts_used)} parts used"
        )

        return {
            "video_id": video_id,
            "manual_id": manual_id,
            "total_frames_extracted": len(frames),
            "frame_analyses": frame_analyses,
            "step_timeline": step_timeline,
            "parts_used": parts_used
        }

    def _analyze_frame_for_step(
        self,
        frame_path: str,
        manual_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect which assembly step is shown in the frame.

        Uses multi-image VLM call: video frame + all subassembly reference images.

        Args:
            frame_path: Path to video frame
            manual_data: Manual data with steps and subassemblies

        Returns:
            {
                "current_step": int | null,
                "confidence": float,
                "reasoning": str
            }
        """
        # Build steps summary for prompt
        steps_summary = []
        for step in manual_data["steps"]:
            sub_desc = "N/A"
            if step.get("subassemblies") and len(step["subassemblies"]) > 0:
                sub_desc = step["subassemblies"][0].get("description", "N/A")
            steps_summary.append(f"Step {step['step_number']}: {sub_desc}")

        # Build prompt
        prompt = self.step_prompt_template.replace(
            "{n_steps}", str(len(manual_data["steps"]))
        ).replace(
            "{steps_summary}", "\n".join(steps_summary)
        )

        # Load video frame
        frame_img = _resize_image(frame_path)
        frame_b64 = _image_to_b64(frame_img)

        # Build multi-image content
        content = [
            {"type": "text", "text": "VIDEO FRAME (current user assembly):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
        ]

        # Add subassembly reference images
        for step in manual_data["steps"]:
            if step.get("subassemblies") and len(step["subassemblies"]) > 0:
                sub = step["subassemblies"][0]
                if sub.get("cropped_image_path"):
                    # Convert relative path to absolute
                    sub_path = self.settings.data_dir.parent / sub["cropped_image_path"]
                    if sub_path.exists():
                        sub_img = _resize_image(str(sub_path))
                        sub_b64 = _image_to_b64(sub_img)
                        content.append({
                            "type": "text",
                            "text": f"REFERENCE: Step {step['step_number']}"
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{sub_b64}"}
                        })

        content.append({"type": "text", "text": prompt})

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        logger.debug(
            f"Step detection: step={result.get('current_step')}, "
            f"confidence={result.get('confidence')}"
        )

        return result

    def _analyze_frame_for_parts(
        self,
        frame_path: str,
        parts_catalog: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect which parts are being used in the frame.

        Uses multi-image VLM call: video frame + cropped part reference images.

        Args:
            frame_path: Path to video frame
            parts_catalog: Parts catalog with reference images

        Returns:
            {
                "detected_parts": [
                    {"description": str, "confidence": float, "state": str}
                ]
            }
        """
        # Build parts list for prompt
        parts_list = []
        for i, part in enumerate(parts_catalog["parts"]):
            steps_str = ", ".join(map(str, part["used_in_steps"]))
            parts_list.append(
                f"{i+1}. {part['description']} (used in steps: [{steps_str}])"
            )

        # Build prompt
        prompt = self.part_prompt_template.replace(
            "{parts_list}", "\n".join(parts_list)
        )

        # Load video frame
        frame_img = _resize_image(frame_path)
        frame_b64 = _image_to_b64(frame_img)

        # Build multi-image content
        content = [
            {"type": "text", "text": "VIDEO FRAME (user's hands and workspace):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
            {"type": "text", "text": "PART REFERENCE IMAGES:"}
        ]

        # Add part reference images
        for part in parts_catalog["parts"]:
            if part.get("images") and len(part["images"]) > 0:
                # Use first reference image
                part_path = self.settings.data_dir.parent / part["images"][0]
                if part_path.exists():
                    part_img = _resize_image(str(part_path))
                    part_b64 = _image_to_b64(part_img)
                    content.append({
                        "type": "text",
                        "text": f"PART: {part['description']}"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{part_b64}"}
                    })

        content.append({"type": "text", "text": prompt})

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        logger.debug(f"Part detection: {len(result.get('detected_parts', []))} parts found")

        return result

    def _build_step_timeline(
        self,
        frame_analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build step timeline from frame analyses.

        Groups consecutive frames by detected step.
        Applies temporal smoothing: requires confidence >0.6 for transitions.

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
                    "frame_numbers": [int, ...]
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
                            "frame_numbers": step_frames
                        })

                    current_step = detected
                    step_start = analysis["timestamp_seconds"]
                    step_frames = [analysis["frame_number"]]
                    step_confidences = [confidence]
                else:
                    # Same step continues
                    step_frames.append(analysis["frame_number"])
                    step_confidences.append(confidence)
            else:
                # Low confidence or null detection, continue current step
                if current_step is not None:
                    step_frames.append(analysis["frame_number"])

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
                "frame_numbers": step_frames
            })

        return timeline

    def _aggregate_parts_usage(
        self,
        frame_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate parts usage across all frames.

        Tracks when each part is first/last seen.
        Marks as "used" if seen in 2+ frames.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            {
                "part description": {
                    "first_seen_timestamp": float,
                    "last_seen_timestamp": float,
                    "marked_as_used": bool,
                    "usage_confidence": float,
                    "frames_visible": [int, ...]
                }
            }
        """
        parts_tracker = {}

        for analysis in frame_analyses:
            for part_desc in analysis["detected_parts"]:
                if part_desc not in parts_tracker:
                    parts_tracker[part_desc] = {
                        "first_seen_timestamp": analysis["timestamp_seconds"],
                        "last_seen_timestamp": analysis["timestamp_seconds"],
                        "frames_visible": [],
                        "confidences": [],
                        "marked_as_used": False
                    }

                # Update tracking
                parts_tracker[part_desc]["frames_visible"].append(analysis["frame_number"])
                parts_tracker[part_desc]["last_seen_timestamp"] = analysis["timestamp_seconds"]
                parts_tracker[part_desc]["confidences"].append(analysis["parts_confidence"])

                # Mark as used if seen in 2+ frames
                if len(parts_tracker[part_desc]["frames_visible"]) >= 2:
                    parts_tracker[part_desc]["marked_as_used"] = True

        # Calculate average confidence and clean up intermediate fields
        for part_desc, data in parts_tracker.items():
            data["usage_confidence"] = round(
                sum(data["confidences"]) / len(data["confidences"]), 2
            ) if data["confidences"] else 0.0
            del data["confidences"]

        return parts_tracker
