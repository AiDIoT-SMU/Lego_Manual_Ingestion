"""
Video Enhancer: Enhances manual steps with detailed sub-steps from video analysis.

Three VLM call pipeline:
1. Frame Classification: Per frame — determine relevance and type (action vs placement).
2. Placement Analysis: Per consecutive placement frame pair — identify what new part was
   added and generate a human-readable action description. Saves video_placements_{id}.json.
3. Reconciliation: Map all placement actions to manual steps, produce sub-steps and
   corrections, output video_enhanced.json.
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
        self.vlm = vlm_extractor
        self.data_service = data_service
        self.settings = settings

        prompts_dir = Path("prompts")

        action_detection_path = prompts_dir / "video_action_detection.txt"
        self.action_detection_template = (
            action_detection_path.read_text()
            if action_detection_path.exists()
            else self._get_default_action_detection_prompt()
        )

        placement_analysis_path = prompts_dir / "video_placement_analysis.txt"
        self.placement_analysis_template = (
            placement_analysis_path.read_text()
            if placement_analysis_path.exists()
            else self._get_default_placement_analysis_prompt()
        )

        reconciliation_path = prompts_dir / "video_reconciliation.txt"
        self.reconciliation_template = (
            reconciliation_path.read_text()
            if reconciliation_path.exists()
            else self._get_default_reconciliation_prompt()
        )

    # ── public ───────────────────────────────────────────────────────────────

    async def enhance_manual_with_video(
        self,
        manual_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """
        Run the full 3-call pipeline and return the video_enhanced.json structure.

        Pipeline:
        1. Extract frames from the full video.
        2. VLM Call 1: Classify every frame as irrelevant / action / placement.
        3. VLM Call 2: For each consecutive placement frame pair, identify what new
           part was added → save video_placements_{video_id}.json.
        4. VLM Call 3: Reconcile placements with enhanced.json → video_enhanced.json.
        """
        logger.info(f"Starting video enhancement for manual {manual_id}, video {video_id}")

        manual_data = self.data_service.get_steps(manual_id)
        logger.info(f"Loaded manual data: {len(manual_data['steps'])} steps")

        frames = await self._extract_all_frames(manual_id, video_id)
        logger.info(f"Extracted {len(frames)} frames")

        # === VLM CALL 1: Frame Classification ===
        logger.info("VLM Call 1: Classifying frames (action vs placement)...")
        classified_frames = await self._classify_all_frames(frames)
        placement_frames = [f for f in classified_frames if f["frame_type"] == "placement"]
        logger.info(
            f"Classification complete: {len(placement_frames)} placement frames, "
            f"{len(classified_frames) - len(placement_frames)} action frames "
            f"(from {len(frames)} total)"
        )

        # === VLM CALL 2: Placement Analysis ===
        logger.info("VLM Call 2: Analyzing consecutive placement frames...")
        placement_actions = await self._analyze_placement_frames(placement_frames)
        logger.info(f"Generated {len(placement_actions)} placement action descriptions")

        video_placements = self._save_video_placements(
            manual_id, video_id, placement_actions, placement_frames
        )

        # === VLM CALL 3: Reconciliation ===
        logger.info("VLM Call 3: Reconciling placements with manual...")
        enhanced_manual = await self._reconcile_placements_with_manual(
            manual_id, video_id, video_placements, manual_data, frames
        )

        logger.info(f"Video enhancement complete: {len(enhanced_manual['steps'])} steps enhanced")
        return enhanced_manual

    # ── frame extraction ─────────────────────────────────────────────────────

    async def _extract_all_frames(self, manual_id: str, video_id: str) -> List[Path]:
        from backend.services.video_processor import VideoProcessor

        video_path = self.settings.data_dir / "videos" / manual_id / f"{video_id}.mp4"
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        video_processor = VideoProcessor(self.settings)
        frames_dir = video_path.parent / f"{video_id}_enhancement_frames"
        frames_dir.mkdir(exist_ok=True)

        frames = video_processor.extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            frame_interval=30,
            max_frames=None
        )
        return [Path(f["frame_path"]) for f in frames]

    # ── VLM Call 1 ────────────────────────────────────────────────────────────

    async def _classify_all_frames(self, frames: List[Path]) -> List[Dict[str, Any]]:
        """
        VLM Call 1: Classify each frame as irrelevant, action, or placement.

        Returns only relevant frames with frame_type set to "action" or "placement".
        """
        classified = []

        for i, frame_path in enumerate(frames):
            frame_num = int(frame_path.stem.split('_')[-1])
            timestamp = frame_num / 30.0

            try:
                frame_img = _resize_image(str(frame_path), width=600)
                frame_b64 = _image_to_b64(frame_img)

                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
                    {"type": "text", "text": self.action_detection_template}
                ]
                messages = [{"role": "user", "content": content}]

                raw = self.vlm._litellm_with_retry(messages)
                result = _parse_json(raw)

                is_relevant = result.get("is_relevant", True)
                frame_type = result.get("frame_type") if is_relevant else None

                logger.info(
                    f"  Frame {frame_num} (t={timestamp:.1f}s): "
                    f"relevant={is_relevant}, type={frame_type}, "
                    f"confidence={result.get('confidence', 0.0):.2f}"
                )

                if is_relevant:
                    classified.append({
                        "frame_number": frame_num,
                        "frame_path": frame_path,
                        "timestamp": timestamp,
                        "frame_type": frame_type,
                        "confidence": result.get("confidence", 0.0)
                    })

            except Exception as e:
                logger.error(f"Failed to classify frame {frame_num}: {e}")
                classified.append({
                    "frame_number": frame_num,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "frame_type": "action",
                    "confidence": 0.0
                })

            if (i + 1) % 50 == 0:
                logger.info(f"  === Classified {i + 1}/{len(frames)} frames ===")

        placement_count = sum(1 for f in classified if f["frame_type"] == "placement")
        logger.info(
            f"Classification summary: {len(classified)} relevant, "
            f"{placement_count} placement, {len(classified) - placement_count} action, "
            f"{len(frames) - len(classified)} irrelevant"
        )
        return classified

    # ── VLM Call 2 ────────────────────────────────────────────────────────────

    async def _analyze_placement_frames(
        self,
        placement_frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        VLM Call 2: For each consecutive placement frame pair, identify the new part
        added and generate a natural-language action description.

        Sends (previous frame + current frame) as two images per call.
        For the first frame, only the current frame is sent.
        """
        if not placement_frames:
            return []

        placement_actions = []

        for i, current_frame in enumerate(placement_frames):
            frame_num = current_frame["frame_number"]
            timestamp = current_frame["timestamp"]

            try:
                current_img = _resize_image(str(current_frame["frame_path"]), width=800)
                current_b64 = _image_to_b64(current_img)

                if i == 0:
                    content = [
                        {"type": "text", "text": "CURRENT PLACEMENT FRAME (first placement — no previous frame):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                        {"type": "text", "text": self.placement_analysis_template}
                    ]
                else:
                    prev_frame = placement_frames[i - 1]
                    prev_img = _resize_image(str(prev_frame["frame_path"]), width=800)
                    prev_b64 = _image_to_b64(prev_img)

                    content = [
                        {"type": "text", "text": "PREVIOUS PLACEMENT FRAME:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_b64}"}},
                        {"type": "text", "text": "CURRENT PLACEMENT FRAME:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                        {"type": "text", "text": self.placement_analysis_template}
                    ]

                messages = [{"role": "user", "content": content}]

                raw = self.vlm._litellm_with_retry(messages)
                result = _parse_json(raw)

                action_desc = result.get("action_description")

                if action_desc:
                    placement_actions.append({
                        "placement_index": i,
                        "frame_number": frame_num,
                        "timestamp": timestamp,
                        "action_description": action_desc,
                        "new_parts": result.get("new_parts", []),
                        "location": result.get("location"),
                        "confidence": result.get("confidence", 0.0)
                    })
                    logger.info(f"  Placement {i} (frame {frame_num}): {action_desc}")
                else:
                    logger.info(f"  Placement {i} (frame {frame_num}): no new parts detected")

            except Exception as e:
                logger.error(f"Failed to analyze placement frame {frame_num}: {e}")

        return placement_actions

    # ── intermediate file ─────────────────────────────────────────────────────

    def _save_video_placements(
        self,
        manual_id: str,
        video_id: str,
        placement_actions: List[Dict[str, Any]],
        placement_frames: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save intermediate video_placements_{video_id}.json and return its content."""
        video_placements = {
            "manual_id": manual_id,
            "video_id": video_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_placement_frames": len(placement_frames),
            "placements": placement_actions
        }

        output_path = (
            self.settings.data_dir / "processed" / manual_id
            / f"video_placements_{video_id}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_placements, f, indent=2)

        logger.info(
            f"Saved {len(placement_actions)} placement actions to {output_path}"
        )
        return video_placements

    # ── VLM Call 3 ────────────────────────────────────────────────────────────

    async def _reconcile_placements_with_manual(
        self,
        manual_id: str,
        video_id: str,
        video_placements: Dict[str, Any],
        manual_data: Dict[str, Any],
        all_frames: List[Path]
    ) -> Dict[str, Any]:
        """
        VLM Call 3: Map placement actions to manual steps, generate micro sub-steps,
        correct errors, and return the video_enhanced.json structure.

        Sends enhanced.json + video_placements.json as text, plus up to 5 sample
        placement frame images for visual grounding.
        """
        enhanced_json_str = json.dumps(manual_data, indent=2)
        placements_json_str = json.dumps(video_placements, indent=2)

        prompt = self.reconciliation_template.replace(
            "{enhanced_json}", enhanced_json_str
        ).replace(
            "{video_placements_json}", placements_json_str
        )

        # Build a frame_number → Path lookup for sample images
        frame_lookup: Dict[int, Path] = {
            int(f.stem.split('_')[-1]): f for f in all_frames
        }

        # Pick up to 5 evenly-spaced placement frames as visual context
        placements = video_placements.get("placements", [])
        step = max(1, len(placements) // 5)
        sample_placements = placements[::step][:5]

        content: List[Dict[str, Any]] = []
        if sample_placements:
            content.append({"type": "text", "text": "SAMPLE PLACEMENT FRAMES FOR VISUAL CONTEXT:"})
            for p in sample_placements:
                frame_path = frame_lookup.get(p["frame_number"])
                if frame_path and frame_path.exists():
                    try:
                        img = _resize_image(str(frame_path), width=600)
                        b64 = _image_to_b64(img)
                        content.append({
                            "type": "text",
                            "text": (
                                f"Placement {p['placement_index']} "
                                f"(frame {p['frame_number']}): "
                                f"{p.get('action_description', '')}"
                            )
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        })
                    except Exception as e:
                        logger.warning(f"Could not include frame {p['frame_number']} as context: {e}")

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        reconciled_by_num = {
            s["step_number"]: s for s in result.get("steps", [])
        }

        enhanced_steps = []
        for step in manual_data["steps"]:
            step_num = step["step_number"]
            reconciled = reconciled_by_num.get(step_num, {})
            enhanced_steps.append({
                **step,
                "sub_steps": reconciled.get("sub_steps", []),
                "corrections": reconciled.get("corrections", [])
            })

        return {
            "manual_id": manual_id,
            "source_video_id": video_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "video_metadata": {
                "frame_count": len(all_frames),
                "filename": f"{video_id}.mp4"
            },
            "steps": enhanced_steps,
            "manual_step_mapping": {
                str(s["step_number"]): [s["step_number"]] for s in manual_data["steps"]
            }
        }

    # ── fallback prompts (used when prompt files are missing) ─────────────────

    def _get_default_action_detection_prompt(self) -> str:
        return (
            "You are analyzing a single frame from a LEGO assembly video.\n\n"
            "Determine if this frame is RELEVANT and classify it as ACTION or PLACEMENT.\n\n"
            "- IRRELEVANT: title screens, black frames, no assembly visible.\n"
            "- ACTION: hands actively handling/placing a part; placement not yet finalized.\n"
            "- PLACEMENT: part is resting on assembly, hands clear, stable state visible.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "is_relevant": true,\n'
            '  "frame_type": "action",\n'
            '  "confidence": 0.85,\n'
            '  "reasoning": "brief explanation"\n'
            "}\n\n"
            "If is_relevant is false, set frame_type to null.\n"
            "Do NOT include any text before or after the JSON."
        )

    def _get_default_placement_analysis_prompt(self) -> str:
        return (
            "You are analyzing LEGO placement frames.\n\n"
            "Compare PREVIOUS PLACEMENT FRAME and CURRENT PLACEMENT FRAME.\n"
            "Identify what new part was added and where.\n\n"
            'Output: "Add a [color] [type] [size] [part] to the [location] of the [reference]"\n\n'
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "new_parts": ["description"],\n'
            '  "action_description": "Add a ... to the ...",\n'
            '  "location": "specific location",\n'
            '  "confidence": 0.9,\n'
            '  "reasoning": "brief explanation"\n'
            "}\n\n"
            "If no new parts visible, set action_description to null and confidence to 0.0.\n"
            "Do NOT include any text before or after the JSON."
        )

    def _get_default_reconciliation_prompt(self) -> str:
        return (
            "Reconcile LEGO assembly video evidence with a manual to create enhanced instructions.\n\n"
            "INPUT 1 — ENHANCED MANUAL:\n{enhanced_json}\n\n"
            "INPUT 2 — VIDEO PLACEMENTS:\n{video_placements_json}\n\n"
            "TASK:\n"
            "1. Map each video placement to the correct manual step.\n"
            "2. Create micro sub-steps (1.1, 1.2, ...) within each step.\n"
            "3. Correct manual errors where video clearly contradicts it.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "step_number": 1,\n'
            '      "sub_steps": [\n'
            "        {\n"
            '          "sub_step_number": "1.1",\n'
            '          "action_description": "Add a ...",\n'
            '          "placement_index": 0,\n'
            '          "parts_involved": ["part"],\n'
            '          "confidence": 0.9\n'
            "        }\n"
            "      ],\n"
            '      "corrections": []\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Do NOT include any text before or after the JSON."
        )
