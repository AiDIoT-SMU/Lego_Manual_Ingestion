"""
Video Enhancer V2: Improved 4-pass pipeline with quality filtering and context awareness.

Four VLM call pipeline:
1. Frame Classification: Batch classify frames with quality metrics (action vs placement_candidate).
2. Placement Validation: Context-aware analysis using enhanced.json parts list + duplicate detection.
3. Per-Placement Reconciliation: Individual VLM calls for each placement to verify against expected parts using annotated frames + reference images.
4. Atomic Sub-steps: Generate 1-part-per-sub-step instructions with spatial positioning from reconciled placements.
"""

import io
import base64
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from PIL import Image as PILImage

from ingestion.vlm_extractor import VLMExtractor, _parse_json, _box2d_to_bbox
from backend.services.data_service import DataService
from backend.services.video_quality_filter import VideoQualityFilter
from backend.services.video_state_tracker import SubassemblyTracker
from backend.services.yolo_world_sam3_detector import (
    detect_objects_yolo_world_sam3,
    annotate_frame_with_objects,
    get_generic_lego_query,
    find_new_masks
)
from config.settings import Settings


def _resize_image(image_path: str, width: int = 800) -> PILImage.Image:
    """Resize image to specified width, preserving aspect ratio."""
    with PILImage.open(image_path) as img:
        orig_w, orig_h = img.size
        new_h = int(width * orig_h / orig_w)
        return img.resize((width, new_h), PILImage.Resampling.LANCZOS)


def _image_to_b64(img: PILImage.Image) -> str:
    """Convert PIL Image to base64 string using JPEG compression."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _crop_to_bbox(
    image_path: str,
    box_2d: Optional[List[int]],
    padding: int = 30
) -> Optional[PILImage.Image]:
    """
    Crop image tightly to bounding box with padding.

    This focuses the VLM's attention on the exact part being placed,
    making stud counting and part identification more accurate.

    Args:
        image_path: Path to source image
        box_2d: [ymin, xmin, ymax, xmax] each 0-1000, or None
        padding: Pixels of padding around the bbox

    Returns:
        Cropped PIL Image, or None if bbox is invalid
    """
    if not box_2d or len(box_2d) != 4:
        return None

    try:
        img = PILImage.open(str(image_path)).convert("RGB")
        img_w, img_h = img.size

        # Convert box_2d [ymin, xmin, ymax, xmax] from 0-1000 to pixels
        bbox = _box2d_to_bbox(box_2d, img_w, img_h)

        # Calculate crop box with padding
        x_min = max(0, bbox.x - padding)
        y_min = max(0, bbox.y - padding)
        x_max = min(img_w, bbox.x + bbox.width + padding)
        y_max = min(img_h, bbox.y + bbox.height + padding)

        # Validate crop box
        if x_max <= x_min or y_max <= y_min:
            logger.warning(f"Invalid crop box after padding: ({x_min}, {y_min}, {x_max}, {y_max})")
            return None

        # Crop the image
        cropped = img.crop((x_min, y_min, x_max, y_max))

        return cropped

    except Exception as e:
        logger.warning(f"Failed to crop to bbox: {e}")
        return None


def _parse_quantity_from_description(description: str) -> int:
    """
    Parse quantity from part description.

    Examples:
        "white 1x2 brick (4x)" -> 4
        "red tile (1x)" -> 1
        "grey baseplate" -> 1
    """
    import re
    match = re.search(r'\((\d+)x\)', description)
    if match:
        return int(match.group(1))
    return 1


def _draw_placement_bbox(
    frame_path: Any,
    box_2d: Optional[List],
    frame_num: int,
    label: str,
    confidence: float,
    output_dir: Path
) -> Optional[Path]:
    """
    Draw a bounding box on the placement frame and save the annotated image.

    Uses the same [ymin, xmin, ymax, xmax] 0-1000 convention as vlm_extractor,
    converted via the shared _box2d_to_bbox helper.

    Args:
        frame_path: Path to the source frame (str or Path)
        box_2d: [ymin, xmin, ymax, xmax] each 0-1000, or None
        frame_num: Frame number used for the output filename
        label: Action description to overlay on the box
        confidence: VLM confidence score
        output_dir: Directory to save the annotated image

    Returns:
        Path to the saved annotated image, or None on failure
    """
    from PIL import ImageDraw

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        img = PILImage.open(str(frame_path)).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        drawn_box = False
        if box_2d and isinstance(box_2d, list) and len(box_2d) == 4:
            bbox = _box2d_to_bbox(box_2d, img_w, img_h)
            x0, y0 = bbox.x, bbox.y
            x1, y1 = bbox.x + bbox.width, bbox.y + bbox.height

            if x1 > x0 and y1 > y0:
                box_color = (0, 220, 0) if confidence >= 0.8 else (255, 165, 0) if confidence >= 0.6 else (255, 50, 50)
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=3)

                short_label = label[:50] + ("…" if len(label) > 50 else "")
                caption = f"{short_label} ({confidence:.2f})"
                text_y = max(0, y0 - 18)
                draw.rectangle([x0, text_y, x0 + len(caption) * 7, text_y + 16], fill=box_color)
                draw.text((x0 + 2, text_y + 1), caption, fill=(0, 0, 0))
                drawn_box = True
            else:
                logger.debug(f"Frame {frame_num}: box_2d degenerated to zero size after conversion (raw: {box_2d})")

        if not drawn_box:
            caption = f"frame {frame_num} | no bbox | conf={confidence:.2f}"
            draw.rectangle([0, 0, img_w, 20], fill=(180, 100, 0))
            draw.text((4, 2), caption, fill=(255, 255, 255))

        out_path = output_dir / f"frame_{frame_num:04d}.jpg"
        img.save(str(out_path), format="JPEG", quality=90)
        return out_path

    except Exception as e:
        logger.warning(f"Could not draw bbox for frame {frame_num}: {e}")
        return None


class VideoEnhancerV2:
    """Enhanced video processor with quality filtering and context awareness."""

    def __init__(
        self,
        vlm_extractor: VLMExtractor,
        data_service: DataService,
        settings: Settings
    ):
        self.vlm = vlm_extractor
        self.data_service = data_service
        self.settings = settings

        # Load prompts
        prompts_dir = Path("prompts")

        frame_quality_path = prompts_dir / "video_frame_quality.txt"
        self.frame_quality_template = (
            frame_quality_path.read_text()
            if frame_quality_path.exists()
            else self._get_default_frame_quality_prompt()
        )

        placement_validation_path = prompts_dir / "video_placement_validation.txt"
        self.placement_validation_template = (
            placement_validation_path.read_text()
            if placement_validation_path.exists()
            else self._get_default_placement_validation_prompt()
        )

        atomic_substeps_path = prompts_dir / "video_atomic_substeps.txt"
        self.atomic_substeps_template = (
            atomic_substeps_path.read_text()
            if atomic_substeps_path.exists()
            else self._get_default_atomic_substeps_prompt()
        )

        placement_reconciliation_path = prompts_dir / "video_placement_reconciliation.txt"
        self.placement_reconciliation_template = (
            placement_reconciliation_path.read_text()
            if placement_reconciliation_path.exists()
            else self._get_default_placement_reconciliation_prompt()
        )

        action_frame_analysis_path = prompts_dir / "video_action_frame_analysis.txt"
        self.action_frame_analysis_template = (
            action_frame_analysis_path.read_text()
            if action_frame_analysis_path.exists()
            else self._get_default_action_frame_analysis_prompt()
        )

        # Initialize quality filter and subassembly tracker
        self.quality_filter = VideoQualityFilter(
            blur_threshold=100.0,
            stability_threshold=0.95,
            hand_detection_enabled=True
        )
        self.subassembly_tracker = SubassemblyTracker()

    # ── public ───────────────────────────────────────────────────────────────

    async def enhance_manual_with_video(
        self,
        manual_id: str,
        video_id: str,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the full 4-pass pipeline and return the video_enhanced.json structure.

        Pipeline:
        1. Extract frames from video (denser sampling: every 5 frames).
        2. VLM Pass 1: Classify frames with quality metrics → placement candidates.
        3. VLM Pass 2: Validate placements with context from enhanced.json + detect duplicates.
        4. VLM Pass 3: Per-placement reconciliation using annotated frames + reference images.
        5. VLM Pass 4: Generate atomic sub-steps (1 part per sub-step) with spatial positioning.

        Args:
            manual_id: Manual identifier
            video_id: Video identifier
            max_frames: Optional limit on frames to process (for testing)
        """
        logger.info(f"Starting video enhancement V2 for manual {manual_id}, video {video_id}")

        # Load manual data (needed for Pass 2+ context)
        manual_data = self.data_service.get_steps(manual_id)
        logger.info(f"Loaded manual data: {len(manual_data['steps'])} steps")

        # Extract frames (denser sampling)
        frames = await self._extract_all_frames(manual_id, video_id, max_frames=max_frames)
        logger.info(f"Extracted {len(frames)} frames (every 5 frames)")

        # === VLM PASS 1: Frame Classification with Quality Metrics ===
        logger.info("VLM Pass 1: Classifying frames with quality assessment...")
        classified_frames = await self._classify_frames_with_quality(frames, manual_id, video_id)
        placement_candidates = [f for f in classified_frames if f["frame_type"] == "placement_candidate"]
        logger.info(
            f"Pass 1 complete: {len(placement_candidates)} high-quality placement candidates, "
            f"{len(classified_frames) - len(placement_candidates)} action frames "
            f"(from {len(frames)} total)"
        )

        # === VLM PASS 2: Context-Aware Placement Validation ===
        logger.info("VLM Pass 2: Validating placements with manual context...")
        validated_placements = await self._validate_placements_with_context(
            placement_candidates, classified_frames, manual_data, manual_id, video_id
        )
        logger.info(f"Pass 2 complete: {len(validated_placements)} unique validated placements")

        # Write validated placements cache
        validated_placements_data = self._write_validated_placements_cache(
            manual_id, video_id, validated_placements, len(placement_candidates)
        )

        # === VLM PASS 3: Per-Placement Reconciliation ===
        logger.info("VLM Pass 3: Reconciling each placement against expected parts...")
        reconciled_placements = await self._reconcile_placements_individually(
            manual_id, video_id, validated_placements_data, manual_data
        )
        logger.info(f"Pass 3 complete: {len(reconciled_placements.get('placements', []))} placements reconciled")

        # Write reconciled placements cache
        reconciled_data = self._write_reconciled_placements_cache(
            manual_id, video_id, reconciled_placements
        )

        # === VLM PASS 4: Atomic Sub-step Generation ===
        logger.info("VLM Pass 4: Generating atomic sub-steps from reconciled placements...")
        enhanced_manual = await self._generate_atomic_substeps(
            manual_id, video_id, reconciled_data, manual_data, frames
        )

        logger.info(f"Video enhancement V2 complete: {len(enhanced_manual['steps'])} steps enhanced")
        return enhanced_manual

    # ── frame extraction ─────────────────────────────────────────────────────

    async def _extract_all_frames(
        self,
        manual_id: str,
        video_id: str,
        max_frames: Optional[int] = None
    ) -> List[Path]:
        """
        Extract frames from video using denser sampling (every 5 frames).

        Args:
            manual_id: Manual ID
            video_id: Video ID
            max_frames: Optional limit on total frames to extract (for testing)
        """
        from backend.services.video_processor import VideoProcessor

        video_path = self.settings.data_dir / "videos" / manual_id / f"{video_id}.mp4"
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        video_processor = VideoProcessor(self.settings)
        frames_dir = video_path.parent / f"{video_id}_enhancement_frames_v2"
        frames_dir.mkdir(exist_ok=True)

        # Extract frames every 5 frames (denser sampling)
        frames = video_processor.extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            frame_interval=5,
            max_frames=max_frames
        )
        return [Path(f["frame_path"]) for f in frames]

    # ── VLM Pass 1: Frame Classification with Quality ────────────────────────

    async def _classify_frames_with_quality(
        self,
        frames: List[Path],
        manual_id: str,
        video_id: str
    ) -> List[Dict[str, Any]]:
        """
        VLM Pass 1: Classify frames as irrelevant/action/placement_candidate with quality metrics.

        Uses batched VLM calls (8-15 frames per call) to reduce API costs.
        Also applies OpenCV quality filtering before VLM analysis.
        Results are cached to video_frame_quality_{video_id}.json.
        """
        # Load cache
        cache_path = (
            self.settings.data_dir / "processed" / manual_id
            / f"video_frame_quality_{video_id}.json"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache: Dict[str, Any] = (
            json.loads(cache_path.read_text()) if cache_path.exists() else {}
        )

        # Separate cached vs uncached
        cached_frames, uncached_frames = self._separate_cached_frames(frames, cache)

        if cache:
            logger.info(
                f"  Pass 1 cache: {len(cached_frames)} frames from cache, "
                f"{len(uncached_frames)} frames to classify"
            )

        newly_classified = []

        if uncached_frames:
            # Apply OpenCV quality pre-filtering
            logger.info("  Pre-filtering frames with OpenCV quality metrics...")
            quality_filter = VideoQualityFilter()
            quality_results = []

            for frame_num, frame_path in uncached_frames:
                metrics = quality_filter.analyze_frame(frame_path)
                quality_results.append((frame_num, frame_path, metrics))

            logger.info(
                f"  OpenCV pre-filter: {sum(1 for _, _, m in quality_results if m['overall_quality'] >= 0.3)} "
                f"frames pass basic quality threshold"
            )

            # Create batches for VLM classification
            batch_size = self.settings.frame_classification_batch_size
            batches = self._create_batches(uncached_frames, batch_size)

            logger.info(
                f"  Classifying {len(uncached_frames)} frames in {len(batches)} "
                f"batches of {batch_size}"
            )

            # Process each batch
            for batch_idx, batch in enumerate(batches):
                try:
                    results = self._classify_frame_batch_with_quality(batch)

                    # Process results and update cache
                    for (frame_num, frame_path), result in zip(batch, results):
                        timestamp = frame_num / 30.0
                        is_relevant = result.get("is_relevant", True)
                        frame_type = result.get("frame_type") if is_relevant else None
                        quality_score = result.get("quality_score", 0.0)
                        has_hand_obstruction = result.get("has_hand_obstruction", False)
                        is_stable = result.get("is_stable", True)
                        confidence = result.get("confidence", 0.0)

                        # Update cache
                        cache[str(frame_num)] = {
                            "is_relevant": is_relevant,
                            "frame_type": frame_type,
                            "quality_score": quality_score,
                            "has_hand_obstruction": has_hand_obstruction,
                            "is_stable": is_stable,
                            "confidence": confidence
                        }

                        # Add to newly classified if relevant and decent quality
                        # Only accept placement_candidates with good quality
                        if is_relevant and frame_type == "placement_candidate":
                            # Strict quality filtering for placement candidates
                            if (quality_score >= 0.5 and
                                not has_hand_obstruction and
                                is_stable):
                                newly_classified.append({
                                    "frame_number": frame_num,
                                    "frame_path": frame_path,
                                    "timestamp": timestamp,
                                    "frame_type": frame_type,
                                    "quality_score": quality_score,
                                    "confidence": confidence
                                })
                            else:
                                reasons = []
                                if quality_score < 0.5:
                                    reasons.append(f"low quality ({quality_score:.2f})")
                                if has_hand_obstruction:
                                    reasons.append("hand obstruction")
                                if not is_stable:
                                    reasons.append("unstable")
                                logger.debug(
                                    f"  Frame {frame_num} classified as placement_candidate "
                                    f"but rejected due to: {', '.join(reasons)}"
                                )
                        elif is_relevant:
                            # Include action frames regardless of quality
                            newly_classified.append({
                                "frame_number": frame_num,
                                "frame_path": frame_path,
                                "timestamp": timestamp,
                                "frame_type": frame_type,
                                "quality_score": quality_score,
                                "confidence": confidence
                            })

                    # Write cache after each batch
                    cache_path.write_text(json.dumps(cache, indent=2))

                    logger.info(
                        f"  Batch {batch_idx + 1}/{len(batches)}: "
                        f"classified {len(batch)} frames"
                    )

                except Exception as e:
                    logger.error(f"Failed to classify batch {batch_idx + 1}: {e}")
                    # Fallback: treat all frames in batch as action frames
                    for frame_num, frame_path in batch:
                        timestamp = frame_num / 30.0
                        cache[str(frame_num)] = {
                            "is_relevant": True,
                            "frame_type": "action",
                            "quality_score": 0.0,
                            "has_hand_obstruction": True,
                            "is_stable": False,
                            "confidence": 0.0
                        }
                        newly_classified.append({
                            "frame_number": frame_num,
                            "frame_path": frame_path,
                            "timestamp": timestamp,
                            "frame_type": "action",
                            "quality_score": 0.0,
                            "confidence": 0.0
                        })
                    cache_path.write_text(json.dumps(cache, indent=2))

        # Merge cached + newly classified
        all_classified = cached_frames + newly_classified

        # Sort by frame_number to maintain temporal order
        all_classified.sort(key=lambda x: x["frame_number"])

        placement_count = sum(1 for f in all_classified if f["frame_type"] == "placement_candidate")
        logger.info(
            f"Pass 1 summary: {len(all_classified)} relevant, "
            f"{placement_count} placement candidates, "
            f"{len(all_classified) - placement_count} action, "
            f"{len(frames) - len(all_classified)} irrelevant"
        )

        return all_classified

    def _separate_cached_frames(
        self,
        frames: List[Path],
        cache: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[tuple[int, Path]]]:
        """Separate frames into cached and uncached."""
        cached_frames = []
        uncached_frames = []

        for frame_path in frames:
            frame_num = int(frame_path.stem.split('_')[-1])
            timestamp = frame_num / 30.0
            key = str(frame_num)

            if key in cache:
                entry = cache[key]
                if entry.get("is_relevant", True):
                    cached_frames.append({
                        "frame_number": frame_num,
                        "frame_path": frame_path,
                        "timestamp": timestamp,
                        "frame_type": entry.get("frame_type", "action"),
                        "quality_score": entry.get("quality_score", 0.0),
                        "confidence": entry.get("confidence", 0.0)
                    })
            else:
                uncached_frames.append((frame_num, frame_path))

        return cached_frames, uncached_frames

    def _create_batches(
        self,
        uncached_frames: List[tuple[int, Path]],
        batch_size: int
    ) -> List[List[tuple[int, Path]]]:
        """Split uncached frames into batches."""
        batches = []
        for i in range(0, len(uncached_frames), batch_size):
            batch = uncached_frames[i:i + batch_size]
            batches.append(batch)
        return batches

    def _classify_frame_batch_with_quality(
        self,
        batch: List[tuple[int, Path]]
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of frames with quality metrics using a single VLM call.

        Returns list of classification results in same order as batch.
        """
        # Build multi-image content payload
        content = []

        # Add frames in order
        for idx, (frame_num, frame_path) in enumerate(batch):
            frame_img = _resize_image(str(frame_path), width=600)
            frame_b64 = _image_to_b64(frame_img)

            content.append({
                "type": "text",
                "text": f"FRAME {idx + 1} of {len(batch)} (frame_number={frame_num}):"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{frame_b64}"}
            })

        # Add prompt at end
        content.append({"type": "text", "text": self.frame_quality_template})

        messages = [{"role": "user", "content": content}]

        # Make VLM call with retry
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        # Validate response is an array
        if not isinstance(result, list):
            raise ValueError(f"Expected array response, got {type(result)}")

        # Validate count matches
        if len(result) != len(batch):
            logger.warning(
                f"Batch size mismatch: sent {len(batch)} frames, "
                f"got {len(result)} classifications. Attempting to recover..."
            )
            # Pad or truncate to match
            while len(result) < len(batch):
                result.append({
                    "is_relevant": True,
                    "frame_type": "action",
                    "quality_score": 0.0,
                    "has_hand_obstruction": True,
                    "is_stable": False,
                    "confidence": 0.0,
                    "reasoning": "Fallback due to incomplete batch response"
                })
            if len(result) > len(batch):
                result = result[:len(batch)]

        # Ensure required fields
        for idx, entry in enumerate(result):
            if not isinstance(entry, dict):
                result[idx] = {
                    "is_relevant": True,
                    "frame_type": "action",
                    "quality_score": 0.0,
                    "has_hand_obstruction": True,
                    "is_stable": False,
                    "confidence": 0.0
                }
            else:
                entry.setdefault("is_relevant", True)
                entry.setdefault("frame_type", "action")
                entry.setdefault("quality_score", 0.0)
                entry.setdefault("has_hand_obstruction", False)
                entry.setdefault("is_stable", True)
                entry.setdefault("confidence", 0.0)

        return result

    # ── VLM Pass 2a: Action Frame Analysis (Sub-call within Pass 2) ──────────

    def _analyze_action_frames(
        self,
        action_frames: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        VLM Pass 2a: Analyze action frames to identify what part is being manipulated.

        This is called WITHIN Pass 2 for each placement candidate, to determine if the
        part being touched is new (pickup) or already placed (adjustment).

        Args:
            action_frames: List of action frame metadata between two placement frames

        Returns:
            {
                "part_being_manipulated": str,
                "color": str,
                "part_type": str,
                "size": str,
                "action_type": "pickup" or "adjustment",
                "confidence": float,
                "reasoning": str
            }
            or None if analysis fails
        """
        if not action_frames:
            return None

        try:
            # Build content with all action frame images
            content = []
            content.append({
                "type": "text",
                "text": f"Analyzing {len(action_frames)} action frames in sequence:"
            })

            for idx, frame_data in enumerate(action_frames, 1):
                frame_path = frame_data.get("frame_path")
                frame_num = frame_data.get("frame_number")
                if frame_path:
                    frame_img = _resize_image(str(frame_path), width=800)
                    frame_b64 = _image_to_b64(frame_img)
                    content.append({
                        "type": "text",
                        "text": f"Action Frame {idx}/{len(action_frames)} (Frame #{frame_num}):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                    })

            # Add prompt
            content.append({
                "type": "text",
                "text": self.action_frame_analysis_template
            })

            # Make VLM call
            messages = [{"role": "user", "content": content}]
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            return result

        except Exception as e:
            logger.warning(f"Action frame analysis failed: {e}")
            return None

    # ── VLM Pass 2: Context-Aware Placement Validation ───────────────────────

    async def _validate_placements_with_context(
        self,
        placement_candidates: List[Dict[str, Any]],
        all_classified_frames: List[Dict[str, Any]],
        manual_data: Dict[str, Any],
        manual_id: str,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """
        VLM Pass 2: Validate placement candidates using context from enhanced.json.

        For each candidate frame:
        - Analyze action frames between placements (Pass 2a sub-call) to identify parts being manipulated
        - Compare against the LAST ACCEPTED placement (not just the preceding candidate)
        - Detect what new part was added using pure visual comparison
        - Check for duplicates using mask-based object tracking (IoU comparison)
        - Track subassembly switches

        Args:
            placement_candidates: Frames classified as placement candidates
            all_classified_frames: ALL classified frames (including action frames)
            manual_data: Enhanced.json data
            manual_id: Manual identifier
            video_id: Video identifier
        """
        if not placement_candidates:
            return []

        # Initialize detailed reasoning log file
        comparison_log_dir = self.settings.data_dir / "processed" / manual_id / f"vlm_reasoning_logs_{video_id}"
        comparison_log_dir.mkdir(parents=True, exist_ok=True)
        comparison_log_path = comparison_log_dir / "placement_reasoning.log"

        # Clear/create new log file at the start
        with open(comparison_log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=" * 100 + "\n")
            log_file.write("VLM PLACEMENT REASONING LOG\n")
            log_file.write(f"Manual ID: {manual_id}\n")
            log_file.write(f"Video ID: {video_id}\n")
            log_file.write(f"Total placement candidates: {len(placement_candidates)}\n")
            log_file.write("=" * 100 + "\n\n")

        logger.info(f"  Detailed VLM reasoning log will be written to: {comparison_log_path}")

        # Build step-indexed parts list from manual
        parts_by_step = self._build_parts_by_step(manual_data)
        current_step = 1  # Start at step 1
        parts_used_in_current_step = 0

        # Use generic LEGO query for YOLO-World + SAM3 detection
        lego_query = get_generic_lego_query()
        logger.info(f"  Using generic LEGO query for YOLO-World + SAM3: '{lego_query}'")

        # Load cache
        cache_path = (
            self.settings.data_dir / "processed" / manual_id
            / f"video_validated_placements_{video_id}.json"
        )

        cached_by_frame: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            existing = json.loads(cache_path.read_text())
            cached_by_frame = {
                str(p["frame_number"]): p
                for p in existing.get("placements", [])
            }
            if cached_by_frame:
                logger.info(
                    f"  Pass 2 cache: {len(cached_by_frame)} placements already validated, resuming..."
                )

        # Reset subassembly tracker
        self.subassembly_tracker.reset()

        validated_placements: List[Dict[str, Any]] = []
        new_calls = 0
        # Tracks the last placement_candidate dict that was accepted into validated_placements
        last_accepted_candidate: Optional[Dict[str, Any]] = None
        # Track masks from previous accepted placement for object-level duplicate detection
        previous_masks: List[np.ndarray] = []

        for i, current_frame in enumerate(placement_candidates):
            frame_num = current_frame["frame_number"]
            timestamp = current_frame["timestamp"]
            key = str(frame_num)

            if key in cached_by_frame:
                cached = {**cached_by_frame[key], "placement_index": i}

                # Ensure frame_path exists (backwards compatibility)
                if "frame_path" not in cached:
                    frame_path_obj = Path(current_frame["frame_path"])
                    try:
                        cached["frame_path"] = str(frame_path_obj.relative_to(self.settings.data_dir))
                    except ValueError:
                        cached["frame_path"] = str(frame_path_obj)

                validated_placements.append(cached)
                last_accepted_candidate = current_frame
                logger.info(
                    f"  Placement {i} (frame {frame_num}): "
                    f"[cached] {cached.get('action_description', 'no description')}"
                )
                continue

            try:
                # Initialize variables that will be used later
                current_detections = []
                current_annotated_path = None
                current_masks = []

                # === YOLO-WORLD + SAM3 OBJECT DETECTION === (TEMPORARILY DISABLED FOR VLM-ONLY TESTING)
                # if lego_query:
                #     import cv2
                #     img = cv2.imread(str(current_frame["frame_path"]))
                #     if img is not None:
                #         img_h, img_w = img.shape[:2]
                #         try:
                #             detections = detect_objects_yolo_world_sam3(
                #                 str(current_frame["frame_path"]),
                #                 lego_query,
                #                 self.settings.roboflow_api_key,
                #                 (img_h, img_w)
                #             )
                #
                #             if detections:
                #                 # Extract masks for object tracking (ignore labels for now)
                #                 current_masks = [obj["mask"] for obj in detections if obj.get("mask") is not None]
                #
                #                 # Check if any NEW objects were detected using mask IoU comparison
                #                 new_mask_indices = find_new_masks(current_masks, previous_masks, iou_threshold=0.3)
                #
                #                 if not new_mask_indices and previous_masks:
                #                     logger.info(
                #                         f"  Placement {i} (frame {frame_num}): "
                #                         f"No new objects detected via mask tracking (all {len(current_masks)} masks match previous) - skipping VLM call"
                #                     )
                #                     continue
                #                 elif new_mask_indices:
                #                     logger.debug(
                #                         f"  Frame {frame_num}: Detected {len(new_mask_indices)} new masks out of {len(current_masks)} total"
                #                     )
                #
                #                 # Group by label and count
                #                 label_counts = {}
                #                 label_data = {}
                #                 for obj in detections:
                #                     label = obj["label"]
                #                     if label not in label_counts:
                #                         label_counts[label] = 0
                #                         label_data[label] = obj
                #                     label_counts[label] += 1
                #
                #                 # Build grouped detections with counts
                #                 grouped_detections = []
                #                 for label, count in label_counts.items():
                #                     obj = label_data[label].copy()
                #                     obj["count"] = count
                #                     grouped_detections.append(obj)
                #
                #                 current_detections = grouped_detections
                #
                #                 # Annotate frame
                #                 annotated_dir = (
                #                     self.settings.data_dir / "processed" / manual_id
                #                     / f"yolo_world_sam3_annotated_{video_id}"
                #                 )
                #                 annotated_dir.mkdir(parents=True, exist_ok=True)
                #
                #                 annotated_path = annotated_dir / f"frame_{frame_num:04d}.jpg"
                #                 annotate_frame_with_objects(
                #                     Path(current_frame["frame_path"]),
                #                     grouped_detections,
                #                     annotated_path,
                #                     previous_placements=None
                #                 )
                #                 current_annotated_path = annotated_path
                #         except Exception as e:
                #             logger.warning(f"YOLO-World + SAM3 detection failed for frame {frame_num}: {e}")

                # === VLM PASS 2a: Analyze action frames between placements ===
                action_frame_analysis = None
                action_frames_between = []

                if last_accepted_candidate is not None:
                    # Get the frame numbers
                    prev_placement_frame_num = last_accepted_candidate.get("frame_number")
                    current_placement_frame_num = frame_num

                    # Find action frames between the two placements
                    for frame in all_classified_frames:
                        f_num = frame.get("frame_number")
                        f_type = frame.get("frame_type")

                        # Action frame that's between the two placements
                        if (f_type == "action" and
                            f_num > prev_placement_frame_num and
                            f_num < current_placement_frame_num):
                            action_frames_between.append(frame)

                    # If no action frames between placements, likely a duplicate
                    if not action_frames_between:
                        logger.info(
                            f"  Placement {i} (frame {frame_num}): "
                            f"No action frames between prev frame {prev_placement_frame_num} and current - likely duplicate, skipping VLM call"
                        )
                        continue

                    # Analyze action frames to identify what part is being manipulated
                    logger.debug(f"  Frame {frame_num}: Analyzing {len(action_frames_between)} action frames between placements")
                    action_frame_analysis = self._analyze_action_frames(action_frames_between)

                    if action_frame_analysis:
                        logger.debug(
                            f"  Action frame analysis: {action_frame_analysis.get('part_being_manipulated')} "
                            f"({action_frame_analysis.get('action_type')})"
                        )

                # Build object count summary
                object_count_summary = ""
                if current_detections:
                    total_objects = sum(obj['count'] for obj in current_detections)
                    object_count_summary = f"**CURRENT FRAME OBJECT COUNTS (YOLO-World detected {total_objects} total LEGO parts):**\n"
                    for obj in current_detections:
                        object_count_summary += f"- {obj['label']}: {obj['count']} detected\n"
                    object_count_summary += "\n"

                # Build previous placement context
                previous_placement_context = ""
                if validated_placements:
                    last_placements = validated_placements[-3:]
                    previous_placement_context = "**PREVIOUS PLACEMENTS (what was already placed):**\n"
                    for idx, placement in enumerate(last_placements, 1):
                        previous_placement_context += f"{idx}. {placement.get('action_description', 'N/A')}\n"
                    previous_placement_context += "\nIf current frame shows the SAME total number of objects as expected from previous placements, then NO new part was added.\n\n"

                # Build action frame context
                action_frame_context = ""
                if action_frame_analysis:
                    action_type = action_frame_analysis.get('action_type', 'unknown')
                    part_desc = action_frame_analysis.get('part_being_manipulated', 'unknown part')
                    reasoning = action_frame_analysis.get('reasoning', '')

                    action_frame_context = f"**ACTION FRAMES BETWEEN PLACEMENTS ({len(action_frames_between)} frames analyzed):**\n"
                    action_frame_context += f"Part being manipulated: {part_desc}\n"
                    action_frame_context += f"Action type: {action_type.upper()}\n"

                    if action_type == "adjustment":
                        action_frame_context += f"Note: This part ({part_desc}) was already placed and is being adjusted.\n"
                    elif action_type == "pickup":
                        action_frame_context += f"Note: This part ({part_desc}) is being picked up for a new placement.\n"

                    action_frame_context += f"Reasoning: {reasoning}\n\n"

                # Use annotated frame if available
                current_img_path = current_annotated_path if current_annotated_path else current_frame["frame_path"]
                current_img = _resize_image(str(current_img_path), width=800)
                current_b64 = _image_to_b64(current_img)

                # Build prompt with context
                prompt_with_context = (
                    previous_placement_context +
                    action_frame_context +
                    object_count_summary +
                    self.placement_validation_template
                )

                # Build content with previous and current frames only
                if i == 0:
                    # First placement - no previous frame
                    content = [
                        {"type": "text", "text": "CURRENT PLACEMENT FRAME (first placement — no previous frame):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                        {"type": "text", "text": prompt_with_context}
                    ]
                else:
                    # Use the last ACCEPTED placement as the reference frame so that
                    # we always compare against a known-good state, not a skipped candidate.
                    prev_frame = last_accepted_candidate if last_accepted_candidate is not None else placement_candidates[i - 1]
                    prev_img = _resize_image(str(prev_frame["frame_path"]), width=800)
                    prev_b64 = _image_to_b64(prev_img)
                    prev_frame_num = prev_frame.get("frame_number", "?")

                    frame_annotation_note = " — with YOLO-World + SAM3 annotations" if current_annotated_path else ""
                    content = [
                        {"type": "text", "text": f"PREVIOUS PLACEMENT FRAME (frame {prev_frame_num} — last accepted placement):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_b64}"}},
                        {"type": "text", "text": f"CURRENT PLACEMENT FRAME (frame {frame_num}){frame_annotation_note}:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                        {"type": "text", "text": prompt_with_context}
                    ]

                messages = [{"role": "user", "content": content}]

                raw = self.vlm._litellm_with_retry(messages)
                result = _parse_json(raw)

                has_new_part = result.get("has_new_part", False)
                is_duplicate = result.get("is_duplicate_of_previous", False)
                confidence = result.get("confidence", 0.0)

                # === DETAILED LOGGING FOR VLM REASONING ===
                # Write detailed comparison log to file
                comparison_log_dir = self.settings.data_dir / "processed" / manual_id / f"vlm_reasoning_logs_{video_id}"
                comparison_log_dir.mkdir(parents=True, exist_ok=True)
                comparison_log_path = comparison_log_dir / "placement_reasoning.log"

                with open(comparison_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("=" * 100 + "\n")
                    log_file.write(f"PLACEMENT CANDIDATE {i} - Frame {frame_num} (timestamp: {timestamp:.2f}s)\n")
                    log_file.write("=" * 100 + "\n\n")

                    # Previous frame info
                    if i == 0:
                        log_file.write("PREVIOUS FRAME: None (this is the first placement)\n\n")
                    else:
                        prev_frame = last_accepted_candidate if last_accepted_candidate is not None else placement_candidates[i - 1]
                        prev_frame_num = prev_frame.get("frame_number", "?")
                        prev_timestamp = prev_frame.get("timestamp", 0.0)
                        log_file.write(f"PREVIOUS FRAME: Frame {prev_frame_num} (timestamp: {prev_timestamp:.2f}s)\n")
                        log_file.write(f"  Path: {prev_frame.get('frame_path', 'N/A')}\n\n")

                    # Previous placement (most recent only)
                    log_file.write("PREVIOUS PLACEMENT CONTEXT:\n")
                    if validated_placements:
                        last_placement = validated_placements[-1]
                        log_file.write(f"  Frame {last_placement.get('frame_number')}: {last_placement.get('action_description', 'N/A')}\n")
                    else:
                        log_file.write("  (No previous placements)\n")
                    log_file.write("\n")

                    # Action frame analysis (VLM Pass 2a)
                    log_file.write(f"ACTION FRAMES BETWEEN PLACEMENTS: {len(action_frames_between)} frames\n")
                    if action_frames_between:
                        log_file.write(f"  Frame numbers: {[f.get('frame_number') for f in action_frames_between]}\n")
                    if action_frame_analysis:
                        log_file.write("VLM PASS 2a - ACTION FRAME ANALYSIS:\n")
                        log_file.write(f"  part_being_manipulated: {action_frame_analysis.get('part_being_manipulated', 'N/A')}\n")
                        log_file.write(f"  color: {action_frame_analysis.get('color', 'N/A')}\n")
                        log_file.write(f"  part_type: {action_frame_analysis.get('part_type', 'N/A')}\n")
                        log_file.write(f"  size: {action_frame_analysis.get('size', 'N/A')}\n")
                        log_file.write(f"  action_type: {action_frame_analysis.get('action_type', 'N/A').upper()}\n")
                        log_file.write(f"  confidence: {action_frame_analysis.get('confidence', 0.0):.2f}\n")
                        log_file.write(f"  reasoning: {action_frame_analysis.get('reasoning', 'N/A')}\n")
                    else:
                        log_file.write("  (No action frame analysis)\n")
                    log_file.write("\n")

                    # Current frame info
                    log_file.write(f"CURRENT FRAME: Frame {frame_num} (timestamp: {timestamp:.2f}s)\n")
                    log_file.write(f"  Path: {current_frame.get('frame_path', 'N/A')}\n\n")

                    # VLM Decision
                    log_file.write("VLM PASS 2b - PLACEMENT VALIDATION DECISION:\n")
                    log_file.write(f"  has_new_part: {has_new_part}\n")
                    log_file.write(f"  is_duplicate_of_previous: {is_duplicate}\n")
                    log_file.write(f"  confidence: {confidence:.2f}\n")
                    log_file.write(f"  action_description: {result.get('action_description', 'N/A')}\n")
                    log_file.write(f"  new_parts_added: {result.get('new_parts_added', [])}\n")
                    log_file.write(f"  reasoning: {result.get('reasoning', 'N/A')}\n")
                    log_file.write(f"  visual_difference: {result.get('visual_difference', 'N/A')}\n\n")

                    # Final verdict
                    if is_duplicate or not has_new_part:
                        log_file.write("VERDICT: REJECTED (duplicate or no new part)\n")
                    elif confidence < getattr(self.settings, "placement_min_confidence", 0.6):
                        log_file.write(f"VERDICT: REJECTED (low confidence: {confidence:.2f})\n")
                    else:
                        log_file.write("VERDICT: ACCEPTED\n")

                    log_file.write("\n\n")

                if is_duplicate or not has_new_part:
                    logger.info(
                        f"  Placement {i} (frame {frame_num}): "
                        f"Duplicate or no new part detected, skipping"
                    )
                    continue

                # Confidence gate: reject low-confidence claims to reduce hallucinations.
                # Threshold is configurable via settings (default 0.6).
                min_confidence = getattr(self.settings, "placement_min_confidence", 0.6)
                if confidence < min_confidence:
                    logger.info(
                        f"  Placement {i} (frame {frame_num}): "
                        f"Low confidence ({confidence:.2f} < {min_confidence}), skipping"
                    )
                    continue

                action_desc = result.get("action_description")

                if action_desc:
                    # Check for subassembly switch
                    is_subassembly_switch = self.subassembly_tracker.detect_subassembly_switch(
                        action_desc, frame_num
                    )

                    # Get relative path from data_dir for storage
                    frame_path_obj = Path(current_frame["frame_path"])
                    try:
                        relative_frame_path = str(frame_path_obj.relative_to(self.settings.data_dir))
                    except ValueError:
                        # If not relative to data_dir, use absolute path
                        relative_frame_path = str(frame_path_obj)

                    # Draw annotated frame with bounding box if the VLM returned one
                    box_2d = result.get("box_2d")
                    annotated_frame_path = _draw_placement_bbox(
                        frame_path=current_frame["frame_path"],
                        box_2d=box_2d,
                        frame_num=frame_num,
                        label=action_desc,
                        confidence=confidence,
                        output_dir=self.settings.data_dir / "processed" / manual_id / f"validated_placement_annotated_{video_id}"
                    )

                    action_data: Dict[str, Any] = {
                        "placement_index": i,
                        "frame_number": frame_num,
                        "timestamp": timestamp,
                        "frame_path": relative_frame_path,
                        "action_description": action_desc,
                        "new_parts": result.get("new_parts_added", []),
                        "spatial_position": result.get("spatial_position", {}),
                        "box_2d": box_2d,
                        "annotated_frame_path": str(annotated_frame_path.relative_to(self.settings.data_dir)) if annotated_frame_path else None,
                        "yolo_world_sam3_annotated_path": str(current_annotated_path.relative_to(self.settings.data_dir)) if current_annotated_path else None,
                        "yolo_world_detections": [{"label": d["label"], "count": d["count"], "bbox": d["bbox"]} for d in current_detections] if current_detections else [],
                        "is_subassembly_switch": is_subassembly_switch,
                        "current_subassembly": self.subassembly_tracker.get_current_subassembly(),
                        "manual_step": current_step,
                        "confidence": confidence
                    }
                    validated_placements.append(action_data)
                    last_accepted_candidate = current_frame

                    # Update mask tracking for next frame comparison
                    if current_masks:
                        previous_masks = current_masks.copy()
                        logger.debug(f"  Updated previous_masks with {len(previous_masks)} masks from frame {frame_num}")

                    # Track parts usage and advance step if needed
                    parts_used_in_current_step += 1
                    step_data = parts_by_step.get(current_step, {})
                    expected_parts_count = step_data.get("total_individual_parts", 0)
                    if parts_used_in_current_step >= expected_parts_count and expected_parts_count > 0:
                        logger.info(f"  Step {current_step} complete ({parts_used_in_current_step}/{expected_parts_count} parts placed), advancing to step {current_step + 1}")
                        current_step += 1
                        parts_used_in_current_step = 0

                    # Write cache incrementally
                    self._write_validated_placements_cache(
                        manual_id, video_id, validated_placements, len(placement_candidates)
                    )

                    new_calls += 1
                    logger.info(f"  Placement {i} (frame {frame_num}, step {current_step}): {action_desc}")
                else:
                    logger.info(f"  Placement {i} (frame {frame_num}): no action description")

            except Exception as e:
                logger.error(f"Failed to validate placement frame {frame_num}: {e}")

        cached_count = len(validated_placements) - new_calls
        logger.info(
            f"  Pass 2 done: {new_calls} new VLM calls, {cached_count} from cache, "
            f"{len(placement_candidates) - len(validated_placements)} duplicates filtered"
        )
        return validated_placements

    def _build_parts_by_step(self, manual_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Build a step-indexed dictionary of expected parts and subassemblies.

        Returns:
            Dictionary mapping step_number -> {
                "parts": list of parts for that step,
                "total_individual_parts": total quantity of individual parts,
                "subassemblies": list of subassembly images,
                "actions": list of actions for that step
            }
        """
        parts_by_step: Dict[int, Dict[str, Any]] = {}

        for step in manual_data.get("steps", []):
            step_num = step.get("step_number")
            parts = []
            total_quantity = 0

            for part in step.get("parts_required", []):
                description = part.get("description", "")
                quantity = _parse_quantity_from_description(description)
                total_quantity += quantity

                parts.append({
                    "description": description,
                    "quantity": quantity,
                    "cropped_image_path": part.get("cropped_image_path"),
                })

            subassemblies = []
            for subassembly in step.get("subassemblies", []):
                subassemblies.append({
                    "description": subassembly.get("description"),
                    "cropped_image_path": subassembly.get("cropped_image_path"),
                })

            parts_by_step[step_num] = {
                "parts": parts,
                "total_individual_parts": total_quantity,
                "subassemblies": subassemblies,
                "actions": step.get("actions", [])
            }

        return parts_by_step

    def _get_expected_parts_for_steps(
        self,
        parts_by_step: Dict[int, Dict[str, Any]],
        current_step: int,
        lookahead_steps: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get expected parts for current step + lookahead steps.

        This prevents VLM from jumping to later steps by only showing relevant parts.

        Args:
            parts_by_step: Step-indexed parts dictionary
            current_step: Current manual step number
            lookahead_steps: How many steps ahead to include (default 1)

        Returns:
            List of parts for current + lookahead steps
        """
        expected_parts = []

        for step_num in range(current_step, current_step + lookahead_steps + 1):
            if step_num in parts_by_step:
                step_data = parts_by_step[step_num]
                for part in step_data["parts"]:
                    expected_parts.append({
                        "step": step_num,
                        "description": part.get("description"),
                        "quantity": part.get("quantity"),
                        "cropped_image_path": part.get("cropped_image_path"),
                    })

        return expected_parts

    def _write_validated_placements_cache(
        self,
        manual_id: str,
        video_id: str,
        validated_placements: List[Dict[str, Any]],
        total_candidates: int
    ) -> Dict[str, Any]:
        """Write validated placements to cache."""
        validated_data = {
            "manual_id": manual_id,
            "video_id": video_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_placement_candidates": total_candidates,
            "placements": validated_placements
        }

        output_path = (
            self.settings.data_dir / "processed" / manual_id
            / f"video_validated_placements_{video_id}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(validated_data, f, indent=2)

        return validated_data

    def _write_reconciled_placements_cache(
        self,
        manual_id: str,
        video_id: str,
        reconciled_placements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write reconciled placements to cache."""
        output_path = (
            self.settings.data_dir / "processed" / manual_id
            / f"video_reconciled_placements_{video_id}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(reconciled_placements, f, indent=2)

        logger.info(f"Reconciled placements written to: {output_path}")
        return reconciled_placements

    # ── VLM Pass 3: Per-Placement Reconciliation ─────────────────────────────

    async def _reconcile_single_placement(
        self,
        placement: Dict[str, Any],
        expected_parts: List[Dict[str, Any]],
        step_number: int,
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Reconcile a single placement against expected parts using VLM analysis.

        Args:
            placement: Single validated placement with annotated_frame_path
            expected_parts: List of expected parts for the current step
            step_number: Current manual step number
            manual_id: Manual identifier for loading images

        Returns:
            Reconciliation result with verified part information
        """
        # Build content for VLM call
        content: List[Dict[str, Any]] = []

        # Get original frame path and box_2d for cropping
        frame_path = placement.get("frame_path")
        box_2d = placement.get("box_2d")

        # Construct full frame path
        if frame_path:
            full_frame_path = self.settings.data_dir / frame_path
        else:
            full_frame_path = None

        # 1. Add full annotated frame (with bounding box) for context
        annotated_frame_path = placement.get("annotated_frame_path")
        if annotated_frame_path:
            full_annotated_path = self.settings.data_dir / annotated_frame_path
            if full_annotated_path.exists():
                try:
                    annotated_img = _resize_image(str(full_annotated_path), width=800)
                    annotated_b64 = _image_to_b64(annotated_img)
                    content.append({
                        "type": "text",
                        "text": f"VIEW 1 - FULL FRAME CONTEXT (Placement {placement['placement_index']}, Frame {placement['frame_number']}):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{annotated_b64}"}
                    })
                except Exception as e:
                    logger.warning(f"Could not load annotated frame {annotated_frame_path}: {e}")
            else:
                logger.warning(f"Annotated frame not found: {full_annotated_path}")

        # 2. Add CROPPED view (tight crop to bounding box) - THIS IS CRITICAL FOR STUD COUNTING
        if full_frame_path and full_frame_path.exists() and box_2d:
            cropped_img = _crop_to_bbox(str(full_frame_path), box_2d, padding=30)
            if cropped_img:
                try:
                    # Resize cropped image to reasonable size (keeping aspect ratio)
                    max_size = 600
                    if cropped_img.width > max_size or cropped_img.height > max_size:
                        if cropped_img.width > cropped_img.height:
                            new_w = max_size
                            new_h = int(max_size * cropped_img.height / cropped_img.width)
                        else:
                            new_h = max_size
                            new_w = int(max_size * cropped_img.width / cropped_img.height)
                        cropped_img = cropped_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

                    cropped_b64 = _image_to_b64(cropped_img)
                    content.append({
                        "type": "text",
                        "text": "VIEW 2 - ZOOMED TO PART (cropped to bounding box - USE THIS FOR STUD COUNTING):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{cropped_b64}"}
                    })
                    logger.debug(f"Added cropped bbox view for placement {placement['placement_index']}")
                except Exception as e:
                    logger.warning(f"Could not process cropped image: {e}")
            else:
                logger.debug(f"Could not crop to bbox for placement {placement['placement_index']}")
        else:
            logger.debug(f"Skipping bbox crop - frame_path: {full_frame_path}, box_2d: {box_2d}")

        # 3. Add expected parts reference images
        if expected_parts:
            content.append({
                "type": "text",
                "text": f"VIEW 3 - EXPECTED PARTS FOR STEP {step_number} (reference images from manual):"
            })
            for idx, part in enumerate(expected_parts):
                part_img_path = part.get("cropped_image_path")
                if part_img_path:
                    full_part_path = self.settings.data_dir / part_img_path
                    if full_part_path.exists():
                        try:
                            part_img = _resize_image(str(full_part_path), width=400)
                            part_b64 = _image_to_b64(part_img)
                            content.append({
                                "type": "text",
                                "text": f"Expected Part {idx + 1}: {part['description']}"
                            })
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{part_b64}"}
                            })
                        except Exception as e:
                            logger.warning(f"Could not load part image {part_img_path}: {e}")

        # 4. Build placement metadata JSON
        placement_metadata = {
            "placement_index": placement["placement_index"],
            "frame_number": placement["frame_number"],
            "timestamp": placement["timestamp"],
            "video_detected_parts": placement.get("new_parts", []),
            "action_description": placement.get("action_description", ""),
            "spatial_position": placement.get("spatial_position", {}),
            "confidence": placement.get("confidence", 0.0)
        }

        # 5. Build expected parts JSON (without images)
        expected_parts_json = [
            {
                "description": part["description"],
                "quantity": part.get("quantity", 1)
            }
            for part in expected_parts
        ]

        # 6. Build prompt from template
        prompt = self.placement_reconciliation_template.replace(
            "{placement_metadata}", json.dumps(placement_metadata, indent=2)
        ).replace(
            "{expected_parts}", json.dumps(expected_parts_json, indent=2)
        ).replace(
            "{step_number}", str(step_number)
        )

        content.append({"type": "text", "text": prompt})

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        return result

    async def _reconcile_placements_individually(
        self,
        manual_id: str,
        video_id: str,
        validated_placements: Dict[str, Any],
        manual_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        VLM Pass 3: Reconcile each placement individually against expected parts.

        Uses annotated frames + reference images to verify/correct part identifications.
        Returns reconciled placements with verified part information.
        """
        logger.info("VLM Pass 3: Starting per-placement reconciliation...")

        # Build parts lookup by step
        parts_by_step = self._build_parts_by_step(manual_data)

        # Process all placements
        placements = validated_placements.get("placements", [])
        reconciled_placements = []
        total_reconciled = 0

        for placement in placements:
            step_num = placement.get("manual_step", 1)
            expected_parts_data = parts_by_step.get(step_num, {}).get("parts", [])

            try:
                # Perform per-placement reconciliation
                reconciliation = await self._reconcile_single_placement(
                    placement=placement,
                    expected_parts=expected_parts_data,
                    step_number=step_num,
                    manual_id=manual_id
                )

                # Enrich placement with reconciliation data
                enriched_placement = {
                    **placement,
                    "reconciliation": {
                        "verified": reconciliation.get("verified", False),
                        "matched_part": reconciliation.get("matched_part"),
                        "video_detection_correct": reconciliation.get("video_detection_correct", True),
                        "correction": reconciliation.get("correction"),
                        "reasoning": reconciliation.get("reasoning", "")
                    }
                }

                reconciled_placements.append(enriched_placement)
                total_reconciled += 1

                verified = reconciliation.get("verified", False)
                video_correct = reconciliation.get("video_detection_correct", True)
                matched_part = reconciliation.get("matched_part")
                part_desc = matched_part.get("description", "unknown") if matched_part else "none"

                logger.info(
                    f"  Placement {placement['placement_index']} (frame {placement['frame_number']}, step {step_num}): "
                    f"{'✓ verified' if verified else '✗ unverified'} | "
                    f"{'✓ correct' if video_correct else '✗ corrected'} | "
                    f"{part_desc}"
                )

            except Exception as e:
                logger.error(f"Failed to reconcile placement {placement['placement_index']}: {e}")
                # Fallback: keep original placement without reconciliation
                reconciled_placements.append({
                    **placement,
                    "reconciliation": {
                        "verified": False,
                        "matched_part": None,
                        "video_detection_correct": False,
                        "correction": None,
                        "reasoning": f"Reconciliation failed: {str(e)}"
                    }
                })

        logger.info(f"VLM Pass 3 complete: {total_reconciled} placements reconciled individually")

        return {
            "manual_id": manual_id,
            "video_id": video_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_placements": len(reconciled_placements),
            "placements": reconciled_placements
        }

    # ── VLM Pass 4: Atomic Sub-step Generation ───────────────────────────────

    async def _generate_atomic_substeps(
        self,
        manual_id: str,
        video_id: str,
        reconciled_placements: Dict[str, Any],
        manual_data: Dict[str, Any],
        all_frames: List[Path]
    ) -> Dict[str, Any]:
        """
        VLM Pass 4: Generate atomic sub-steps (1 part per sub-step) with spatial positioning.

        Uses reconciled placements to create the final video_enhanced.json structure.
        Sends enhanced.json + reconciled_placements.json as text, plus sample frames.
        """
        enhanced_json_str = json.dumps(manual_data, indent=2)
        reconciled_json_str = json.dumps(reconciled_placements, indent=2)

        prompt = self.atomic_substeps_template.replace(
            "{enhanced_json}", enhanced_json_str
        ).replace(
            "{validated_placements_json}", reconciled_json_str
        )

        # Build a frame_number → Path lookup for sample images
        frame_lookup: Dict[int, Path] = {
            int(f.stem.split('_')[-1]): f for f in all_frames
        }

        # Pick up to 5 evenly-spaced placements as visual context
        placements = reconciled_placements.get("placements", [])
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

                        # Include reconciliation info if available
                        reconciliation = p.get("reconciliation", {})
                        matched_part = reconciliation.get("matched_part", {})
                        part_info = matched_part.get("description", "unknown") if matched_part else "unknown"

                        content.append({
                            "type": "text",
                            "text": (
                                f"Placement {p['placement_index']} "
                                f"(frame {p['frame_number']}, step {p.get('manual_step', '?')}): "
                                f"{p.get('action_description', '')} "
                                f"[Reconciled: {part_info}]"
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

        # Build mapping from placement sequence to frame paths
        placements_by_step_and_order: Dict[int, List[Dict[str, Any]]] = {}
        for placement in placements:
            step_num = placement.get("manual_step", 1)
            if step_num not in placements_by_step_and_order:
                placements_by_step_and_order[step_num] = []
            placements_by_step_and_order[step_num].append(placement)

        # Merge with original manual data and enrich sub-steps with frame paths + reconciliation
        reconciled_by_num = {
            s["step_number"]: s for s in result.get("steps", [])
        }

        enhanced_steps = []
        for step in manual_data["steps"]:
            step_num = step["step_number"]
            reconciled = reconciled_by_num.get(step_num, {})

            # Get sub-steps from VLM
            sub_steps = reconciled.get("sub_steps", [])

            # Get placements for this step
            step_placements = placements_by_step_and_order.get(step_num, [])

            # Enrich each sub-step with frame path and APPLY reconciliation corrections
            enriched_sub_steps = []
            for idx, sub_step in enumerate(sub_steps):
                # Match sub-step to placement by order (assuming VLM maintains sequence)
                if idx < len(step_placements):
                    placement = step_placements[idx]
                    reconciliation = placement.get("reconciliation", {})

                    # Extract corrected part information from reconciliation
                    matched_part = reconciliation.get("matched_part")
                    verified = reconciliation.get("verified", False)

                    # Use reconciled part description if available, otherwise fall back to sub_step
                    if matched_part and verified:
                        # Use the CORRECTED part from reconciliation
                        corrected_part_desc = matched_part.get("description", sub_step.get("parts_involved", ["unknown"])[0])
                        corrected_confidence = matched_part.get("confidence", sub_step.get("confidence", 0.0))

                        enriched_sub_steps.append({
                            **sub_step,
                            "parts_involved": [corrected_part_desc],  # Use reconciled part
                            "confidence": corrected_confidence,  # Use reconciled confidence
                            "frame_path": placement.get("frame_path"),
                            "frame_number": placement.get("frame_number"),
                            "timestamp": placement.get("timestamp"),
                            "verified": verified  # Keep minimal metadata: true/false
                        })
                    else:
                        # Reconciliation failed or not verified - use original sub_step data
                        enriched_sub_steps.append({
                            **sub_step,
                            "frame_path": placement.get("frame_path"),
                            "frame_number": placement.get("frame_number"),
                            "timestamp": placement.get("timestamp"),
                            "verified": False
                        })
                else:
                    # No matching placement found
                    enriched_sub_steps.append(sub_step)

            # Build enhanced step (corrections are already applied in sub_steps)
            step_data = {
                **step,
                "sub_steps": enriched_sub_steps
            }

            enhanced_steps.append(step_data)

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

    # ── fallback prompts ──────────────────────────────────────────────────────

    def _get_default_frame_quality_prompt(self) -> str:
        return "Classify frames with quality metrics. Return JSON array."

    def _get_default_placement_validation_prompt(self) -> str:
        return "Validate placements with context. Return JSON."

    def _get_default_atomic_substeps_prompt(self) -> str:
        return "Generate atomic sub-steps. Return JSON."

    def _get_default_placement_reconciliation_prompt(self) -> str:
        return "Reconcile placement against expected parts. Return JSON with verified, matched_part, video_detection_correct, correction, and reasoning fields."

    def _get_default_action_frame_analysis_prompt(self) -> str:
        return "Analyze action frames to identify what LEGO part is being manipulated. Return JSON with part_being_manipulated, color, part_type, size, action_type (pickup/adjustment), confidence, and reasoning."
