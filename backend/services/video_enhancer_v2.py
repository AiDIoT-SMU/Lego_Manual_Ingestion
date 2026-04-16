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
import os
import cv2
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
from config.settings import Settings

# Import SAM3 detector
import sys
sam3_path = Path(__file__).parent.parent.parent / "archive" / "yolo_world_sam3"
if str(sam3_path) not in sys.path:
    sys.path.insert(0, str(sam3_path))

from yolo_world_sam3_detector import (
    call_sam3_api,
    annotate_frame_with_objects,
    SAM3_CONFIDENCE_THRESHOLD
)


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


def _apply_sam3_segmentation(
    image_path: str,
    output_path: Path,
    api_key: str,
    text_prompt: str = "lego assembly"
) -> Optional[Path]:
    """
    Apply SAM3 segmentation to crop out the LEGO assembly with white background.

    Uses SAM3 to detect the entire LEGO assembly region, crops it out, and places it
    on a white background to remove noise (table, hands, other objects). This gives
    the VLM a clean, focused view of just the assembly for better analysis.

    Args:
        image_path: Path to source image
        output_path: Path to save cropped assembly on white background
        api_key: Roboflow API key
        text_prompt: Text prompt for SAM3 (default: "lego assembly")

    Returns:
        Path to cropped image with white background, or None if segmentation fails
    """
    try:
        # Call SAM3 API with text prompt
        sam3_response = call_sam3_api(
            image_path=str(image_path),
            text_prompts=[text_prompt],
            api_key=api_key,
            confidence_threshold=SAM3_CONFIDENCE_THRESHOLD
        )

        if not sam3_response:
            logger.warning(f"SAM3 returned no response for {image_path}")
            return None

        # Load the original image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not load image {image_path}")
            return None

        # Parse SAM3 response and extract the assembly region
        prompt_results = sam3_response.get("prompt_results", [])
        if not prompt_results:
            logger.warning(f"SAM3 returned no prompt_results for {image_path}")
            return None

        # Get the first (and typically only) detection - the entire assembly
        predictions = prompt_results[0].get("predictions", [])
        if not predictions:
            logger.warning(f"SAM3 detected no assembly in {image_path}")
            return None

        # Use the first prediction (highest confidence)
        pred = predictions[0]
        masks = pred.get("masks", [])
        if not masks:
            logger.warning(f"SAM3 prediction has no mask for {image_path}")
            return None

        # Get bounding box from polygon mask
        polygon = masks[0]  # [[x, y], [x, y], ...]
        pts = np.array(polygon, dtype=np.int32)
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding around the assembly (15% on each side for context)
        height, width = img.shape[:2]
        padding_x = max(20, int((x_max - x_min) * 0.15))
        padding_y = max(20, int((y_max - y_min) * 0.15))

        x_min_padded = max(0, x_min - padding_x)
        x_max_padded = min(width, x_max + padding_x)
        y_min_padded = max(0, y_min - padding_y)
        y_max_padded = min(height, y_max + padding_y)

        # Create a binary mask from the polygon
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Crop both the image and mask to the padded bounding box
        cropped_img = img[y_min_padded:y_max_padded, x_min_padded:x_max_padded].copy()
        cropped_mask = mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

        if cropped_img.size == 0:
            logger.warning(f"SAM3 crop resulted in empty image for {image_path}")
            return None

        # Create white background with same dimensions as cropped region
        white_bg = np.ones_like(cropped_img, dtype=np.uint8) * 255

        # Apply mask to extract only the assembly pixels
        # Where mask is 255 (assembly), use original pixels; where 0, use white
        cropped_mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)
        result = np.where(cropped_mask_3ch > 0, cropped_img, white_bg)

        # Save the result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        logger.debug(f"SAM3 cropped assembly (white background) saved to {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"SAM3 segmentation failed for {image_path}: {e}")
        return None


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

        # Load Pass 1a prompt (frame classification with temporal context)
        pass1a_path = prompts_dir / "video_pass1a_classification.txt"
        self.pass1a_template = (
            pass1a_path.read_text()
            if pass1a_path.exists()
            else self._get_default_frame_quality_prompt()  # Fallback to old prompt
        )

        # Load Pass 1b prompt (quality filtering)
        pass1b_path = prompts_dir / "video_pass1b_quality_filter.txt"
        self.pass1b_template = (
            pass1b_path.read_text()
            if pass1b_path.exists()
            else ""
        )

        # Load Pass 2a prompt (action analysis)
        pass2a_path = prompts_dir / "video_pass2a_action_analysis.txt"
        self.pass2a_template = (
            pass2a_path.read_text()
            if pass2a_path.exists()
            else ""
        )

        # Load Pass 2b prompt (SAM3 comparison)
        pass2b_path = prompts_dir / "video_pass2b_sam3_comparison.txt"
        self.pass2b_template = (
            pass2b_path.read_text()
            if pass2b_path.exists()
            else ""
        )

        # Load Pass 2c prompt (reconciliation)
        pass2c_path = prompts_dir / "video_pass2c_reconciliation.txt"
        self.pass2c_template = (
            pass2c_path.read_text()
            if pass2c_path.exists()
            else ""
        )

        # Load Pass 2d prompt (step completion verification)
        pass2d_path = prompts_dir / "video_pass2d_step_completion.txt"
        self.pass2d_template = (
            pass2d_path.read_text()
            if pass2d_path.exists()
            else ""
        )

        atomic_substeps_path = prompts_dir / "video_atomic_substeps.txt"
        self.atomic_substeps_template = (
            atomic_substeps_path.read_text()
            if atomic_substeps_path.exists()
            else self._get_default_atomic_substeps_prompt()
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
        Run the full 3-pass pipeline and return the video_enhanced.json structure.

        Pipeline:
        1. Extract frames from video (denser sampling: every 5 frames).
        2. VLM Pass 1: Classify frames with quality metrics → placement candidates.
        3. VLM Pass 2: Validate placements with context from enhanced.json + detect duplicates.
           - Pass 2a: Action analysis (what action is being performed?)
           - Pass 2b: SAM3 comparison (what changed in segmented images?)
           - Pass 2c: Reconciliation (combine 2a + 2b + expected parts)
           - Pass 2d: Step completion verification (is step complete?)
        4. VLM Pass 3: Generate atomic sub-steps (1 part per sub-step) with spatial positioning.

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

        # === VLM PASS 3: Atomic Sub-step Generation ===
        # Note: Reconciliation is already done in Pass 2c during validation
        logger.info("VLM Pass 3: Generating atomic sub-steps from validated placements...")
        enhanced_manual = await self._generate_atomic_substeps(
            manual_id, video_id, validated_placements_data, manual_data, frames
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
        VLM Pass 1: Two-stage frame classification.

        Pass 1a: Classify frames (action vs placement_candidate) using temporal context
        Pass 1b: Filter placement candidates for quality (assembly visibility)

        Uses batched VLM calls (8 frames per call) to reduce API costs.
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

        # Initialize Pass 1 reasoning log file
        pass1_log_dir = self.settings.data_dir / "processed" / manual_id / f"vlm_reasoning_logs_{video_id}"
        pass1_log_dir.mkdir(parents=True, exist_ok=True)
        pass1_log_path = pass1_log_dir / "pass1_classification_reasoning.log"

        # Clear/create new log file at the start
        with open(pass1_log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=" * 100 + "\n")
            log_file.write("VLM PASS 1 CLASSIFICATION REASONING LOG\n")
            log_file.write(f"Manual ID: {manual_id}\n")
            log_file.write(f"Video ID: {video_id}\n")
            log_file.write(f"Total frames to classify: {len(frames)}\n")
            log_file.write("=" * 100 + "\n\n")

        logger.info(f"  Pass 1 reasoning log will be written to: {pass1_log_path}")

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

            # === VLM PASS 1a: Frame Classification (action vs placement) ===
            logger.info(f"  VLM Pass 1a: Classifying {len(uncached_frames)} frames using temporal context...")
            pass1a_placement_candidates = []

            for batch_idx, batch in enumerate(batches):
                try:
                    results = self._pass1a_classify_batch(batch)

                    # Count placement candidates in this batch
                    batch_placements = []

                    # Process results
                    for (frame_num, frame_path), result in zip(batch, results):
                        is_relevant = result.get("is_relevant", True)
                        frame_type = result.get("frame_type") if is_relevant else None
                        is_stable = result.get("is_stable", True)
                        confidence = result.get("confidence", 0.0)

                        # Store Pass 1a result
                        if is_relevant and frame_type == "placement_candidate":
                            pass1a_placement_candidates.append((frame_num, frame_path, result))
                            batch_placements.append(frame_num)
                        elif is_relevant:
                            # Action frames go directly to final list
                            timestamp = frame_num / 30.0
                            newly_classified.append({
                                "frame_number": frame_num,
                                "frame_path": frame_path,
                                "timestamp": timestamp,
                                "frame_type": frame_type,
                                "quality_score": 0.5,  # Default for action frames
                                "confidence": confidence
                            })

                    logger.info(
                        f"    Pass 1a - Batch {batch_idx + 1}/{len(batches)}: "
                        f"Found {len(batch_placements)} placement candidates: {batch_placements}"
                    )

                except Exception as e:
                    logger.error(f"Failed to classify batch {batch_idx + 1} in Pass 1a: {e}")
                    # Fallback: treat all frames as action
                    for frame_num, frame_path in batch:
                        timestamp = frame_num / 30.0
                        newly_classified.append({
                            "frame_number": frame_num,
                            "frame_path": frame_path,
                            "timestamp": timestamp,
                            "frame_type": "action",
                            "quality_score": 0.0,
                            "confidence": 0.0
                        })

            # Log Pass 1a results
            with open(pass1_log_path, "a", encoding="utf-8") as log_file:
                log_file.write("=" * 100 + "\n")
                log_file.write("VLM PASS 1a - FRAME CLASSIFICATION (action vs placement_candidate)\n")
                log_file.write("=" * 100 + "\n\n")
                log_file.write(f"Total frames analyzed: {len(uncached_frames)}\n")
                log_file.write(f"Placement candidates found: {len(pass1a_placement_candidates)}\n")
                log_file.write(f"Action frames found: {len([f for f in newly_classified if f.get('frame_type') == 'action'])}\n\n")

                if pass1a_placement_candidates:
                    log_file.write("PLACEMENT CANDIDATES:\n")
                    for frame_num, frame_path, result in pass1a_placement_candidates:
                        log_file.write(f"\n  Frame {frame_num} (timestamp: {frame_num / 30.0:.2f}s):\n")
                        log_file.write(f"    Path: {frame_path}\n")
                        log_file.write(f"    frame_type: {result.get('frame_type', 'N/A')}\n")
                        log_file.write(f"    is_stable: {result.get('is_stable', 'N/A')}\n")
                        log_file.write(f"    confidence: {result.get('confidence', 0.0):.2f}\n")
                        log_file.write(f"    reasoning: {result.get('reasoning', 'N/A')}\n")
                log_file.write("\n")

            # === VLM PASS 1b: Quality Filtering (assembly visibility) ===
            if pass1a_placement_candidates:
                logger.info(
                    f"  VLM Pass 1b: Filtering {len(pass1a_placement_candidates)} placement candidates "
                    f"for assembly visibility..."
                )

                # Create batches for Pass 1b
                pass1b_batches = self._create_batches(
                    [(num, path) for num, path, _ in pass1a_placement_candidates],
                    batch_size
                )

                for batch_idx, batch in enumerate(pass1b_batches):
                    try:
                        filter_results = self._pass1b_filter_batch(batch)

                        # Count accepted frames
                        batch_accepted = []
                        batch_rejected = []

                        # Process filter results
                        for (frame_num, frame_path), filter_result in zip(batch, filter_results):
                            timestamp = frame_num / 30.0
                            accept = filter_result.get("accept", False)
                            quality_score = filter_result.get("quality_score", 0.0)
                            has_hand_obstruction = filter_result.get("has_hand_obstruction", False)
                            confidence = filter_result.get("confidence", 0.0)

                            # Update cache
                            cache[str(frame_num)] = {
                                "is_relevant": True,
                                "frame_type": "placement_candidate" if accept else "action",
                                "quality_score": quality_score,
                                "has_hand_obstruction": has_hand_obstruction,
                                "is_stable": True,
                                "confidence": confidence,
                                "pass1b_accepted": accept
                            }

                            if accept:
                                newly_classified.append({
                                    "frame_number": frame_num,
                                    "frame_path": frame_path,
                                    "timestamp": timestamp,
                                    "frame_type": "placement_candidate",
                                    "quality_score": quality_score,
                                    "confidence": confidence
                                })
                                batch_accepted.append(frame_num)
                            else:
                                batch_rejected.append(frame_num)

                        logger.info(
                            f"    Pass 1b - Batch {batch_idx + 1}/{len(pass1b_batches)}: "
                            f"Accepted {len(batch_accepted)} frames {batch_accepted}, "
                            f"Rejected {len(batch_rejected)} frames {batch_rejected}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to filter batch {batch_idx + 1} in Pass 1b: {e}")
                        # Fallback: accept all frames in batch
                        for frame_num, frame_path in batch:
                            timestamp = frame_num / 30.0
                            newly_classified.append({
                                "frame_number": frame_num,
                                "frame_path": frame_path,
                                "timestamp": timestamp,
                                "frame_type": "placement_candidate",
                                "quality_score": 0.5,
                                "confidence": 0.5
                            })

                # Write cache after Pass 1b
                cache_path.write_text(json.dumps(cache, indent=2))

                # Log Pass 1b results
                with open(pass1_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("=" * 100 + "\n")
                    log_file.write("VLM PASS 1b - QUALITY FILTERING (assembly visibility)\n")
                    log_file.write("=" * 100 + "\n\n")
                    log_file.write(f"Total placement candidates from Pass 1a: {len(pass1a_placement_candidates)}\n")

                    # Count accepted and rejected
                    accepted_frames = [f for f in newly_classified if f.get('frame_type') == 'placement_candidate']
                    rejected_count = len(pass1a_placement_candidates) - len(accepted_frames)
                    log_file.write(f"ACCEPTED frames: {len(accepted_frames)}\n")
                    log_file.write(f"REJECTED frames: {rejected_count}\n\n")

                    # Log each frame's Pass 1b result
                    for frame_num, frame_path, pass1a_result in pass1a_placement_candidates:
                        cache_entry = cache.get(str(frame_num), {})
                        accepted = cache_entry.get("pass1b_accepted", False)
                        log_file.write(f"\n  Frame {frame_num} (timestamp: {frame_num / 30.0:.2f}s):\n")
                        log_file.write(f"    Path: {frame_path}\n")
                        log_file.write(f"    VERDICT: {'ACCEPTED' if accepted else 'REJECTED'}\n")
                        log_file.write(f"    quality_score: {cache_entry.get('quality_score', 0.0):.2f}\n")
                        log_file.write(f"    has_hand_obstruction: {cache_entry.get('has_hand_obstruction', False)}\n")
                        log_file.write(f"    confidence: {cache_entry.get('confidence', 0.0):.2f}\n")
                    log_file.write("\n")
            else:
                logger.info("  VLM Pass 1b: No placement candidates to filter")
                # Log empty Pass 1b
                with open(pass1_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("=" * 100 + "\n")
                    log_file.write("VLM PASS 1b - QUALITY FILTERING (assembly visibility)\n")
                    log_file.write("=" * 100 + "\n\n")
                    log_file.write("No placement candidates to filter from Pass 1a\n\n")

        # Merge cached + newly classified
        all_classified = cached_frames + newly_classified

        # Sort by frame_number to maintain temporal order
        all_classified.sort(key=lambda x: x["frame_number"])

        placement_count = sum(1 for f in all_classified if f["frame_type"] == "placement_candidate")
        action_count = len(all_classified) - placement_count
        irrelevant_count = len(frames) - len(all_classified)

        logger.info(
            f"Pass 1 summary: {len(all_classified)} relevant, "
            f"{placement_count} placement candidates, "
            f"{action_count} action, "
            f"{irrelevant_count} irrelevant"
        )

        # Write final summary to log
        with open(pass1_log_path, "a", encoding="utf-8") as log_file:
            log_file.write("=" * 100 + "\n")
            log_file.write("PASS 1 FINAL SUMMARY\n")
            log_file.write("=" * 100 + "\n\n")
            log_file.write(f"Total frames processed: {len(frames)}\n")
            log_file.write(f"Frames from cache: {len(cached_frames)}\n")
            log_file.write(f"Newly classified frames: {len(newly_classified)}\n\n")
            log_file.write(f"CLASSIFICATION RESULTS:\n")
            log_file.write(f"  Placement candidates: {placement_count}\n")
            log_file.write(f"  Action frames: {action_count}\n")
            log_file.write(f"  Irrelevant frames: {irrelevant_count}\n\n")

            # List all placement candidates
            if placement_count > 0:
                log_file.write("FINAL PLACEMENT CANDIDATES:\n")
                placement_frames = [f for f in all_classified if f["frame_type"] == "placement_candidate"]
                for f in placement_frames:
                    log_file.write(f"  Frame {f['frame_number']} (timestamp: {f['timestamp']:.2f}s, quality: {f.get('quality_score', 0.0):.2f}, confidence: {f.get('confidence', 0.0):.2f})\n")
            log_file.write("\n")

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

    def _pass1a_classify_batch(
        self,
        batch: List[tuple[int, Path]]
    ) -> List[Dict[str, Any]]:
        """
        VLM Pass 1a: Classify a batch of frames (action vs placement) using temporal context.

        Returns list of classification results in same order as batch.
        """
        # Build multi-image content payload
        content = []

        # Add frames in chronological order
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

        # Add Pass 1a prompt at end
        content.append({"type": "text", "text": self.pass1a_template})

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

    def _pass1b_filter_batch(
        self,
        batch: List[tuple[int, Path]]
    ) -> List[Dict[str, Any]]:
        """
        VLM Pass 1b: Filter placement candidates for quality (assembly visibility).

        Returns list of filter results in same order as batch.
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

        # Add Pass 1b prompt at end
        content.append({"type": "text", "text": self.pass1b_template})

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
                f"Pass 1b batch size mismatch: sent {len(batch)} frames, "
                f"got {len(result)} results. Attempting to recover..."
            )
            # Pad or truncate to match
            while len(result) < len(batch):
                result.append({
                    "accept": True,  # Default to accepting
                    "quality_score": 0.5,
                    "has_hand_obstruction": False,
                    "confidence": 0.5,
                    "reasoning": "Fallback due to incomplete batch response"
                })
            if len(result) > len(batch):
                result = result[:len(batch)]

        # Ensure required fields
        for idx, entry in enumerate(result):
            if not isinstance(entry, dict):
                result[idx] = {
                    "accept": True,
                    "quality_score": 0.5,
                    "has_hand_obstruction": False,
                    "confidence": 0.5
                }
            else:
                entry.setdefault("accept", True)
                entry.setdefault("quality_score", 0.5)
                entry.setdefault("has_hand_obstruction", False)
                entry.setdefault("confidence", 0.5)

        return result

    # ── VLM Pass 2a: Action Frame Analysis (DEPRECATED - now unified with Pass 2b) ──────────

    def _analyze_action_frames(
        self,
        action_frames: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: VLM Pass 2a: Analyze action frames to identify what part is being manipulated.

        This method is deprecated and replaced by _unified_placement_analysis which combines
        Pass 2a and 2b into a single multimodal call.

        Kept for backwards compatibility only.
        """
        logger.warning("_analyze_action_frames is deprecated. Use _unified_placement_analysis instead.")
        return None

    # ── VLM Pass 2a: Action Sequence Analysis (No SAM3) ──────────

    def _pass2a_action_analysis(
        self,
        prev_placement_frame: Optional[Dict[str, Any]],
        action_frames: List[Dict[str, Any]],
        current_placement_frame: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Pass 2a: Analyze action sequence to understand what action is being performed.

        Args:
            prev_placement_frame: Previous accepted placement frame dict (or None for first placement)
            action_frames: ALL action frames between prev and current placement
            current_placement_frame: Current placement candidate frame dict

        Returns:
            {
                "action_type": str,
                "has_new_part": bool,
                "part_from_actions": {description, color, type, approximate_size, confidence},
                "action_narrative": str,
                "confidence": float,
                "reasoning": str
            }
            or None if analysis fails
        """
        try:
            content = []

            # === PREVIOUS PLACEMENT FRAME ===
            if prev_placement_frame:
                prev_frame_path_str = prev_placement_frame.get("frame_path")
                prev_frame_num = prev_placement_frame.get("frame_number")

                # Resolve the full path
                prev_frame_path = Path(prev_frame_path_str)
                if not prev_frame_path.is_absolute():
                    if prev_frame_path.parts[0] == 'data':
                        prev_frame_path = self.settings.data_dir / Path(*prev_frame_path.parts[1:])
                    else:
                        prev_frame_path = self.settings.data_dir / prev_frame_path

                prev_img = _resize_image(str(prev_frame_path), width=800)
                prev_b64 = _image_to_b64(prev_img)
                content.append({
                    "type": "text",
                    "text": f"PREVIOUS PLACEMENT FRAME (frame {prev_frame_num}):"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"}
                })
            else:
                content.append({
                    "type": "text",
                    "text": "PREVIOUS PLACEMENT FRAME: None (this is the first placement)"
                })

            # === ALL ACTION FRAMES (NO SAMPLING) ===
            if action_frames:
                content.append({
                    "type": "text",
                    "text": f"ACTION FRAMES ({len(action_frames)} frames showing hands manipulating parts):"
                })
                for idx, action_frame in enumerate(action_frames):
                    action_frame_path_str = action_frame.get("frame_path")
                    action_frame_num = action_frame.get("frame_number")

                    # Resolve the full path
                    action_frame_path = Path(action_frame_path_str)
                    if not action_frame_path.is_absolute():
                        if action_frame_path.parts[0] == 'data':
                            action_frame_path = self.settings.data_dir / Path(*action_frame_path.parts[1:])
                        else:
                            action_frame_path = self.settings.data_dir / action_frame_path

                    action_img = _resize_image(str(action_frame_path), width=600)
                    action_b64 = _image_to_b64(action_img)
                    content.append({
                        "type": "text",
                        "text": f"Action Frame {idx+1}/{len(action_frames)} (frame {action_frame_num}):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{action_b64}"}
                    })

            # === CURRENT PLACEMENT FRAME ===
            current_frame_path_str = current_placement_frame.get("frame_path")
            current_frame_num = current_placement_frame.get("frame_number")

            # Resolve the full path
            current_frame_path = Path(current_frame_path_str)
            if not current_frame_path.is_absolute():
                if current_frame_path.parts[0] == 'data':
                    current_frame_path = self.settings.data_dir / Path(*current_frame_path.parts[1:])
                else:
                    current_frame_path = self.settings.data_dir / current_frame_path

            current_img = _resize_image(str(current_frame_path), width=800)
            current_b64 = _image_to_b64(current_img)
            content.append({
                "type": "text",
                "text": f"CURRENT PLACEMENT FRAME (frame {current_frame_num}):"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"}
            })

            # === PROMPT ===
            content.append({
                "type": "text",
                "text": self.pass2a_template
            })

            # Make VLM call
            messages = [{"role": "user", "content": content}]
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            return result

        except Exception as e:
            logger.error(f"Pass 2a action analysis failed: {e}")
            return None

    # ── VLM Pass 2b: SAM3 Comparison ──────────

    def _pass2b_sam3_comparison(
        self,
        prev_placement_frame: Optional[Dict[str, Any]],
        current_placement_frame: Dict[str, Any],
        manual_id: str,
        video_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Pass 2b: Compare SAM3-segmented images (prev vs current) to identify what changed.

        Args:
            prev_placement_frame: Previous accepted placement frame dict (or None for first placement)
            current_placement_frame: Current placement candidate frame dict
            manual_id: Manual identifier
            video_id: Video identifier

        Returns:
            {
                "is_duplicate": bool,
                "has_new_part": bool,
                "what_changed": str,
                "new_part_detected": {description, color, type, size, stud_count},
                "spatial_position": {location, reference_object, orientation},
                "box_2d": [ymin, xmin, ymax, xmax],
                "confidence": float,
                "reasoning": str,
                "sam3_prev_path": str,
                "sam3_current_path": str
            }
            or None if analysis fails
        """
        try:
            content = []
            sam3_prev_used = None
            sam3_current_used = None

            # === PREVIOUS PLACEMENT FRAME ===
            if prev_placement_frame is None:
                # First placement - explicitly tell the VLM there's no previous frame
                content.append({
                    "type": "text",
                    "text": "PREVIOUS PLACEMENT: None (this is the FIRST placement)"
                })
            elif prev_placement_frame:
                prev_frame_path_str = prev_placement_frame.get("frame_path")
                prev_frame_num = prev_placement_frame.get("frame_number")

                # Resolve the full path - handle both absolute and relative paths
                prev_frame_path = Path(prev_frame_path_str)
                if not prev_frame_path.is_absolute():
                    # If relative, it might be relative to data_dir or project root
                    if prev_frame_path.parts[0] == 'data':
                        # Relative from project root, remove 'data' prefix
                        prev_frame_path = self.settings.data_dir / Path(*prev_frame_path.parts[1:])
                    else:
                        # Relative from data_dir
                        prev_frame_path = self.settings.data_dir / prev_frame_path

                # Apply SAM3 segmentation to previous frame (always use SAM3 for Pass 2b)
                if self.settings.roboflow_api_key:
                    sam3_dir = self.settings.data_dir / "processed" / manual_id / f"sam3_segmented_{video_id}"
                    sam3_prev_path = sam3_dir / f"prev_frame_{prev_frame_num}_sam3.jpg"

                    segmented_path = _apply_sam3_segmentation(
                        image_path=str(prev_frame_path),
                        output_path=sam3_prev_path,
                        api_key=self.settings.roboflow_api_key,
                        text_prompt="lego assembly"
                    )

                    if segmented_path:
                        # Use SAM3 segmented version
                        sam3_prev_used = str(segmented_path.relative_to(self.settings.data_dir))
                        prev_img = _resize_image(str(segmented_path), width=800)
                        prev_b64 = _image_to_b64(prev_img)
                        content.append({
                            "type": "text",
                            "text": f"PREVIOUS PLACEMENT (SAM3 segmented, frame {prev_frame_num}):"
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"}
                        })
                    else:
                        logger.warning(f"SAM3 failed for prev frame {prev_frame_num} - cannot run Pass 2b without SAM3")
                        return None
                else:
                    logger.warning("Pass 2b requires SAM3 segmentation but ROBOFLOW_API_KEY not set")
                    return None

            # === CURRENT PLACEMENT FRAME ===
            current_frame_path_str = current_placement_frame.get("frame_path")
            current_frame_num = current_placement_frame.get("frame_number")

            # Resolve the full path
            current_frame_path = Path(current_frame_path_str)
            if not current_frame_path.is_absolute():
                if current_frame_path.parts[0] == 'data':
                    current_frame_path = self.settings.data_dir / Path(*current_frame_path.parts[1:])
                else:
                    current_frame_path = self.settings.data_dir / current_frame_path

            # Apply SAM3 segmentation to current frame (always use SAM3 for Pass 2b)
            if self.settings.roboflow_api_key:
                sam3_dir = self.settings.data_dir / "processed" / manual_id / f"sam3_segmented_{video_id}"
                sam3_current_path = sam3_dir / f"current_frame_{current_frame_num}_sam3.jpg"

                segmented_path = _apply_sam3_segmentation(
                    image_path=str(current_frame_path),
                    output_path=sam3_current_path,
                    api_key=self.settings.roboflow_api_key,
                    text_prompt="lego assembly"
                )

                if segmented_path:
                    # Use SAM3 segmented version
                    sam3_current_used = str(segmented_path.relative_to(self.settings.data_dir))
                    current_img = _resize_image(str(segmented_path), width=800)
                    current_b64 = _image_to_b64(current_img)
                    content.append({
                        "type": "text",
                        "text": f"CURRENT PLACEMENT (SAM3 segmented, frame {current_frame_num}):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"}
                    })
                else:
                    logger.warning(f"SAM3 failed for current frame {current_frame_num} - cannot run Pass 2b without SAM3")
                    return None
            else:
                logger.warning("Pass 2b requires SAM3 segmentation but ROBOFLOW_API_KEY not set")
                return None

            # === PROMPT ===
            content.append({
                "type": "text",
                "text": self.pass2b_template
            })

            # Make VLM call
            messages = [{"role": "user", "content": content}]
            raw = self.vlm._litellm_with_retry(messages)
            result = _parse_json(raw)

            # Add SAM3 paths to result for logging
            if result:
                result["sam3_prev_path"] = sam3_prev_used
                result["sam3_current_path"] = sam3_current_used

                # Draw bounding box on SAM3 segmented frame if box_2d is provided
                box_2d = result.get("box_2d")
                if box_2d and sam3_current_used and result.get("has_new_part"):
                    # Get the full SAM3 current path
                    sam3_current_full_path = self.settings.data_dir / sam3_current_used

                    # Get part description for label
                    new_part = result.get("new_part_detected", {})
                    part_desc = new_part.get("description", "new part") if isinstance(new_part, dict) else "new part"
                    confidence = result.get("confidence", 0.0)

                    # Draw annotated SAM3 frame
                    annotated_sam3_dir = self.settings.data_dir / "processed" / manual_id / f"sam3_annotated_{video_id}"
                    annotated_sam3_path = _draw_placement_bbox(
                        frame_path=sam3_current_full_path,
                        box_2d=box_2d,
                        frame_num=current_frame_num,
                        label=f"Pass 2b: {part_desc}",
                        confidence=confidence,
                        output_dir=annotated_sam3_dir
                    )

                    # Add annotated SAM3 path to result
                    if annotated_sam3_path:
                        result["sam3_annotated_path"] = str(annotated_sam3_path.relative_to(self.settings.data_dir))

            return result

        except Exception as e:
            logger.error(f"Pass 2b SAM3 comparison failed: {e}")
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
                # === Find action frames between placements ===
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

                # === VLM PASS 2a: Action Analysis ===
                logger.debug(f"  Frame {frame_num}: Pass 2a - Analyzing action sequence ({len(action_frames_between)} action frames)")

                pass2a_result = self._pass2a_action_analysis(
                    prev_placement_frame=last_accepted_candidate,
                    action_frames=action_frames_between,
                    current_placement_frame=current_frame
                )

                if not pass2a_result:
                    logger.warning(f"  Placement {i} (frame {frame_num}): Pass 2a failed, skipping")
                    continue

                # === VLM PASS 2b: SAM3 Comparison ===
                logger.debug(f"  Frame {frame_num}: Pass 2b - Comparing SAM3 segmented images")

                pass2b_result = self._pass2b_sam3_comparison(
                    prev_placement_frame=last_accepted_candidate,
                    current_placement_frame=current_frame,
                    manual_id=manual_id,
                    video_id=video_id
                )

                if not pass2b_result:
                    logger.warning(f"  Placement {i} (frame {frame_num}): Pass 2b failed (SAM3 required), skipping")
                    continue

                # Check if either pass detected a duplicate
                pass2a_has_new = pass2a_result.get("has_new_part", False)
                pass2b_has_new = pass2b_result.get("has_new_part", False)
                pass2b_is_dup = pass2b_result.get("is_duplicate", False)

                # If Pass 2b (SAM3 comparison) says duplicate, trust it
                if pass2b_is_dup or not pass2b_has_new:
                    is_duplicate = True
                    has_new_part = False
                else:
                    is_duplicate = False
                    has_new_part = pass2b_has_new

                # Use Pass 2b confidence for filtering (SAM3 is more reliable)
                confidence = pass2b_result.get("confidence", 0.0)

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

                    # Action frames info
                    log_file.write(f"ACTION FRAMES BETWEEN PLACEMENTS: {len(action_frames_between)} frames\n")
                    if action_frames_between:
                        log_file.write(f"  Frame numbers: {[f.get('frame_number') for f in action_frames_between]}\n")
                    log_file.write("\n")

                    # Current frame info
                    log_file.write(f"CURRENT FRAME: Frame {frame_num} (timestamp: {timestamp:.2f}s)\n")
                    log_file.write(f"  Path: {current_frame.get('frame_path', 'N/A')}\n\n")

                    # VLM Pass 2a (Action Analysis)
                    log_file.write("VLM PASS 2a - ACTION ANALYSIS:\n")
                    log_file.write(f"  Analyzed {len(action_frames_between)} action frames between prev and current\n")
                    log_file.write(f"  VLM Output:\n")
                    log_file.write(f"    action_type: {pass2a_result.get('action_type', 'N/A')}\n")
                    log_file.write(f"    has_new_part: {pass2a_result.get('has_new_part', False)}\n")
                    log_file.write(f"    part_from_actions: {pass2a_result.get('part_from_actions', 'N/A')}\n")
                    log_file.write(f"    box_2d (for original frame): {pass2a_result.get('box_2d', 'N/A')}\n")
                    log_file.write(f"    action_narrative: {pass2a_result.get('action_narrative', 'N/A')}\n")
                    log_file.write(f"    confidence: {pass2a_result.get('confidence', 0.0):.2f}\n")
                    log_file.write(f"    reasoning: {pass2a_result.get('reasoning', 'N/A')}\n\n")

                    # VLM Pass 2b (SAM3 Comparison)
                    log_file.write("VLM PASS 2b - SAM3 COMPARISON:\n")
                    sam3_prev = pass2b_result.get('sam3_prev_path', None)
                    sam3_current = pass2b_result.get('sam3_current_path', None)
                    sam3_annotated = pass2b_result.get('sam3_annotated_path', None)
                    log_file.write(f"  Comparing SAM3 segmented images:\n")
                    if sam3_prev:
                        log_file.write(f"    Previous (SAM3): {sam3_prev}\n")
                    else:
                        log_file.write(f"    Previous: SAM3 failed\n")
                    if sam3_current:
                        log_file.write(f"    Current (SAM3): {sam3_current}\n")
                    else:
                        log_file.write(f"    Current: SAM3 failed\n")
                    if sam3_annotated:
                        log_file.write(f"    Annotated (SAM3 with bbox): {sam3_annotated}\n")

                    # Log comparison analysis (new structured field)
                    comparison_analysis = pass2b_result.get('comparison_analysis', {})
                    if comparison_analysis:
                        log_file.write(f"\n  Systematic Comparison Analysis:\n")
                        log_file.write(f"    Grid Scan: {comparison_analysis.get('grid_scan_summary', 'N/A')}\n")
                        log_file.write(f"    Previous Region: {comparison_analysis.get('previous_region_description', 'N/A')}\n")
                        log_file.write(f"    Current Region: {comparison_analysis.get('current_region_description', 'N/A')}\n")
                        log_file.write(f"    Identified Difference: {comparison_analysis.get('identified_difference', 'N/A')}\n")

                    log_file.write(f"\n  VLM Output:\n")
                    log_file.write(f"    is_duplicate: {pass2b_result.get('is_duplicate', False)}\n")
                    log_file.write(f"    has_new_part: {pass2b_result.get('has_new_part', False)}\n")
                    log_file.write(f"    what_changed: {pass2b_result.get('what_changed', 'N/A')}\n")
                    log_file.write(f"    new_part_detected: {pass2b_result.get('new_part_detected', 'N/A')}\n")
                    log_file.write(f"    spatial_position: {pass2b_result.get('spatial_position', {})}\n")
                    log_file.write(f"    box_2d (for SAM3 frame): {pass2b_result.get('box_2d', 'N/A')}\n")
                    log_file.write(f"    confidence: {pass2b_result.get('confidence', 0.0):.2f}\n")
                    log_file.write(f"    reasoning: {pass2b_result.get('reasoning', 'N/A')}\n\n")

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

                # Use Pass 2b's detected part info for now (will be reconciled by Pass 2c)
                new_part_detected = pass2b_result.get("new_part_detected", {})
                spatial_position = pass2b_result.get("spatial_position", {})
                box_2d_sam3 = pass2b_result.get("box_2d")  # Box for SAM3 images

                # Get bounding box from Pass 2a for original frame annotation
                box_2d_original = pass2a_result.get("box_2d")  # Box for original frames

                # Create preliminary action description from Pass 2b
                if isinstance(new_part_detected, dict):
                    part_desc = new_part_detected.get("description", "unknown part")
                    action_desc = f"Add {part_desc}"
                else:
                    action_desc = "Add part"

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

                # Draw bounding box on original frame using Pass 2a's box_2d
                # Pass 2a analyzes the original frames with hands, so its box coordinates are correct for original frames
                # Pass 2b's box_2d is for SAM3 images (already annotated at line 1083)
                annotated_frame_path = None
                if box_2d_original:
                    annotated_frame_path = _draw_placement_bbox(
                        frame_path=current_frame["frame_path"],
                        box_2d=box_2d_original,
                        frame_num=frame_num,
                        label=action_desc,
                        confidence=confidence,
                        output_dir=self.settings.data_dir / "processed" / manual_id / f"validated_placement_annotated_{video_id}"
                    )
                else:
                    logger.debug(f"  Frame {frame_num}: Pass 2a did not provide box_2d for original frame")

                # Prepare placement metadata for Pass 2c
                # Pass 2c needs both box coordinates for correct cropping
                placement_metadata: Dict[str, Any] = {
                    "placement_index": i,
                    "frame_number": frame_num,
                    "timestamp": timestamp,
                    "frame_path": relative_frame_path,
                    "box_2d_original": box_2d_original,  # Box for cropping original frame (from Pass 2a)
                    "box_2d_sam3": box_2d_sam3,  # Box for cropping SAM3 frame (from Pass 2b)
                    "annotated_frame_path": str(annotated_frame_path.relative_to(self.settings.data_dir)) if annotated_frame_path else None,
                }

                # === VLM PASS 2c: IMMEDIATE RECONCILIATION ===
                # Reconcile Pass 2a + Pass 2b + expected parts to determine final part identity
                logger.debug(f"  Placement {i}: Pass 2c - Reconciling all sources...")
                step_data = parts_by_step.get(current_step, {})
                expected_parts_data = step_data.get("parts", [])

                try:
                    reconciliation = await self._pass2c_reconcile_all_sources(
                        pass2a_result=pass2a_result,
                        pass2b_result=pass2b_result,
                        placement=placement_metadata,
                        expected_parts=expected_parts_data,
                        step_number=current_step,
                        manual_id=manual_id
                    )

                    # Extract final part from Pass 2c
                    final_part = reconciliation.get("final_part", {})
                    if isinstance(final_part, dict):
                        final_desc = final_part.get("description", "unknown part")
                        final_action_desc = f"Add {final_desc}"
                    else:
                        final_action_desc = action_desc  # Fallback to preliminary

                    # Build final action_data with reconciled information
                    action_data: Dict[str, Any] = {
                        "placement_index": i,
                        "frame_number": frame_num,
                        "timestamp": timestamp,
                        "frame_path": relative_frame_path,
                        "action_description": final_action_desc,  # Use Pass 2c's final result
                        "new_parts": [final_part] if final_part else [],
                        "spatial_position": spatial_position,
                        "box_2d": box_2d_original,  # Bounding box for original frame
                        "box_2d_sam3": box_2d_sam3,  # Bounding box for SAM3 frame (for reference)
                        "annotated_frame_path": str(annotated_frame_path.relative_to(self.settings.data_dir)) if annotated_frame_path else None,
                        "is_subassembly_switch": is_subassembly_switch,
                        "current_subassembly": self.subassembly_tracker.get_current_subassembly(),
                        "manual_step": current_step,
                        "confidence": confidence,
                        "reconciliation": {
                            "pass2a_result": pass2a_result,
                            "pass2b_result": pass2b_result,
                            "pass2c_result": reconciliation,
                            "sources_agree": reconciliation.get("sources_agree", False),
                            "matched_part": reconciliation.get("matched_part"),
                            "video_detection_correct": reconciliation.get("video_detection_correct", True),
                            "correction": reconciliation.get("correction"),
                            "reasoning": reconciliation.get("reasoning", "")
                        }
                    }

                    # Add to validated placements
                    validated_placements.append(action_data)
                    last_accepted_candidate = current_frame

                    # Log Pass 2c reconciliation results
                    with open(comparison_log_path, "a", encoding="utf-8") as log_file:
                        log_file.write("VLM PASS 2c - RECONCILIATION (Pass 2a + Pass 2b + Expected Parts):\n")
                        log_file.write(f"  Inputs:\n")
                        log_file.write(f"\n    Pass 2a Result (Action Analysis):\n")
                        log_file.write(f"      action_type: {pass2a_result.get('action_type', 'N/A')}\n")
                        log_file.write(f"      has_new_part: {pass2a_result.get('has_new_part', 'N/A')}\n")
                        pass2a_part = pass2a_result.get('part_from_actions', {})
                        if pass2a_part:
                            log_file.write(f"      part_from_actions:\n")
                            log_file.write(f"        description: {pass2a_part.get('description', 'N/A')}\n")
                            log_file.write(f"        color: {pass2a_part.get('color', 'N/A')}\n")
                            log_file.write(f"        type: {pass2a_part.get('type', 'N/A')}\n")
                            log_file.write(f"        approximate_size: {pass2a_part.get('approximate_size', 'N/A')}\n")
                            log_file.write(f"        confidence: {pass2a_part.get('confidence', 'N/A')}\n")
                        else:
                            log_file.write(f"      part_from_actions: None\n")
                        log_file.write(f"      action_narrative: {pass2a_result.get('action_narrative', 'N/A')}\n")

                        log_file.write(f"\n    Pass 2b Result (SAM3 Comparison):\n")
                        log_file.write(f"      is_duplicate: {pass2b_result.get('is_duplicate', 'N/A')}\n")
                        log_file.write(f"      has_new_part: {pass2b_result.get('has_new_part', 'N/A')}\n")
                        pass2b_part = pass2b_result.get('new_part_detected', {})
                        if pass2b_part:
                            log_file.write(f"      new_part_detected:\n")
                            log_file.write(f"        description: {pass2b_part.get('description', 'N/A')}\n")
                            log_file.write(f"        color: {pass2b_part.get('color', 'N/A')}\n")
                            log_file.write(f"        type: {pass2b_part.get('type', 'N/A')}\n")
                            log_file.write(f"        size: {pass2b_part.get('size', 'N/A')}\n")
                            log_file.write(f"        stud_count: {pass2b_part.get('stud_count', 'N/A')}\n")
                        else:
                            log_file.write(f"      new_part_detected: None\n")
                        pass2b_spatial = pass2b_result.get('spatial_position', {})
                        if pass2b_spatial:
                            log_file.write(f"      spatial_position:\n")
                            log_file.write(f"        location: {pass2b_spatial.get('location', 'N/A')}\n")
                            log_file.write(f"        reference_object: {pass2b_spatial.get('reference_object', 'N/A')}\n")
                        log_file.write(f"      box_2d: {pass2b_result.get('box_2d', 'N/A')}\n")
                        # Show annotated SAM3 path if available
                        sam3_annotated = pass2b_result.get('sam3_annotated_path')
                        if sam3_annotated:
                            log_file.write(f"      sam3_annotated: {sam3_annotated}\n")

                        log_file.write(f"\n    Expected Parts for Step {current_step}:\n")
                        log_file.write(f"      Total parts: {len(expected_parts_data)}\n")
                        if expected_parts_data:
                            for idx, part in enumerate(expected_parts_data[:3], 1):  # Show first 3 parts
                                log_file.write(f"      Part {idx}: {part.get('description', 'N/A')} (color: {part.get('color', 'N/A')}, type: {part.get('type', 'N/A')}, size: {part.get('size', 'N/A')})\n")
                            if len(expected_parts_data) > 3:
                                log_file.write(f"      ... and {len(expected_parts_data) - 3} more parts\n")

                        log_file.write(f"\n  VLM Output:\n")
                        log_file.write(f"    sources_agree: {reconciliation.get('sources_agree', False)}\n")
                        log_file.write(f"    pass2a_correct: {reconciliation.get('pass2a_correct', 'N/A')}\n")
                        log_file.write(f"    pass2b_correct: {reconciliation.get('pass2b_correct', 'N/A')}\n")
                        log_file.write(f"    final_part: {final_part.get('description', 'N/A')}\n")
                        log_file.write(f"    stud_count_analysis: {reconciliation.get('stud_count_analysis', {})}\n")
                        log_file.write(f"    corrections: {reconciliation.get('corrections', {})}\n")
                        log_file.write(f"    reasoning: {reconciliation.get('reasoning', 'N/A')}\n\n")

                    logger.info(f"  Placement {i} (frame {frame_num}, step {current_step}): {final_action_desc}")

                    # === VLM PASS 2d: STEP COMPLETION VERIFICATION ===
                    # Check if current step is complete and determine step advancement
                    logger.debug(f"  Placement {i}: Pass 2d - Verifying step completion...")

                    # Get SAM3 current path for comparison
                    sam3_current_path = pass2b_result.get("sam3_current_path")

                    # Extract subassembly image paths from step data
                    # step_data["subassemblies"] is a list of dicts with structure: {"cropped_image_path": "...", "description": "...", "bounding_box": {...}}
                    subassemblies_data = step_data.get("subassemblies", [])
                    expected_subassembly_images = [
                        sa.get("cropped_image_path", "")
                        for sa in subassemblies_data
                        if isinstance(sa, dict) and sa.get("cropped_image_path")
                    ]

                    if not expected_subassembly_images:
                        logger.debug(f"  No expected subassembly images found for step {current_step}, skipping Pass 2d verification")
                        parts_used_in_current_step += 1
                        continue

                    try:
                        step_verification = await self._pass2d_verify_step_completion(
                            current_sam3_path=sam3_current_path,
                            step_number=current_step,
                            expected_subassembly_images=expected_subassembly_images,
                            manual_id=manual_id
                        )

                        # Store Pass 2d result in action_data
                        action_data["step_verification"] = step_verification

                        # Log Pass 2d verification
                        with open(comparison_log_path, "a", encoding="utf-8") as log_file:
                            log_file.write("VLM PASS 2d - STEP COMPLETION VERIFICATION:\n")
                            log_file.write(f"  is_step_complete: {step_verification.get('is_step_complete', False)}\n")
                            log_file.write(f"  current_step_verified: {step_verification.get('current_step_verified', current_step)}\n")
                            log_file.write(f"  should_advance_to_step: {step_verification.get('should_advance_to_step', current_step)}\n")
                            log_file.write(f"  confidence: {step_verification.get('confidence', 0.0)}\n")
                            log_file.write(f"  reasoning: {step_verification.get('reasoning', 'N/A')}\n")
                            discrepancies = step_verification.get('discrepancies')
                            if discrepancies:
                                log_file.write(f"  discrepancies: {json.dumps(discrepancies, indent=4)}\n")
                            log_file.write("\n")

                        # Determine step advancement based on Pass 2d result
                        is_step_complete = step_verification.get("is_step_complete", False)
                        should_advance_to_step = step_verification.get("should_advance_to_step", current_step)

                        if is_step_complete and should_advance_to_step > current_step:
                            logger.info(f"  Step {current_step} verified complete by Pass 2d, advancing to step {should_advance_to_step}")
                            current_step = should_advance_to_step
                            parts_used_in_current_step = 0
                        else:
                            # Track parts usage for fallback (in case verification doesn't trigger)
                            parts_used_in_current_step += 1
                            logger.debug(f"  Step {current_step} not yet complete ({parts_used_in_current_step} parts placed so far)")

                    except Exception as e:
                        logger.warning(f"  Pass 2d step verification failed: {e}. Falling back to part counting.")
                        logger.debug(f"  Pass 2d error details:", exc_info=True)

                        # Fallback to part counting if Pass 2d fails
                        parts_used_in_current_step += 1
                        expected_parts_count = step_data.get("total_individual_parts", 0)
                        if parts_used_in_current_step >= expected_parts_count and expected_parts_count > 0:
                            logger.info(f"  Step {current_step} complete by part count ({parts_used_in_current_step}/{expected_parts_count} parts placed), advancing to step {current_step + 1}")
                            current_step += 1
                            parts_used_in_current_step = 0

                        # Log failure
                        with open(comparison_log_path, "a", encoding="utf-8") as log_file:
                            log_file.write("VLM PASS 2d - STEP COMPLETION VERIFICATION:\n")
                            log_file.write(f"  ERROR: {str(e)}\n\n")

                except Exception as e:
                    logger.warning(f"  Pass 2c reconciliation failed for placement {i}: {e}")
                    # Use Pass 2b result as fallback
                    action_data: Dict[str, Any] = {
                        "placement_index": i,
                        "frame_number": frame_num,
                        "timestamp": timestamp,
                        "frame_path": relative_frame_path,
                        "action_description": action_desc,
                        "new_parts": [new_part_detected] if new_part_detected else [],
                        "spatial_position": spatial_position,
                        "box_2d": box_2d_original,  # Bounding box for original frame
                        "box_2d_sam3": box_2d_sam3,  # Bounding box for SAM3 frame (for reference)
                        "annotated_frame_path": str(annotated_frame_path.relative_to(self.settings.data_dir)) if annotated_frame_path else None,
                        "is_subassembly_switch": is_subassembly_switch,
                        "current_subassembly": self.subassembly_tracker.get_current_subassembly(),
                        "manual_step": current_step,
                        "confidence": confidence,
                        "reconciliation": {
                            "error": str(e)
                        }
                    }
                    validated_placements.append(action_data)
                    last_accepted_candidate = current_frame

                    # Log failure
                    with open(comparison_log_path, "a", encoding="utf-8") as log_file:
                        log_file.write("VLM PASS 2c - RECONCILIATION:\n")
                        log_file.write(f"  ERROR: {str(e)}\n\n")

                    logger.info(f"  Placement {i} (frame {frame_num}, step {current_step}): {action_desc} (reconciliation failed)")

                    # Track parts usage even on failure
                    parts_used_in_current_step += 1
                    expected_parts_count = step_data.get("total_individual_parts", 0)
                    if parts_used_in_current_step >= expected_parts_count and expected_parts_count > 0:
                        logger.info(f"  Step {current_step} complete by part count ({parts_used_in_current_step}/{expected_parts_count} parts placed), advancing to step {current_step + 1}")
                        current_step += 1
                        parts_used_in_current_step = 0

                # Write cache incrementally
                self._write_validated_placements_cache(
                    manual_id, video_id, validated_placements, len(placement_candidates)
                )

                new_calls += 1

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

    # ── VLM Pass 2c: Part Reconciliation ─────────────────────────────────────

    async def _pass2c_reconcile_all_sources(
        self,
        pass2a_result: Dict[str, Any],
        pass2b_result: Dict[str, Any],
        placement: Dict[str, Any],
        expected_parts: List[Dict[str, Any]],
        step_number: int,
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Pass 2c: Reconcile Pass 2a + Pass 2b + expected parts to determine final part identity.

        Args:
            pass2a_result: Result from Pass 2a action analysis
            pass2b_result: Result from Pass 2b SAM3 comparison
            placement: Placement metadata with annotated_frame_path and box_2d
            expected_parts: List of expected parts for the current step
            step_number: Current manual step number
            manual_id: Manual identifier for loading images

        Returns:
            Reconciliation result with final part identification
        """
        # Build content for VLM call
        content: List[Dict[str, Any]] = []

        # Get original frame path and bounding boxes
        frame_path = placement.get("frame_path")
        box_2d_original = placement.get("box_2d_original")  # Bounding box for original frame (from Pass 2a)
        box_2d_sam3 = placement.get("box_2d_sam3")  # Bounding box for SAM3 frame (from Pass 2b)

        # Construct full frame path
        if frame_path:
            full_frame_path = self.settings.data_dir / frame_path
        else:
            full_frame_path = None

        # Get SAM3 segmented frame path from Pass 2b result
        sam3_current_path = pass2b_result.get("sam3_current_path")
        if sam3_current_path:
            full_sam3_path = self.settings.data_dir / sam3_current_path
        else:
            full_sam3_path = None

        # 1. Add CROPPED view from ORIGINAL FRAME (Pass 2a context - shows placement on assembly)
        # Use box_2d_original since it's based on the original frame coordinate system
        if full_frame_path and full_frame_path.exists() and box_2d_original:
            cropped_img = _crop_to_bbox(str(full_frame_path), box_2d_original, padding=30)
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
                        "text": "VIEW 1 - Pass 2a PLACEMENT CONTEXT (cropped from original frame - shows what the part covers on assembly):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{cropped_b64}"}
                    })
                    logger.debug(f"Added Pass 2a cropped bbox view for placement {placement['placement_index']}")
                except Exception as e:
                    logger.warning(f"Could not process Pass 2a cropped image: {e}")
            else:
                logger.debug(f"Could not crop to bbox from original frame for placement {placement['placement_index']}")
        else:
            logger.debug(f"Skipping Pass 2a bbox crop - frame_path: {full_frame_path}, box_2d_original: {box_2d_original}")

        # 2. Add CROPPED view from SAM3 SEGMENTED FRAME (Pass 2b context - shows isolated part)
        # Use box_2d_sam3 since it's based on the SAM3 frame coordinate system
        if full_sam3_path and full_sam3_path.exists() and box_2d_sam3:
            cropped_sam3_img = _crop_to_bbox(str(full_sam3_path), box_2d_sam3, padding=30)
            if cropped_sam3_img:
                try:
                    # Resize cropped image to reasonable size (keeping aspect ratio)
                    max_size = 600
                    if cropped_sam3_img.width > max_size or cropped_sam3_img.height > max_size:
                        if cropped_sam3_img.width > cropped_sam3_img.height:
                            new_w = max_size
                            new_h = int(max_size * cropped_sam3_img.height / cropped_sam3_img.width)
                        else:
                            new_h = max_size
                            new_w = int(max_size * cropped_sam3_img.width / cropped_sam3_img.height)
                        cropped_sam3_img = cropped_sam3_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

                    cropped_sam3_b64 = _image_to_b64(cropped_sam3_img)
                    content.append({
                        "type": "text",
                        "text": "VIEW 2 - Pass 2b SAM3 SEGMENTED (cropped from SAM3 frame - shows isolated part on white background):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{cropped_sam3_b64}"}
                    })
                    logger.debug(f"Added Pass 2b SAM3 cropped bbox view for placement {placement['placement_index']}")
                except Exception as e:
                    logger.warning(f"Could not process Pass 2b SAM3 cropped image: {e}")
            else:
                logger.debug(f"Could not crop to bbox from SAM3 frame for placement {placement['placement_index']}")
        else:
            logger.debug(f"Skipping Pass 2b SAM3 bbox crop - sam3_path: {full_sam3_path}, box_2d_sam3: {box_2d_sam3}")

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

        # Build expected parts JSON (without images)
        expected_parts_json = [
            {
                "description": part["description"],
                "quantity": part.get("quantity", 1)
            }
            for part in expected_parts
        ]

        # Build prompt from template with all three sources
        prompt = self.pass2c_template.replace(
            "{pass2a_result}", json.dumps(pass2a_result, indent=2)
        ).replace(
            "{pass2b_result}", json.dumps(pass2b_result, indent=2)
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

    # ── VLM Pass 2d: Step Completion Verification ────────────────────────────

    async def _pass2d_verify_step_completion(
        self,
        current_sam3_path: str,
        step_number: int,
        expected_subassembly_images: List[str],
        manual_id: str
    ) -> Dict[str, Any]:
        """
        Pass 2d: Verify whether the current assembly state has reached step completion.

        Performs pure visual comparison between the current SAM3-segmented assembly
        and the expected subassembly image(s) for the current step. No parts list needed.

        Args:
            current_sam3_path: Path to current SAM3 segmented assembly image
            step_number: Current step number to verify
            expected_subassembly_images: List of expected subassembly image paths for this step
            manual_id: Manual identifier for loading images

        Returns:
            Step completion verification result with is_step_complete, should_advance_to_step, etc.
        """
        # Build content for VLM call
        content: List[Dict[str, Any]] = []

        # 1. Add CURRENT ASSEMBLY (SAM3 segmented - clean view)
        if current_sam3_path:
            full_sam3_path = self.settings.data_dir / current_sam3_path
            if full_sam3_path.exists():
                try:
                    current_img = _resize_image(str(full_sam3_path), width=800)
                    current_b64 = _image_to_b64(current_img)
                    content.append({
                        "type": "text",
                        "text": "CURRENT ASSEMBLY (SAM3-segmented, white background):"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{current_b64}"}
                    })
                    logger.debug(f"Added current SAM3 assembly for step {step_number} verification")
                except Exception as e:
                    logger.warning(f"Could not load current SAM3 image: {e}")
            else:
                logger.warning(f"Current SAM3 image not found: {full_sam3_path}")
        else:
            logger.warning("No current SAM3 path provided for step completion verification")

        # 2. Add EXPECTED SUBASSEMBLY images
        if expected_subassembly_images:
            content.append({
                "type": "text",
                "text": f"EXPECTED SUBASSEMBLY FOR STEP {step_number} COMPLETION (reference from manual):"
            })
            for idx, subassembly_path in enumerate(expected_subassembly_images):
                full_subassembly_path = self.settings.data_dir / subassembly_path
                if full_subassembly_path.exists():
                    try:
                        expected_img = _resize_image(str(full_subassembly_path), width=800)
                        expected_b64 = _image_to_b64(expected_img)
                        content.append({
                            "type": "text",
                            "text": f"Expected Subassembly View {idx + 1}:"
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{expected_b64}"}
                        })
                    except Exception as e:
                        logger.warning(f"Could not load expected subassembly image {subassembly_path}: {e}")
                else:
                    logger.warning(f"Expected subassembly image not found: {full_subassembly_path}")
        else:
            logger.warning(f"No expected subassembly images provided for step {step_number}")

        # Build prompt from template (pure visual comparison, no parts list)
        prompt = self.pass2d_template.replace(
            "{step_number}", str(step_number)
        )

        content.append({"type": "text", "text": prompt})

        # Make VLM call
        messages = [{"role": "user", "content": content}]
        raw = self.vlm._litellm_with_retry(messages)
        result = _parse_json(raw)

        logger.debug(f"Pass 2d result for step {step_number}: {result}")
        return result

    # ── VLM Pass 3: Atomic Sub-step Generation ───────────────────────────────

    async def _generate_atomic_substeps(
        self,
        manual_id: str,
        video_id: str,
        reconciled_placements: Dict[str, Any],
        manual_data: Dict[str, Any],
        all_frames: List[Path]
    ) -> Dict[str, Any]:
        """
        VLM Pass 3: Generate atomic sub-steps (1 part per sub-step) with spatial positioning.

        Uses validated placements (with Pass 2c reconciliation) to create the final video_enhanced.json structure.
        Sends enhanced.json + validated_placements.json as text, plus sample frames.
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

    def _get_default_atomic_substeps_prompt(self) -> str:
        return "Generate atomic sub-steps. Return JSON."

    def _get_default_placement_reconciliation_prompt(self) -> str:
        return "Reconcile placement against expected parts. Return JSON with verified, matched_part, video_detection_correct, correction, and reasoning fields."

    def _get_default_action_frame_analysis_prompt(self) -> str:
        return "Analyze action frames to identify what LEGO part is being manipulated. Return JSON with part_being_manipulated, color, part_type, size, action_type (pickup/adjustment), confidence, and reasoning."

    def _get_default_unified_placement_analysis_prompt(self) -> str:
        return "Analyze the sequence of frames (previous placement → action frames → current placement) to determine if a new part was added. Return JSON with has_new_part, is_duplicate, what_changed, new_parts_added (with stud_count), action_description, spatial_position, box_2d, consistent_with_sequence, confidence, and reasoning."
