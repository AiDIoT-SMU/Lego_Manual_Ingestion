"""
Video Enhancer V2: Improved 3-pass pipeline with quality filtering and context awareness.

Three VLM call pipeline:
1. Frame Classification: Batch classify frames with quality metrics (action vs placement_candidate).
2. Placement Validation: Context-aware analysis using enhanced.json parts list + duplicate detection.
3. Atomic Sub-steps: Generate 1-part-per-sub-step instructions with spatial positioning.
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
from backend.services.video_quality_filter import VideoQualityFilter
from backend.services.video_state_tracker import AssemblyStateTracker, SubassemblyTracker
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

        # Initialize quality filter and state trackers
        self.quality_filter = VideoQualityFilter(
            blur_threshold=100.0,
            stability_threshold=0.95,
            hand_detection_enabled=True
        )
        self.state_tracker = AssemblyStateTracker(hash_size=16, similarity_threshold=10)
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
        4. VLM Pass 3: Generate atomic sub-steps (1 part per sub-step) with spatial positioning.

        Args:
            manual_id: Manual identifier
            video_id: Video identifier
            max_frames: Optional limit on frames to process (for testing)
        """
        logger.info(f"Starting video enhancement V2 for manual {manual_id}, video {video_id}")

        # Load manual data (needed for Pass 2 context)
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
            placement_candidates, manual_data, manual_id, video_id
        )
        logger.info(f"Pass 2 complete: {len(validated_placements)} unique validated placements")

        # Write validated placements cache
        validated_placements_data = self._write_validated_placements_cache(
            manual_id, video_id, validated_placements, len(placement_candidates)
        )

        # === VLM PASS 3: Atomic Sub-step Generation ===
        logger.info("VLM Pass 3: Generating atomic sub-steps...")
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

    # ── VLM Pass 2: Context-Aware Placement Validation ───────────────────────

    async def _validate_placements_with_context(
        self,
        placement_candidates: List[Dict[str, Any]],
        manual_data: Dict[str, Any],
        manual_id: str,
        video_id: str
    ) -> List[Dict[str, Any]]:
        """
        VLM Pass 2: Validate placement candidates using context from enhanced.json.

        For each consecutive pair:
        - Load expected parts list from CURRENT step (follows manual sequence)
        - Detect what new part was added
        - Check for duplicates using perceptual hashing
        - Track subassembly switches
        """
        if not placement_candidates:
            return []

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

        # Reset state trackers
        self.state_tracker.reset()
        self.subassembly_tracker.reset()

        validated_placements: List[Dict[str, Any]] = []
        new_calls = 0

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
                # Register with state tracker
                self.state_tracker.register_placement(
                    current_frame["frame_path"], frame_num, cached
                )
                logger.info(
                    f"  Placement {i} (frame {frame_num}): "
                    f"[cached] {cached.get('action_description', 'no description')}"
                )
                continue

            try:
                # PRE-CHECK: Use perceptual hash to detect duplicates BEFORE expensive VLM call
                is_hash_duplicate, original = self.state_tracker.is_duplicate_state(
                    self.state_tracker.compute_state_hash(current_frame["frame_path"]),
                    frame_num
                )

                if is_hash_duplicate:
                    logger.info(
                        f"  Placement {i} (frame {frame_num}): "
                        f"Duplicate detected via perceptual hash (matches frame {original['frame_number']}) - skipping VLM call"
                    )
                    continue

                current_img = _resize_image(str(current_frame["frame_path"]), width=800)
                current_b64 = _image_to_b64(current_img)

                # Get expected parts for current step + next step (lookahead)
                expected_parts = self._get_expected_parts_for_steps(
                    parts_by_step, current_step, lookahead_steps=1
                )

                # Load cropped part images for visual reference
                part_images = []
                for part in expected_parts:
                    img_path = part.get("cropped_image_path")
                    if img_path:
                        full_path = self.settings.data_dir / img_path
                        if full_path.exists():
                            try:
                                part_img = _resize_image(str(full_path), width=400)
                                part_b64 = _image_to_b64(part_img)
                                part_images.append({
                                    "description": part.get("description"),
                                    "step": part.get("step"),
                                    "quantity": part.get("quantity"),
                                    "image": part_b64
                                })
                            except Exception as e:
                                logger.warning(f"Failed to load part image {img_path}: {e}")

                # Load subassembly reference image for current step
                subassembly_images = []
                step_data = parts_by_step.get(current_step, {})
                for subassembly in step_data.get("subassemblies", []):
                    img_path = subassembly.get("cropped_image_path")
                    if img_path:
                        full_path = self.settings.data_dir / img_path
                        if full_path.exists():
                            try:
                                sub_img = _resize_image(str(full_path), width=600)
                                sub_b64 = _image_to_b64(sub_img)
                                subassembly_images.append({
                                    "description": subassembly.get("description"),
                                    "image": sub_b64
                                })
                            except Exception as e:
                                logger.warning(f"Failed to load subassembly image {img_path}: {e}")

                # Build prompt with step-aware context
                step_context = f"Current Manual Step: {current_step}\n"
                prompt_with_context = self.placement_validation_template.replace(
                    "{expected_parts_list}",
                    step_context + json.dumps(expected_parts, indent=2)
                )

                # Build content with cropped reference images
                if i == 0:
                    # First placement - no previous frame
                    content = [
                        {"type": "text", "text": "REFERENCE: Expected Parts (from instruction manual):"},
                    ]
                    for idx, part_img_data in enumerate(part_images):
                        content.append({
                            "type": "text",
                            "text": f"Part {idx+1}: {part_img_data['description']} (Step {part_img_data['step']}, Qty: {part_img_data['quantity']})"
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{part_img_data['image']}"}
                        })

                    if subassembly_images:
                        content.append({"type": "text", "text": f"REFERENCE: Expected Subassembly (Step {current_step}):"})
                        for sub_img_data in subassembly_images:
                            content.append({"type": "text", "text": f"{sub_img_data['description']}"})
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{sub_img_data['image']}"}
                            })

                    content.extend([
                        {"type": "text", "text": "CURRENT PLACEMENT FRAME (first placement — no previous frame):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                        {"type": "text", "text": prompt_with_context}
                    ])
                else:
                    # Compare with previous placement
                    prev_frame = placement_candidates[i - 1]
                    prev_img = _resize_image(str(prev_frame["frame_path"]), width=800)
                    prev_b64 = _image_to_b64(prev_img)

                    content = [
                        {"type": "text", "text": "REFERENCE: Expected Parts (from instruction manual):"},
                    ]
                    for idx, part_img_data in enumerate(part_images):
                        content.append({
                            "type": "text",
                            "text": f"Part {idx+1}: {part_img_data['description']} (Step {part_img_data['step']}, Qty: {part_img_data['quantity']})"
                        })
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{part_img_data['image']}"}
                        })

                    if subassembly_images:
                        content.append({"type": "text", "text": f"REFERENCE: Expected Subassembly (Step {current_step}):"})
                        for sub_img_data in subassembly_images:
                            content.append({"type": "text", "text": f"{sub_img_data['description']}"})
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{sub_img_data['image']}"}
                            })

                    content.extend([
                        {"type": "text", "text": "PREVIOUS PLACEMENT FRAME:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_b64}"}},
                        {"type": "text", "text": "CURRENT PLACEMENT FRAME:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}},
                        {"type": "text", "text": prompt_with_context}
                    ])

                messages = [{"role": "user", "content": content}]

                raw = self.vlm._litellm_with_retry(messages)
                result = _parse_json(raw)

                has_new_part = result.get("has_new_part", False)
                is_duplicate = result.get("is_duplicate_of_previous", False)

                if is_duplicate or not has_new_part:
                    logger.info(
                        f"  Placement {i} (frame {frame_num}): "
                        f"Duplicate or no new part detected, skipping"
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

                    action_data: Dict[str, Any] = {
                        "placement_index": i,
                        "frame_number": frame_num,
                        "timestamp": timestamp,
                        "frame_path": relative_frame_path,
                        "action_description": action_desc,
                        "new_parts": result.get("new_parts_added", []),
                        "spatial_position": result.get("spatial_position", {}),
                        "is_subassembly_switch": is_subassembly_switch,
                        "current_subassembly": self.subassembly_tracker.get_current_subassembly(),
                        "manual_step": current_step,
                        "confidence": result.get("confidence", 0.0)
                    }
                    validated_placements.append(action_data)

                    # Register with state tracker
                    self.state_tracker.register_placement(
                        current_frame["frame_path"], frame_num, action_data
                    )

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

    # ── VLM Pass 3: Atomic Sub-step Generation ───────────────────────────────

    async def _generate_atomic_substeps(
        self,
        manual_id: str,
        video_id: str,
        validated_placements: Dict[str, Any],
        manual_data: Dict[str, Any],
        all_frames: List[Path]
    ) -> Dict[str, Any]:
        """
        VLM Pass 3: Generate atomic sub-steps (1 part per sub-step) with spatial positioning.

        Sends enhanced.json + validated_placements.json as text, plus sample frames.
        """
        enhanced_json_str = json.dumps(manual_data, indent=2)
        placements_json_str = json.dumps(validated_placements, indent=2)

        prompt = self.atomic_substeps_template.replace(
            "{enhanced_json}", enhanced_json_str
        ).replace(
            "{validated_placements_json}", placements_json_str
        )

        # Build a frame_number → Path lookup for sample images
        frame_lookup: Dict[int, Path] = {
            int(f.stem.split('_')[-1]): f for f in all_frames
        }

        # Pick up to 5 evenly-spaced placements as visual context
        placements = validated_placements.get("placements", [])
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

        # Build mapping from placement sequence to frame paths
        placements_by_step_and_order: Dict[int, List[Dict[str, Any]]] = {}
        for placement in placements:
            step_num = placement.get("manual_step", 1)
            if step_num not in placements_by_step_and_order:
                placements_by_step_and_order[step_num] = []
            placements_by_step_and_order[step_num].append(placement)

        # Merge with original manual data and enrich sub-steps with frame paths
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

            # Enrich each sub-step with frame path
            enriched_sub_steps = []
            for idx, sub_step in enumerate(sub_steps):
                # Match sub-step to placement by order (assuming VLM maintains sequence)
                if idx < len(step_placements):
                    placement = step_placements[idx]
                    enriched_sub_steps.append({
                        **sub_step,
                        "frame_path": placement.get("frame_path"),
                        "frame_number": placement.get("frame_number"),
                        "timestamp": placement.get("timestamp")
                    })
                else:
                    # No matching placement found
                    enriched_sub_steps.append(sub_step)

            enhanced_steps.append({
                **step,
                "sub_steps": enriched_sub_steps,
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

    # ── fallback prompts ──────────────────────────────────────────────────────

    def _get_default_frame_quality_prompt(self) -> str:
        return "Classify frames with quality metrics. Return JSON array."

    def _get_default_placement_validation_prompt(self) -> str:
        return "Validate placements with context. Return JSON."

    def _get_default_atomic_substeps_prompt(self) -> str:
        return "Generate atomic sub-steps. Return JSON."
