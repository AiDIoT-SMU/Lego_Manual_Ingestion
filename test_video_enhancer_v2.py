"""
Test script for Video Enhancer V2.

Tests the improved 3-pass pipeline with the complete video.
"""

import asyncio
import json
import shutil
from pathlib import Path
from loguru import logger

from ingestion.vlm_extractor import VLMExtractor
from backend.services.data_service import DataService
from backend.services.video_enhancer_v2 import VideoEnhancerV2
from config.settings import get_settings


def clear_cache(settings, manual_id: str, video_id: str):
    """Clear all cache files related to this video processing."""
    logger.info("=" * 80)
    logger.info("CLEARING CACHE")
    logger.info("=" * 80)

    processed_dir = settings.data_dir / "processed" / manual_id
    videos_dir = settings.data_dir / "videos" / manual_id

    cache_items = [
        # VLM Pass 1: Frame quality classification cache
        (processed_dir / f"video_frame_quality_{video_id}.json", "Frame quality cache"),

        # VLM Pass 2: Validated placements cache
        (processed_dir / f"video_validated_placements_{video_id}.json", "Validated placements cache"),

        # VLM Pass 3: Reconciled placements cache
        (processed_dir / f"video_reconciled_placements_{video_id}.json", "Reconciled placements cache"),

        # VLM Pass 4: Final output
        (processed_dir / f"video_enhanced_v2_{video_id}.json", "Final enhanced output"),

        # Extracted frames directory
        (videos_dir / f"{video_id}_enhancement_frames_v2", "Extracted frames directory"),

        # Annotated frames from YOLO-World + SAM3
        (processed_dir / f"yolo_world_sam3_annotated_{video_id}", "YOLO-World + SAM3 annotated frames"),

        # Annotated frames from placement validation
        (processed_dir / f"validated_placement_annotated_{video_id}", "Validated placement annotated frames"),

        # Annotated frames from reconciliation
        (processed_dir / f"reconciled_placement_annotated_{video_id}", "Reconciled placement annotated frames"),
    ]

    removed_count = 0
    for path, description in cache_items:
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    logger.info(f"  ✓ Removed {description}: {path}")
                else:
                    path.unlink()
                    logger.info(f"  ✓ Removed {description}: {path}")
                removed_count += 1
            except Exception as e:
                logger.warning(f"  ✗ Failed to remove {description}: {e}")
        else:
            logger.debug(f"  - Not found: {description}")

    if removed_count == 0:
        logger.info("  No cache files found to remove")
    else:
        logger.info(f"\nRemoved {removed_count} cache items")

    logger.info("=" * 80)
    logger.info("")


async def main():
    """Run video enhancement V2 test."""
    logger.info("=" * 80)
    logger.info("Testing Video Enhancer V2 (Improved 3-Pass Pipeline)")
    logger.info("=" * 80)

    # Test parameters
    manual_id = "111111"
    video_id = "changi_airport"
    max_frames = 500  # Limit to first 500 frames for testing (None = entire video)

    # Initialize settings
    settings = get_settings()

    # Ask if user wants to clear cache
    logger.info("")
    logger.info(f"Manual ID: {manual_id}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Max frames: {max_frames if max_frames else 'All (entire video)'}")
    logger.info("")

    clear_cache_input = input("Do you want to clear cache before running? (y/n): ").strip().lower()

    if clear_cache_input in ['y', 'yes']:
        clear_cache(settings, manual_id, video_id)
    else:
        logger.info("Keeping existing cache (will resume from cached results)")
        logger.info("")

    # Initialize services
    vlm_extractor = VLMExtractor(
        vlm_model=settings.vlm_model,
        api_key=settings.gemini_api_key,
        max_retries=settings.vlm_max_retries,
        timeout=settings.vlm_timeout
    )
    data_service = DataService()

    # Create V2 enhancer
    enhancer_v2 = VideoEnhancerV2(vlm_extractor, data_service, settings)

    # Verify inputs exist
    video_path = settings.data_dir / "videos" / manual_id / f"{video_id}.mp4"
    enhanced_path = settings.data_dir / "processed" / manual_id / "enhanced.json"

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    if not enhanced_path.exists():
        logger.error(f"Enhanced.json not found: {enhanced_path}")
        return

    logger.info("")
    logger.info("STARTING VIDEO ENHANCEMENT")
    logger.info(f"  Video path: {video_path}")
    logger.info(f"  Enhanced.json path: {enhanced_path}")
    logger.info("")

    # Run the enhanced pipeline
    try:
        result = await enhancer_v2.enhance_manual_with_video(
            manual_id=manual_id,
            video_id=video_id,
            max_frames=max_frames
        )

        # Save output
        output_path = settings.data_dir / "processed" / manual_id / f"video_enhanced_v2_{video_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("")
        logger.info("=" * 80)
        logger.info("VIDEO ENHANCEMENT V2 COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Output saved to: {output_path}")
        logger.info("")

        # Show where annotated frames are saved
        annotated_frames_dir = settings.data_dir / "processed" / manual_id / f"yolo_world_sam3_annotated_{video_id}"
        if annotated_frames_dir.exists():
            frame_count = len(list(annotated_frames_dir.glob("*.jpg")))
            logger.info("ANNOTATED FRAMES:")
            logger.info(f"  Location: {annotated_frames_dir}")
            logger.info(f"  Count: {frame_count} frames with YOLO-World + SAM3 annotations")
            logger.info(f"  View these frames to see bounding boxes and object counts!")
            logger.info("")

        # Print summary statistics
        total_steps = len(result["steps"])
        total_substeps = sum(len(step.get("sub_steps", [])) for step in result["steps"])
        total_corrections = sum(len(step.get("corrections", [])) for step in result["steps"])

        logger.info("SUMMARY STATISTICS:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Total sub-steps: {total_substeps}")
        logger.info(f"  Total corrections: {total_corrections}")
        logger.info(f"  Average sub-steps per step: {total_substeps / total_steps:.1f}" if total_steps > 0 else "  No steps")
        logger.info("")

        # Show sample sub-steps from first step
        if result["steps"] and result["steps"][0].get("sub_steps"):
            logger.info("SAMPLE SUB-STEPS (Step 1):")
            for substep in result["steps"][0]["sub_steps"][:5]:  # Show first 5
                logger.info(f"  {substep['sub_step_number']}: {substep['action_description']}")
            if len(result["steps"][0]["sub_steps"]) > 5:
                logger.info(f"  ... and {len(result["steps"][0]["sub_steps"]) - 5} more")
            logger.info("")

        # Compare with cache files
        cache_quality = settings.data_dir / "processed" / manual_id / f"video_frame_quality_{video_id}.json"
        cache_validated = settings.data_dir / "processed" / manual_id / f"video_validated_placements_{video_id}.json"

        if cache_quality.exists():
            quality_data = json.loads(cache_quality.read_text())
            logger.info(f"PASS 1 CACHE: {len(quality_data)} frames classified")

        if cache_validated.exists():
            validated_data = json.loads(cache_validated.read_text())
            logger.info(f"PASS 2 CACHE: {len(validated_data.get('placements', []))} validated placements")
            logger.info(f"  (from {validated_data.get('total_placement_candidates', 0)} candidates)")

            # Show duplicate filtering effectiveness
            if validated_data.get('total_placement_candidates', 0) > 0:
                duplicate_rate = 1.0 - (len(validated_data.get('placements', [])) / validated_data.get('total_placement_candidates', 1))
                logger.info(f"  Duplicate filtering: {duplicate_rate * 100:.1f}% of candidates were duplicates")

    except Exception as e:
        logger.error(f"Error during video enhancement: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
