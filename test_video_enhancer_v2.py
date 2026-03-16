"""
Test script for Video Enhancer V2.

Tests the improved 3-pass pipeline with the first 500 frames of changi_airport.mp4.
"""

import asyncio
import json
from pathlib import Path
from loguru import logger

from ingestion.vlm_extractor import VLMExtractor
from backend.services.data_service import DataService
from backend.services.video_enhancer_v2 import VideoEnhancerV2
from config.settings import get_settings


async def main():
    """Run video enhancement V2 test."""
    logger.info("=" * 80)
    logger.info("Testing Video Enhancer V2 (Improved 3-Pass Pipeline)")
    logger.info("=" * 80)

    # Initialize services
    settings = get_settings()
    vlm_extractor = VLMExtractor(
        vlm_model=settings.vlm_model,
        api_key=settings.gemini_api_key,
        max_retries=settings.vlm_max_retries
    )
    data_service = DataService()

    # Create V2 enhancer
    enhancer_v2 = VideoEnhancerV2(vlm_extractor, data_service, settings)

    # Test parameters
    manual_id = "111111"
    video_id = "changi_airport"
    max_frames = 500  # Test with first 500 frames

    logger.info(f"Manual ID: {manual_id}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Max frames: {max_frames}")
    logger.info("")

    # Verify inputs exist
    video_path = settings.data_dir / "videos" / manual_id / f"{video_id}.mp4"
    enhanced_path = settings.data_dir / "processed" / manual_id / "enhanced.json"

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    if not enhanced_path.exists():
        logger.error(f"Enhanced.json not found: {enhanced_path}")
        return

    logger.info(f"Video path: {video_path}")
    logger.info(f"Enhanced.json path: {enhanced_path}")
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
