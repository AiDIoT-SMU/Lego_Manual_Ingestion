"""
Test script to re-run VLM extraction with improved prompt.
This will re-process manual 123456 to verify improved accuracy.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_settings
from ingestion.pipeline import IngestionPipeline
from loguru import logger
import json


def main():
    """Re-process manual 123456 with improved extraction prompt."""
    logger.info("Testing improved VLM extraction...")

    # Get settings
    settings = get_settings()

    # Manual to re-process
    manual_id = "123456"
    manual_dir = settings.manuals_dir / manual_id

    # Verify manual images exist
    if not manual_dir.exists():
        logger.error(f"Manual directory not found: {manual_dir}")
        return False

    # Count existing images
    images = sorted(manual_dir.glob("page_*.jpg"))
    logger.info(f"Found {len(images)} images in {manual_dir}")

    # Create pipeline
    pipeline = IngestionPipeline(settings)

    try:
        # Re-process using existing images
        logger.info(f"Re-processing manual {manual_id} with improved prompt...")
        result = pipeline.process_image_directory(
            manual_id=manual_id,
            image_dir=manual_dir
        )

        # Show result
        logger.success(f"✓ Extraction complete!")
        logger.info(f"  - Steps extracted: {len(result.steps)}")
        logger.info(f"  - Output: data/processed/{manual_id}/extraction.json")

        # Read and display the first step's parts to verify
        extraction_path = settings.processed_dir / manual_id / "extraction.json"
        if extraction_path.exists():
            with open(extraction_path) as f:
                data = json.load(f)

            logger.info("\n=== First Step Parts (for verification) ===")
            if data["steps"]:
                for i, part in enumerate(data["steps"][0]["parts_required"], 1):
                    logger.info(f"  {i}. {part['description']}")

            # Specifically check for the white bricks issue
            logger.info("\n=== Checking for white brick accuracy ===")
            for step in data["steps"]:
                for part in step["parts_required"]:
                    desc = part["description"].lower()
                    if "white" in desc and "brick" in desc:
                        logger.info(f"  Found: {part['description']}")

        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
