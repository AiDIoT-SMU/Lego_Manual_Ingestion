"""
Quick test script for the ingestion pipeline.
Tests the complete flow: PDF → VLM → Cropping → JSON
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from ingestion.pipeline import IngestionPipeline
from loguru import logger


def test_ingestion():
    """
    Test the ingestion pipeline with a sample manual.

    Before running:
    1. Ensure .env file exists with valid GEMINI_API_KEY
    2. Place a test PDF in the project directory or update pdf_path below
    """
    logger.info("Starting ingestion test...")

    # Get settings
    settings = get_settings()

    # Configure test parameters
    manual_id = "test_manual"
    pdf_path = Path("path/to/your/test/manual.pdf")  # UPDATE THIS PATH
    instruction_pages = [1, 2, 3]  # UPDATE WITH ACTUAL PAGE NUMBERS

    # Verify PDF exists
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.info("Please update pdf_path in test_ingestion.py to point to a valid PDF")
        return False

    # Create pipeline
    pipeline = IngestionPipeline(settings)

    try:
        # Run ingestion
        logger.info(f"Processing manual {manual_id}...")
        result = pipeline.process_manual(
            manual_id=manual_id,
            pdf_path=pdf_path,
            instruction_pages=instruction_pages
        )

        # Verify results
        logger.info(f"✓ Extraction complete!")
        logger.info(f"  - Total steps: {len(result.steps)}")

        # Count parts and subassemblies
        total_parts = sum(len(step.parts_required) for step in result.steps)
        total_subassemblies = sum(len(step.subassemblies) for step in result.steps)

        logger.info(f"  - Total parts: {total_parts}")
        logger.info(f"  - Total subassemblies: {total_subassemblies}")

        # Check cropped images
        cropped_parts = sum(
            1 for step in result.steps
            for part in step.parts_required
            if part.cropped_image_path
        )
        cropped_subs = sum(
            1 for step in result.steps
            for sub in step.subassemblies
            if sub.cropped_image_path
        )

        logger.info(f"  - Cropped part images: {cropped_parts}")
        logger.info(f"  - Cropped subassembly images: {cropped_subs}")

        # Show output locations
        logger.info(f"\nOutput files:")
        logger.info(f"  - Extraction JSON: data/processed/{manual_id}/extraction.json")
        logger.info(f"  - Enhanced JSON: data/processed/{manual_id}/enhanced.json")
        logger.info(f"  - Cropped images: data/cropped/{manual_id}/")

        # Show sample step
        if result.steps:
            step = result.steps[0]
            logger.info(f"\nSample step (Step {step.step_number}):")
            logger.info(f"  - Parts: {[p.description for p in step.parts_required[:3]]}")
            logger.info(f"  - Actions: {step.actions[:2]}")

        return True

    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LEGO Assembly Refactored - Ingestion Pipeline Test")
    logger.info("=" * 70)

    success = test_ingestion()

    if success:
        logger.info("\n✓ Test completed successfully!")
    else:
        logger.error("\n✗ Test failed. Check the logs above for details.")

    sys.exit(0 if success else 1)
