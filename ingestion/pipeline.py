"""
Ingestion Pipeline: Orchestrates the full manual processing workflow.
PDF → VLM Extraction → Image Cropping → JSON Output
"""

import json
from pathlib import Path
from typing import List, Optional
from loguru import logger

from config.settings import Settings
from .pdf_processor import PDFProcessor
from .vlm_extractor import VLMExtractor
from .image_cropper import ImageCropper
from .schemas import ManualExtraction
from .url_handler import URLHandler


class IngestionPipeline:
    """Orchestrates the complete ingestion workflow for LEGO manuals."""

    def __init__(self, settings: Settings):
        """
        Initialize the ingestion pipeline.

        Args:
            settings: Application settings with API keys and paths
        """
        self.settings = settings

        prompts_dir = Path(__file__).parent.parent / "prompts"

        # Call 1 — semantic prompt (text only, no coordinates)
        with open(prompts_dir / "step_extraction.txt") as f:
            self.prompt_template = f.read()

        # Call 2 — spatial prompt template (bounding boxes only)
        with open(prompts_dir / "step_spatial.txt") as f:
            spatial_prompt_template = f.read()

        # Initialize VLM extractor
        self.vlm = VLMExtractor(
            vlm_model=settings.vlm_model,
            api_key=settings.gemini_api_key,
            max_retries=settings.vlm_max_retries,
            spatial_prompt_template=spatial_prompt_template,
        )

        logger.info("IngestionPipeline initialized")

    def process_manual(
        self,
        manual_id: str,
        pdf_path: Path,
        instruction_pages: Optional[List[int]] = None
    ) -> ManualExtraction:
        """
        Process a complete LEGO instruction manual.

        Pipeline stages:
        1. Extract pages from PDF
        2. Run VLM extraction to get structured data with bounding boxes
        3. Crop parts and subassemblies using bounding boxes
        4. Save JSON outputs (extraction.json and enhanced.json)

        Args:
            manual_id: Unique identifier for this manual
            pdf_path: Path to the PDF file
            instruction_pages: List of page numbers to process (1-indexed).
                              If None, processes all pages.

        Returns:
            ManualExtraction object with all steps and cropped image paths

        Raises:
            Exception: If any stage of the pipeline fails
        """
        logger.info(f"Starting ingestion pipeline for manual {manual_id}")

        # Create output directories for this manual
        manual_output_dir = self.settings.manual_dir / manual_id
        processed_output_dir = self.settings.processed_dir / manual_id
        cropped_output_dir = self.settings.cropped_dir / manual_id

        manual_output_dir.mkdir(parents=True, exist_ok=True)
        processed_output_dir.mkdir(parents=True, exist_ok=True)
        cropped_output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Extract pages from PDF
        logger.info(f"Stage 1: Extracting pages from PDF {pdf_path}")
        processor = PDFProcessor(output_dir=manual_output_dir)
        image_paths = processor.process_pdf(
            pdf_path=pdf_path,
            page_numbers=instruction_pages
        )
        logger.info(f"Extracted {len(image_paths)} pages")

        # Stage 2: VLM extraction
        logger.info(f"Stage 2: Running VLM extraction on {len(image_paths)} pages")
        steps = self.vlm.extract_steps(
            image_paths=image_paths,
            prompt_template=self.prompt_template
        )
        extraction = ManualExtraction(manual_id=manual_id, steps=steps)
        logger.info(f"Extracted {len(steps)} total steps")

        # Save extraction JSON (without cropped images)
        extraction_path = processed_output_dir / "extraction.json"
        with open(extraction_path, 'w') as f:
            json.dump(extraction.model_dump(), f, indent=2)
        logger.info(f"Saved extraction JSON to {extraction_path}")

        # Stage 3: Crop images
        logger.info(f"Stage 3: Cropping parts and subassemblies")
        cropper = ImageCropper(output_dir=cropped_output_dir)
        enhanced = cropper.crop_and_save(extraction)
        logger.info(f"Image cropping complete")

        # Save enhanced JSON (with cropped image paths)
        enhanced_path = processed_output_dir / "enhanced.json"
        with open(enhanced_path, 'w') as f:
            json.dump(enhanced.model_dump(), f, indent=2)
        logger.info(f"Saved enhanced JSON to {enhanced_path}")

        logger.info(f"✓ Ingestion pipeline complete for manual {manual_id}")

        return enhanced

    def process_image_directory(
        self,
        manual_id: str,
        image_dir: Path,
        image_numbers: Optional[List[int]] = None
    ) -> ManualExtraction:
        """
        Process a directory of instruction images (alternative to PDF input).

        Args:
            manual_id: Unique identifier for this manual
            image_dir: Path to directory containing instruction images
            image_numbers: List of image indices to process (1-indexed).
                          If None, processes all images.

        Returns:
            ManualExtraction object with all steps and cropped image paths
        """
        logger.info(f"Starting ingestion pipeline for image directory {image_dir}")

        # Create output directories
        manual_output_dir = self.settings.manual_dir / manual_id
        processed_output_dir = self.settings.processed_dir / manual_id
        cropped_output_dir = self.settings.cropped_dir / manual_id

        manual_output_dir.mkdir(parents=True, exist_ok=True)
        processed_output_dir.mkdir(parents=True, exist_ok=True)
        cropped_output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Copy images
        logger.info(f"Stage 1: Processing image directory")
        processor = PDFProcessor(output_dir=manual_output_dir)
        image_paths = processor.process_image_directory(
            dir_path=image_dir,
            image_numbers=image_numbers
        )
        logger.info(f"Copied {len(image_paths)} images")

        # Stage 2: VLM extraction
        logger.info(f"Stage 2: Running VLM extraction on {len(image_paths)} images")
        steps = self.vlm.extract_steps(
            image_paths=image_paths,
            prompt_template=self.prompt_template
        )
        extraction = ManualExtraction(manual_id=manual_id, steps=steps)
        logger.info(f"Extracted {len(steps)} total steps")

        # Save extraction JSON
        extraction_path = processed_output_dir / "extraction.json"
        with open(extraction_path, 'w') as f:
            json.dump(extraction.model_dump(), f, indent=2)
        logger.info(f"Saved extraction JSON to {extraction_path}")

        # Stage 3: Crop images
        logger.info(f"Stage 3: Cropping parts and subassemblies")
        cropper = ImageCropper(output_dir=cropped_output_dir)
        enhanced = cropper.crop_and_save(extraction)

        # Save enhanced JSON
        enhanced_path = processed_output_dir / "enhanced.json"
        with open(enhanced_path, 'w') as f:
            json.dump(enhanced.model_dump(), f, indent=2)
        logger.info(f"Saved enhanced JSON to {enhanced_path}")

        logger.info(f"✓ Ingestion pipeline complete for manual {manual_id}")

        return enhanced

    def process_url(
        self,
        manual_id: str,
        url: str,
        instruction_pages: Optional[List[int]] = None
    ) -> ManualExtraction:
        """
        Download a PDF manual from a URL and run the ingestion pipeline.

        Downloads the PDF to a temporary directory, runs the standard
        process_manual pipeline, then cleans up the temporary file.

        Args:
            manual_id: Unique identifier for this manual
            url: Direct URL to the PDF manual
            instruction_pages: List of page numbers to process (1-indexed).
                              If None, processes all pages.

        Returns:
            ManualExtraction object with all steps and cropped image paths
        """
        logger.info(f"Starting URL ingestion for manual {manual_id}: {url}")

        url_handler = URLHandler()
        try:
            pdf_path = url_handler.download_pdf(url)
            return self.process_manual(manual_id, pdf_path, instruction_pages)
        finally:
            url_handler.cleanup()
