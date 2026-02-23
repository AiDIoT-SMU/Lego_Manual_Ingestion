"""
Manual Input Handler: Preprocesses LEGO instruction manual images.
Provides step segmentation, image enhancement, and batch processing
as an optional preprocessing layer before VLM extraction.
"""

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

import shutil
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union
from loguru import logger


class ManualInputHandler:
    """
    Handles preprocessing of LEGO instruction manual images.

    Stages images into temp_pages/{manual_id}/ with page_XXX.png naming,
    and provides enhancement/segmentation utilities before VLM extraction.
    """

    def __init__(self, output_dir: Optional[Path] = None, manual_id: Optional[str] = None):
        """
        Initialize manual input handler.

        Args:
            output_dir: Base output directory. Images are staged under
                        output_dir/temp_pages/{manual_id}/
            manual_id: Manual identifier used to namespace the staging directory.
        """
        if manual_id and output_dir:
            self.output_dir = output_dir / "temp_pages" / manual_id
        else:
            self.output_dir = output_dir or Path("./temp_manual_pages")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_manual(self, input_path: Union[str, Path]) -> List[str]:
        """
        Stage a LEGO instruction manual (PDF or image directory) into temp_pages.

        Args:
            input_path: Path to a PDF file, image directory, or single image.

        Returns:
            List of staged page image paths (page_XXX.png naming).
        """
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                return self._process_pdf(input_path)
            elif input_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                return [str(input_path)]
        elif input_path.is_dir():
            return self._process_image_directory(input_path)

        raise ValueError(f"Invalid input path: {input_path}")

    def _process_pdf(self, pdf_path: Path) -> List[str]:
        """Extract pages from PDF into the staging directory."""
        logger.info(f"Processing PDF: {pdf_path}")

        if not HAS_PYMUPDF:
            raise RuntimeError(
                "PyMuPDF is not installed. Run: uv add pymupdf"
            )

        try:
            doc = fitz.open(pdf_path)
            page_paths = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                output_path = self.output_dir / f"page_{page_num + 1:03d}.png"
                pix.save(str(output_path))
                page_paths.append(str(output_path))
                logger.debug(f"Extracted page {page_num + 1}/{len(doc)}")

            doc.close()
            logger.info(f"Extracted {len(page_paths)} pages from PDF")
            return page_paths

        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}")
            raise RuntimeError(f"PDF extraction failed: {e}")

    def _process_image_directory(self, dir_path: Path) -> List[str]:
        """
        Copy images from a directory into the staging directory,
        renaming them to page_XXX.png for consistent downstream handling.

        Args:
            dir_path: Source directory containing instruction images.

        Returns:
            Sorted list of paths to copied images in the staging directory.
        """
        logger.info(f"Processing image directory: {dir_path}")

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        source_images = sorted([
            p for p in dir_path.iterdir()
            if p.suffix.lower() in image_extensions
        ])
        logger.info(f"Found {len(source_images)} images")

        copied_paths = []
        for i, source_path in enumerate(source_images):
            output_path = self.output_dir / f"page_{i + 1:03d}{source_path.suffix}"
            shutil.copy2(source_path, output_path)
            copied_paths.append(str(output_path))
            logger.debug(f"Copied {source_path.name} -> {output_path.name}")

        logger.info(f"Copied {len(copied_paths)} images to {self.output_dir}")
        return copied_paths

    def detect_step_boundaries(self, image_paths: List[str]) -> List[List[str]]:
        """
        Group pages that belong to the same assembly step.

        Currently uses a 1-page-per-step heuristic. Extend this method
        for OCR-based step number detection or visual similarity grouping.

        Args:
            image_paths: List of page image paths.

        Returns:
            List of step groups, each a list of one or more page paths.
        """
        logger.info("Detecting step boundaries...")
        step_groups = [[img] for img in image_paths]
        logger.info(f"Detected {len(step_groups)} steps")
        return step_groups

    def segment_multi_step_page(self, image_path: str) -> List[str]:
        """
        Split a single page containing multiple steps into separate images
        by detecting horizontal separator lines.

        Args:
            image_path: Path to the page image.

        Returns:
            List of paths to segmented step images, or the original path
            if no clear separators are found.
        """
        logger.info(f"Segmenting multi-step page: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return [image_path]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (img.shape[1] // 2, 1)
        )
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)

        line_positions = [
            i for i, row in enumerate(horizontal_lines)
            if np.sum(row) > img.shape[1] * 0.3 * 255
        ]

        if len(line_positions) < 2:
            logger.debug("No clear step separators detected, returning original page")
            return [image_path]

        segmented_paths = []
        prev_y = 0

        for i, y in enumerate(line_positions + [img.shape[0]]):
            if y - prev_y > img.shape[0] * 0.1:
                segment = img[prev_y:y, :]
                output_path = self.output_dir / f"{Path(image_path).stem}_segment_{i + 1}.png"
                cv2.imwrite(str(output_path), segment)
                segmented_paths.append(str(output_path))
                prev_y = y

        logger.info(f"Segmented into {len(segmented_paths)} sub-steps")
        return segmented_paths if segmented_paths else [image_path]

    def preprocess_image(self, image_path: str, enhance: bool = True) -> str:
        """
        Apply contrast and sharpness enhancement to improve VLM recognition.

        Args:
            image_path: Path to the source image.
            enhance: Whether to apply enhancement (default: True).

        Returns:
            Path to the preprocessed image (written alongside originals).
        """
        if not enhance:
            return image_path

        logger.debug(f"Preprocessing image: {image_path}")

        from PIL import ImageEnhance
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.3)

        output_path = self.output_dir / f"preprocessed_{Path(image_path).name}"
        img.save(output_path)
        return str(output_path)
