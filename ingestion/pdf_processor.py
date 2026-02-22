"""
PDF Processor: Extracts pages from LEGO instruction manuals.
Handles PDF and image directory inputs.
"""

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

from PIL import Image
from pathlib import Path
from typing import List, Union
import shutil
from loguru import logger


class PDFProcessor:
    """Handles extraction of pages from PDF manuals or image directories."""

    def __init__(self, output_dir: Path):
        """
        Initialize PDF processor.

        Args:
            output_dir: Directory to save extracted page images
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(
        self,
        pdf_path: Path,
        page_numbers: List[int] = None
    ) -> List[str]:
        """
        Extract pages from PDF and convert to images.

        Args:
            pdf_path: Path to PDF file
            page_numbers: List of page numbers to extract (1-indexed).
                         If None, extracts all pages.

        Returns:
            List of paths to extracted page images

        Raises:
            RuntimeError: If PyMuPDF is not installed or extraction fails
        """
        logger.info(f"Processing PDF: {pdf_path}")

        if not HAS_PYMUPDF:
            raise RuntimeError(
                "PyMuPDF is not installed. Please install it with: pip install PyMuPDF"
            )

        try:
            doc = fitz.open(str(pdf_path))
            page_paths = []

            # Determine which pages to extract
            if page_numbers is None:
                pages_to_extract = range(len(doc))
            else:
                # Convert 1-indexed to 0-indexed
                pages_to_extract = [p - 1 for p in page_numbers if 0 < p <= len(doc)]

            for page_num in pages_to_extract:
                page = doc[page_num]
                # 2x zoom for better quality
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                output_path = self.output_dir / f"page_{page_num + 1:03d}.png"
                pix.save(str(output_path))
                page_paths.append(str(output_path))

                logger.debug(f"Extracted page {page_num + 1}/{len(doc)}")

            doc.close()
            logger.info(
                f"Extracted {len(page_paths)} pages from PDF using PyMuPDF"
            )
            return page_paths

        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}")
            raise RuntimeError(f"PDF extraction failed: {e}")

    def process_image_directory(
        self,
        dir_path: Path,
        image_numbers: List[int] = None
    ) -> List[str]:
        """
        Process a directory containing instruction images.
        Copies images to output directory for consistent path handling.

        Args:
            dir_path: Path to directory with images
            image_numbers: List of image indices to copy (1-indexed).
                          If None, copies all images.

        Returns:
            Sorted list of paths to copied images
        """
        logger.info(f"Processing image directory: {dir_path}")

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        source_images = sorted([
            p for p in dir_path.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        logger.info(f"Found {len(source_images)} images")

        # Determine which images to copy
        if image_numbers is None:
            images_to_copy = source_images
        else:
            images_to_copy = [
                source_images[i - 1]
                for i in image_numbers
                if 0 < i <= len(source_images)
            ]

        # Copy images to output directory
        copied_paths = []
        for i, source_path in enumerate(images_to_copy):
            # Use page_XXX.png naming convention like PDF extraction
            output_path = self.output_dir / f"page_{i + 1:03d}{source_path.suffix}"
            shutil.copy2(source_path, output_path)
            copied_paths.append(str(output_path))
            logger.debug(f"Copied {source_path.name} -> {output_path.name}")

        logger.info(f"Copied {len(copied_paths)} images to {self.output_dir}")
        return copied_paths

    def process_manual(
        self,
        input_path: Union[str, Path],
        page_numbers: List[int] = None
    ) -> List[str]:
        """
        Process a LEGO instruction manual (PDF or image directory).

        Args:
            input_path: Path to PDF file or directory containing images
            page_numbers: List of page/image numbers to extract (1-indexed).
                         If None, extracts all.

        Returns:
            List of paths to extracted/copied images

        Raises:
            ValueError: If input path is invalid
        """
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                return self.process_pdf(input_path, page_numbers)
            elif input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Single image file - copy to output directory
                output_path = self.output_dir / f"page_001{input_path.suffix}"
                shutil.copy2(input_path, output_path)
                return [str(output_path)]
        elif input_path.is_dir():
            return self.process_image_directory(input_path, page_numbers)

        raise ValueError(f"Invalid input path: {input_path}")
