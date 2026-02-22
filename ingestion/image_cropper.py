"""
Image Cropper: Crops parts and subassemblies from instruction pages
using bounding boxes extracted by the VLM.
"""

from PIL import Image
from pathlib import Path
from typing import Optional
from loguru import logger

from .schemas import Step, ManualExtraction, BoundingBox


class ImageCropper:
    """Crops images based on bounding boxes and saves organized outputs."""

    def __init__(self, output_dir: Path):
        """
        Initialize image cropper.

        Args:
            output_dir: Base directory for saving cropped images
                       (will create parts/ and subassemblies/ subdirectories)
        """
        self.output_dir = output_dir
        self.parts_dir = output_dir / "parts"
        self.subassemblies_dir = output_dir / "subassemblies"

        # Create output directories
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.subassemblies_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ImageCropper initialized with output_dir: {output_dir}")

    def crop_and_save(self, extraction: ManualExtraction) -> ManualExtraction:
        """
        Crop all parts and subassemblies from source images.
        Updates the extraction object with cropped image paths.

        Args:
            extraction: ManualExtraction object with bounding boxes

        Returns:
            Updated ManualExtraction with cropped_image_path fields populated
        """
        logger.info(f"Cropping images for manual {extraction.manual_id}")

        total_parts_cropped = 0
        total_subassemblies_cropped = 0

        for step in extraction.steps:
            step_num = step.step_number

            # Crop parts
            for i, part in enumerate(step.parts_required):
                if part.bounding_box:
                    try:
                        filename = f"step_{step_num}_part_{i}.png"
                        cropped_path = self._crop_image(
                            step.source_page_path,
                            part.bounding_box,
                            filename,
                            self.parts_dir
                        )
                        # Update the part with the cropped image path (relative)
                        part.cropped_image_path = str(
                            cropped_path.relative_to(self.output_dir.parent.parent)
                        )
                        total_parts_cropped += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to crop part {i} from step {step_num}: {e}"
                        )

            # Crop subassemblies
            for i, subassembly in enumerate(step.subassemblies):
                if subassembly.bounding_box:
                    try:
                        filename = f"step_{step_num}_subassembly_{i}.png"
                        cropped_path = self._crop_image(
                            step.source_page_path,
                            subassembly.bounding_box,
                            filename,
                            self.subassemblies_dir
                        )
                        # Update the subassembly with the cropped image path (relative)
                        subassembly.cropped_image_path = str(
                            cropped_path.relative_to(self.output_dir.parent.parent)
                        )
                        total_subassemblies_cropped += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to crop subassembly {i} from step {step_num}: {e}"
                        )

        logger.info(
            f"Cropped {total_parts_cropped} parts and "
            f"{total_subassemblies_cropped} subassemblies"
        )

        return extraction

    def _crop_image(
        self,
        source_path: str,
        bbox: BoundingBox,
        filename: str,
        output_dir: Path
    ) -> Path:
        """
        Crop a region from the source image and save it.

        Args:
            source_path: Path to the source image
            bbox: Bounding box with x, y, width, height
            filename: Output filename
            output_dir: Directory to save the cropped image

        Returns:
            Path to the saved cropped image

        Raises:
            FileNotFoundError: If source image doesn't exist
            ValueError: If bounding box is invalid
            Exception: If cropping or saving fails
        """
        source_path_obj = Path(source_path)
        if not source_path_obj.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")

        # Validate bounding box
        if bbox.width <= 0 or bbox.height <= 0:
            raise ValueError(
                f"Invalid bounding box dimensions: width={bbox.width}, height={bbox.height}"
            )

        # Open source image
        img = Image.open(source_path)

        # Validate bbox coordinates are within image bounds
        if bbox.x < 0 or bbox.y < 0:
            logger.warning(
                f"Bounding box has negative coordinates: ({bbox.x}, {bbox.y}). "
                "Adjusting to (0, 0)"
            )
            bbox.x = max(0, bbox.x)
            bbox.y = max(0, bbox.y)

        if bbox.x + bbox.width > img.width or bbox.y + bbox.height > img.height:
            logger.warning(
                f"Bounding box extends beyond image bounds. "
                f"Image size: {img.width}x{img.height}, "
                f"Box: ({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height}). "
                "Adjusting to fit."
            )
            # Adjust width and height to fit within image
            bbox.width = min(bbox.width, img.width - bbox.x)
            bbox.height = min(bbox.height, img.height - bbox.y)

        # Crop using PIL's crop method (left, upper, right, lower)
        cropped = img.crop((
            bbox.x,
            bbox.y,
            bbox.x + bbox.width,
            bbox.y + bbox.height
        ))

        # Save cropped image
        output_path = output_dir / filename
        cropped.save(output_path, format='PNG')

        logger.debug(f"Cropped and saved: {filename}")

        return output_path
