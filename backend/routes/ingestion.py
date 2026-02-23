"""
Ingestion Routes: API endpoints for triggering manual processing.
"""

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List, Optional
from pathlib import Path
import json
from loguru import logger

from config.settings import get_settings
from ingestion.pipeline import IngestionPipeline


router = APIRouter()


@router.post("/pdf")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    instruction_pages: str = Form(None),  # JSON array as string, e.g., "[13,14,15]"
    pdf_file: UploadFile = File(...)
):
    """
    Upload a PDF manual and trigger ingestion pipeline.

    The ingestion runs in the background and processes:
    1. PDF page extraction
    2. VLM step extraction with bounding boxes
    3. Image cropping for parts and subassemblies
    4. JSON output generation

    Args:
        manual_id: Unique identifier for this manual
        instruction_pages: JSON array of page numbers to process (1-indexed),
                          e.g., "[13,14,15,16]". If not provided, processes all pages.
        pdf_file: PDF file upload

    Returns:
        Status message with manual_id

    Example:
        curl -X POST "http://localhost:8000/api/ingest/pdf" \\
             -F "manual_id=6262059" \\
             -F "instruction_pages=[13,14,15,16]" \\
             -F "pdf_file=@manual.pdf"
    """
    settings = get_settings()

    # Parse instruction pages if provided
    page_numbers: Optional[List[int]] = None
    if instruction_pages:
        try:
            page_numbers = json.loads(instruction_pages)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="instruction_pages must be a valid JSON array"
            )

    # Save uploaded PDF
    pdf_dir = settings.manual_dir / manual_id
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / pdf_file.filename

    try:
        with open(pdf_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)
        logger.info(f"Saved uploaded PDF to {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save PDF: {str(e)}"
        )

    # Run ingestion in background
    background_tasks.add_task(
        run_ingestion_pipeline,
        manual_id,
        pdf_path,
        page_numbers
    )

    return {
        "status": "processing",
        "manual_id": manual_id,
        "message": f"Ingestion started for manual {manual_id}"
    }


@router.post("/images")
async def ingest_images(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    image_numbers: str = Form(None),  # JSON array as string, e.g. "[1,2,3]"
    images: List[UploadFile] = File(...),
    assembled_image: Optional[UploadFile] = File(None),
    parts_images: Optional[List[UploadFile]] = File(None),
):
    """
    Upload instruction images and trigger ingestion pipeline.

    Args:
        manual_id: Unique identifier for this manual
        image_numbers: JSON array of image indices to process as instruction
                      pages (1-indexed). If not provided, processes all images.
        images: Instruction page image files
        assembled_image: Optional single image of the final assembled product
        parts_images: Optional images of the parts catalog

    Returns:
        Status message with manual_id
    """
    settings = get_settings()

    # Parse image numbers if provided
    img_numbers: Optional[List[int]] = None
    if image_numbers:
        try:
            img_numbers = json.loads(image_numbers)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="image_numbers must be a valid JSON array"
            )

    base_dir = settings.manual_dir / manual_id
    img_dir = base_dir / "uploaded_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save instruction images
        for i, image_file in enumerate(images):
            img_path = img_dir / f"{i+1:03d}_{image_file.filename}"
            with open(img_path, "wb") as f:
                f.write(await image_file.read())
        logger.info(f"Saved {len(images)} instruction images to {img_dir}")

        # Save assembled product image
        if assembled_image and assembled_image.filename:
            assembled_dir = base_dir / "assembled"
            assembled_dir.mkdir(parents=True, exist_ok=True)
            assembled_path = assembled_dir / assembled_image.filename
            with open(assembled_path, "wb") as f:
                f.write(await assembled_image.read())
            logger.info(f"Saved assembled image to {assembled_path}")

        # Save parts catalog images
        if parts_images:
            parts_dir = base_dir / "parts_catalog"
            parts_dir.mkdir(parents=True, exist_ok=True)
            for i, parts_file in enumerate(parts_images):
                if parts_file and parts_file.filename:
                    parts_path = parts_dir / f"{i+1:03d}_{parts_file.filename}"
                    with open(parts_path, "wb") as f:
                        f.write(await parts_file.read())
            logger.info(f"Saved {len(parts_images)} parts images to {parts_dir}")

    except Exception as e:
        logger.error(f"Failed to save uploaded images: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save images: {str(e)}"
        )

    # Run ingestion in background (only on instruction images)
    background_tasks.add_task(
        run_ingestion_pipeline_images,
        manual_id,
        img_dir,
        img_numbers
    )

    return {
        "status": "processing",
        "manual_id": manual_id,
        "message": f"Ingestion started for manual {manual_id}"
    }


def run_ingestion_pipeline(
    manual_id: str,
    pdf_path: Path,
    instruction_pages: Optional[List[int]]
):
    """
    Background task to run the ingestion pipeline for PDF input.

    Args:
        manual_id: Manual identifier
        pdf_path: Path to PDF file
        instruction_pages: List of page numbers to process
    """
    try:
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        pipeline.process_manual(manual_id, pdf_path, instruction_pages)
        logger.info(f"✓ Ingestion complete for manual {manual_id}")
    except Exception as e:
        logger.error(f"✗ Ingestion failed for manual {manual_id}: {e}")


def run_ingestion_pipeline_images(
    manual_id: str,
    image_dir: Path,
    image_numbers: Optional[List[int]]
):
    """
    Background task to run the ingestion pipeline for image directory input.

    Args:
        manual_id: Manual identifier
        image_dir: Path to directory with images
        image_numbers: List of image indices to process
    """
    try:
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        pipeline.process_image_directory(manual_id, image_dir, image_numbers)
        logger.info(f"✓ Ingestion complete for manual {manual_id}")
    except Exception as e:
        logger.error(f"✗ Ingestion failed for manual {manual_id}: {e}")


@router.post("/url")
async def ingest_url(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    url: str = Form(...),
    instruction_pages: str = Form(None),  # JSON array as string, e.g., "[13,14,15]"
):
    """
    Provide a URL to a PDF manual and trigger the ingestion pipeline.

    The PDF is downloaded to a temporary directory, processed, then cleaned up.

    Args:
        manual_id: Unique identifier for this manual
        url: Direct URL to the PDF manual
        instruction_pages: JSON array of page numbers to process (1-indexed).
                          If not provided, processes all pages.

    Returns:
        Status message with manual_id

    Example:
        curl -X POST "http://localhost:8000/api/ingest/url" \\
             -F "manual_id=6262059" \\
             -F "url=https://www.lego.com/cdn/product-assets/.../6262059.pdf" \\
             -F "instruction_pages=[13,14,15,16]"
    """
    page_numbers: Optional[List[int]] = None
    if instruction_pages:
        try:
            page_numbers = json.loads(instruction_pages)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="instruction_pages must be a valid JSON array"
            )

    background_tasks.add_task(
        run_ingestion_pipeline_url,
        manual_id,
        url,
        page_numbers
    )

    return {
        "status": "processing",
        "manual_id": manual_id,
        "message": f"Ingestion started for manual {manual_id} from URL"
    }


def run_ingestion_pipeline_url(
    manual_id: str,
    url: str,
    instruction_pages: Optional[List[int]]
):
    """
    Background task to download a PDF from a URL and run the ingestion pipeline.

    Args:
        manual_id: Manual identifier
        url: URL to the PDF manual
        instruction_pages: List of page numbers to process
    """
    try:
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        pipeline.process_url(manual_id, url, instruction_pages)
        logger.info(f"✓ Ingestion complete for manual {manual_id}")
    except Exception as e:
        logger.error(f"✗ Ingestion failed for manual {manual_id}: {e}")
