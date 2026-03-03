"""
Video Enhancement API Routes: Endpoints for video-enhanced assembly instructions.

Provides endpoints to trigger video enhancement, retrieve enhanced steps, and list enhancements.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Form, File, UploadFile
from typing import Dict, List, Any
from loguru import logger

from backend.services.data_service import DataService
from backend.services.video_enhancer import VideoEnhancer
from ingestion.vlm_extractor import VLMExtractor
from config.settings import Settings


router = APIRouter(prefix="/api/video", tags=["video-enhancement"])


def get_data_service() -> DataService:
    """Dependency for data service."""
    return DataService()


def get_video_enhancer() -> VideoEnhancer:
    """Dependency for video enhancer."""
    settings = Settings()
    vlm_extractor = VLMExtractor(
        model=settings.vlm_model,
        api_key=settings.openai_api_key
    )
    data_service = DataService()
    return VideoEnhancer(vlm_extractor, data_service, settings)


async def process_video_enhancement(
    manual_id: str,
    video_id: str,
    enhancer: VideoEnhancer,
    data_service: DataService
):
    """
    Background task to process video enhancement.

    Args:
        manual_id: Manual identifier
        video_id: Video identifier
        enhancer: VideoEnhancer instance
        data_service: DataService instance
    """
    try:
        logger.info(f"Starting background enhancement for {manual_id}/{video_id}")

        # Run enhancement
        enhanced_data = await enhancer.enhance_manual_with_video(manual_id, video_id)

        # Save results
        data_service.save_video_enhanced_steps(manual_id, enhanced_data)

        logger.info(f"Enhancement complete for {manual_id}/{video_id}")

    except Exception as e:
        logger.error(f"Enhancement failed for {manual_id}/{video_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


@router.post("/upload-and-enhance")
async def upload_and_enhance_video(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    video_file: UploadFile = File(...),
    data_service: DataService = Depends(get_data_service),
    enhancer: VideoEnhancer = Depends(get_video_enhancer)
) -> Dict[str, Any]:
    """
    Upload video and directly start enhancement (bypasses video_analyzer).

    This endpoint is specifically for video enhancement flow - it does NOT trigger
    the video_analyzer (which has 1000 frame limit). Instead, it directly uploads
    the video and starts enhancement processing on the ENTIRE video.

    Use this endpoint for: Enhancement workflow (processes whole video)
    Use /api/video/upload for: Video verification/analysis workflow (1000 frame limit)

    Args:
        background_tasks: FastAPI background tasks
        manual_id: Manual identifier
        video_file: Video file to upload
        data_service: Data service dependency
        enhancer: Video enhancer dependency

    Returns:
        {
            "video_id": str,
            "status": "processing",
            "message": str
        }

    Raises:
        HTTPException 404: If manual not found
        HTTPException 500: If upload fails
    """
    import uuid
    from pathlib import Path
    from config.settings import Settings

    settings = Settings()

    # Verify manual exists
    try:
        data_service.get_steps(manual_id)
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail=f"Manual '{manual_id}' not found. Please ingest the manual first."
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())[:8]

    # Create video directory
    video_dir = settings.data_dir / "videos" / manual_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = video_dir / f"{video_id}.mp4"

    try:
        content = await video_file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved video to {video_path} ({len(content) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to save video: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save video: {str(e)}"
        )

    # Start background enhancement (NOT video_analyzer)
    background_tasks.add_task(
        process_video_enhancement,
        manual_id,
        video_id,
        enhancer,
        data_service
    )

    logger.info(f"Started direct video enhancement for {manual_id}/{video_id}")

    return {
        "video_id": video_id,
        "status": "processing",
        "message": f"Video uploaded successfully. Enhancement started (processing entire video)."
    }


@router.post("/enhance/{manual_id}/{video_id}")
async def enhance_manual_with_video(
    background_tasks: BackgroundTasks,
    manual_id: str,
    video_id: str,
    data_service: DataService = Depends(get_data_service),
    enhancer: VideoEnhancer = Depends(get_video_enhancer)
) -> Dict[str, Any]:
    """
    Trigger video enhancement process for already-uploaded video.

    This endpoint can be used if video was already uploaded via /api/video/upload.
    For direct enhancement workflow, use /api/video/upload-and-enhance instead.

    Args:
        background_tasks: FastAPI background tasks
        manual_id: Manual identifier
        video_id: Video identifier
        data_service: Data service dependency
        enhancer: Video enhancer dependency

    Returns:
        {
            "status": "processing",
            "message": "Video enhancement started for video {video_id}"
        }

    Raises:
        HTTPException 404: If manual or video not found
    """
    # Verify manual exists
    try:
        data_service.get_steps(manual_id)
    except HTTPException as e:
        logger.warning(f"Manual {manual_id} not found")
        raise

    # Verify video file exists
    from config.settings import Settings
    settings = Settings()
    video_path = settings.data_dir / "videos" / manual_id / f"{video_id}.mp4"

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found for video '{video_id}' in manual '{manual_id}'. "
                   f"Please upload the video first."
        )

    # Start background enhancement
    background_tasks.add_task(
        process_video_enhancement,
        manual_id,
        video_id,
        enhancer,
        data_service
    )

    logger.info(f"Started video enhancement task for {manual_id}/{video_id}")

    return {
        "status": "processing",
        "message": f"Video enhancement started for video {video_id}. "
                   f"This may take several minutes depending on video length."
    }


@router.get("/manuals/{manual_id}/video-enhanced")
async def get_video_enhanced_steps(
    manual_id: str,
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """
    Get video-enhanced steps for a manual.

    Returns the complete video_enhanced.json with hierarchical sub-steps,
    spatial descriptions, and error corrections.

    Args:
        manual_id: Manual identifier
        data_service: Data service dependency

    Returns:
        Complete video_enhanced.json structure:
        {
            "manual_id": str,
            "source_video_id": str,
            "created_at": str,
            "video_metadata": {...},
            "steps": [...],
            "manual_step_mapping": {...}
        }

    Raises:
        HTTPException 404: If no video enhancement found for this manual
    """
    try:
        enhanced_steps = data_service.get_video_enhanced_steps(manual_id)
        logger.info(f"Retrieved video-enhanced steps for manual {manual_id}")
        return enhanced_steps
    except HTTPException as e:
        logger.warning(f"Video-enhanced steps not found for {manual_id}")
        raise


@router.get("/manuals/{manual_id}/video-enhanced/list")
async def list_video_enhancements(
    manual_id: str,
    data_service: DataService = Depends(get_data_service)
) -> List[Dict[str, Any]]:
    """
    List all video enhancements for a manual.

    Currently supports one enhancement per manual. Future: could support
    multiple enhancements from different videos.

    Args:
        manual_id: Manual identifier
        data_service: Data service dependency

    Returns:
        List of enhancement metadata:
        [
            {
                "video_id": str,
                "created_at": str,
                "sub_steps_count": int,
                "corrections_count": int
            }
        ]

    Returns empty list if no enhancements exist.
    """
    enhancements = data_service.list_video_enhancements(manual_id)
    logger.info(f"Listed {len(enhancements)} video enhancements for manual {manual_id}")
    return enhancements
