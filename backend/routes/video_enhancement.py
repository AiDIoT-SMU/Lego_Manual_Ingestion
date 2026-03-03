"""
Video Enhancement API Routes: Endpoints for video-enhanced assembly instructions.

Provides endpoints to trigger video enhancement, retrieve enhanced steps, and list enhancements.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
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


@router.post("/enhance/{manual_id}/{video_id}")
async def enhance_manual_with_video(
    background_tasks: BackgroundTasks,
    manual_id: str,
    video_id: str,
    data_service: DataService = Depends(get_data_service),
    enhancer: VideoEnhancer = Depends(get_video_enhancer)
) -> Dict[str, Any]:
    """
    Trigger video enhancement process.

    This endpoint starts a background task to analyze the video and generate
    video-enhanced.json with detailed sub-steps and spatial information.

    Prerequisites:
    - Video analysis must be completed (status: "completed")
    - Manual enhanced.json must exist

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
        HTTPException 404: If manual or video analysis not found
        HTTPException 400: If video analysis not completed yet
    """
    # Verify manual exists
    try:
        data_service.get_steps(manual_id)
    except HTTPException as e:
        logger.warning(f"Manual {manual_id} not found")
        raise

    # Verify video analysis exists and is completed
    try:
        video_analysis = data_service.get_video_analysis(manual_id, video_id)
    except HTTPException as e:
        logger.warning(f"Video analysis for {video_id} not found")
        raise HTTPException(
            status_code=404,
            detail=f"Video analysis not found for video '{video_id}' in manual '{manual_id}'. "
                   f"Please upload and analyze the video first."
        )

    # Check video analysis status
    if video_analysis.get("status") == "processing":
        raise HTTPException(
            status_code=400,
            detail=f"Video analysis is still processing. Please wait for it to complete."
        )
    elif video_analysis.get("status") == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Video analysis failed. Cannot enhance manual with failed analysis."
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
