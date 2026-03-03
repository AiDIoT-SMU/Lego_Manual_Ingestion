"""
Video Routes: API endpoints for video upload and assembly verification.
"""

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import Dict, Any, List
from pathlib import Path
import uuid
from datetime import datetime
from loguru import logger

from backend.services.data_service import DataService
from backend.services.video_processor import VideoProcessor
from backend.services.video_analyzer import VideoAnalyzer
from config.settings import Settings
from ingestion.vlm_extractor import VLMExtractor


router = APIRouter()
data_service = DataService()
settings = Settings()


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    video_file: UploadFile = File(...)
):
    """
    Upload a video for assembly verification.

    Args:
        manual_id: Manual ID to analyze against
        video_file: Video file (MP4, MOV, AVI)

    Returns:
        {
            "video_id": str,
            "status": "processing",
            "message": str
        }

    Raises:
        HTTPException: If manual not found or upload fails
    """
    # Verify manual exists
    try:
        data_service.get_steps(manual_id)
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail=f"Manual '{manual_id}' not found. Please ingest the manual first."
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())[:8]  # Use short ID for simplicity

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

    # Run analysis in background
    background_tasks.add_task(
        process_video_analysis,
        manual_id,
        video_id,
        video_path,
        video_file.filename or f"video_{video_id}.mp4"
    )

    return {
        "video_id": video_id,
        "status": "processing",
        "message": f"Video uploaded successfully. Analysis started for video {video_id}."
    }


@router.get("/analysis/{manual_id}/{video_id}")
async def get_video_analysis(manual_id: str, video_id: str) -> Dict[str, Any]:
    """
    Get video analysis results or processing status.

    Args:
        manual_id: Manual identifier
        video_id: Video identifier

    Returns:
        Complete analysis JSON with step timeline and parts usage,
        OR status information if still processing

    Raises:
        HTTPException: If analysis not found
    """
    # First check if there's a status file (processing or failed)
    status_file = settings.data_dir / "videos" / manual_id / f"{video_id}_status.json"

    if status_file.exists():
        try:
            import json
            with open(status_file, 'r') as f:
                status_data = json.load(f)

            # If completed, try to load the full analysis
            if status_data.get("status") == "completed":
                try:
                    return data_service.get_video_analysis(manual_id, video_id)
                except HTTPException:
                    # Status says completed but analysis file missing, return status
                    return status_data
            else:
                # Still processing or failed, return status
                return status_data
        except Exception as e:
            logger.error(f"Failed to load status file: {e}")

    # No status file, try to load analysis directly
    return data_service.get_video_analysis(manual_id, video_id)


@router.get("/list/{manual_id}")
async def list_videos(manual_id: str) -> List[Dict[str, Any]]:
    """
    List all analyzed videos for a manual.

    Args:
        manual_id: Manual identifier

    Returns:
        List of video metadata:
        [
            {
                "video_id": str,
                "filename": str,
                "duration_seconds": float,
                "processed_at": str
            }
        ]
    """
    return data_service.list_video_analyses(manual_id)


def _save_processing_status(
    manual_id: str,
    video_id: str,
    status: str,
    message: str = "",
    progress: float = 0.0,
    error: str = None
):
    """Save processing status to a temporary file."""
    settings = Settings()
    status_dir = settings.data_dir / "videos" / manual_id
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f"{video_id}_status.json"

    status_data = {
        "video_id": video_id,
        "manual_id": manual_id,
        "status": status,
        "message": message,
        "progress": progress,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    if error:
        status_data["error"] = error

    try:
        with open(status_file, 'w') as f:
            import json
            json.dump(status_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save status: {e}")


def process_video_analysis(
    manual_id: str,
    video_id: str,
    video_path: Path,
    original_filename: str
):
    """
    Background task to process video analysis.

    Pipeline:
    1. Extract frames (every 50th frame)
    2. Analyze each frame for step detection
    3. Analyze each frame for part detection
    4. Build step timeline and parts usage
    5. Save results to JSON

    Args:
        manual_id: Manual identifier
        video_id: Video identifier
        video_path: Path to uploaded video file
        original_filename: Original filename from upload
    """
    try:
        logger.info(f"Starting video analysis for {video_id} (manual: {manual_id})")
        _save_processing_status(manual_id, video_id, "processing", "Starting video analysis", 0.0)

        # Stage 1: Get video metadata
        _save_processing_status(manual_id, video_id, "processing", "Loading video metadata", 10.0)
        video_processor = VideoProcessor(settings)
        metadata = video_processor.get_video_metadata(video_path)
        logger.info(
            f"Video metadata: {metadata['duration_seconds']}s, "
            f"{metadata['total_frames']} frames, {metadata['fps']:.2f} FPS"
        )

        # Stage 2: Extract frames
        _save_processing_status(manual_id, video_id, "processing", "Extracting video frames", 20.0)
        frames_dir = video_path.parent / f"{video_id}_frames"
        frames_dir.mkdir(exist_ok=True)

        frames = video_processor.extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            frame_interval=30,  # Extract every 30th frame
            max_frames=1000  # Limit for video analysis (quick preview) - enhancement processes full video
        )
        logger.info(f"Extracted {len(frames)} frames for analysis")

        if not frames:
            raise ValueError("No frames extracted from video")

        # Stage 3: Analyze with VLM
        _save_processing_status(
            manual_id, video_id, "processing",
            f"Analyzing {len(frames)} frames with VLM (this may take several minutes)", 30.0
        )

        vlm = VLMExtractor(
            vlm_model=settings.vlm_model,
            api_key=settings.gemini_api_key,
            max_retries=settings.vlm_max_retries,
            spatial_prompt_template=""  # Not needed for video analysis
        )

        analyzer = VideoAnalyzer(vlm, data_service, settings)
        analysis_result = analyzer.analyze_video(
            manual_id=manual_id,
            frames=frames,
            video_id=video_id
        )

        # Add metadata
        _save_processing_status(manual_id, video_id, "processing", "Saving analysis results", 90.0)
        analysis_result["video_filename"] = original_filename
        analysis_result["total_duration_seconds"] = metadata["duration_seconds"]
        analysis_result["processed_at"] = datetime.utcnow().isoformat() + "Z"
        analysis_result["status"] = "completed"

        # Stage 4: Save results
        data_service.save_video_analysis(manual_id, video_id, analysis_result)
        _save_processing_status(manual_id, video_id, "completed", "Analysis complete", 100.0)

        logger.info(
            f"Video analysis complete for {video_id}: "
            f"{len(analysis_result['step_timeline'])} steps detected, "
            f"{len(analysis_result['parts_used'])} parts used"
        )

    except Exception as e:
        logger.error(f"Video analysis failed for {video_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Save status with error
        _save_processing_status(
            manual_id, video_id, "failed",
            f"Processing failed: {str(e)}", 0.0, error=str(e)
        )

        # Save error state
        error_result = {
            "video_id": video_id,
            "manual_id": manual_id,
            "error": str(e),
            "status": "failed",
            "processed_at": datetime.utcnow().isoformat() + "Z"
        }

        try:
            data_service.save_video_analysis(manual_id, video_id, error_result)
        except Exception as save_error:
            logger.error(f"Failed to save error state: {save_error}")
