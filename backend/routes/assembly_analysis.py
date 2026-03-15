"""
Assembly analysis routes for consensus workflow dashboard.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from backend.services.assembly_analysis_service import AssemblyAnalysisService


router = APIRouter()
analysis_service = AssemblyAnalysisService()


@router.get("/assembly/items")
async def list_items():
    """List analysis items discovered from root `data/` folder."""
    return {"items": analysis_service.list_items()}


@router.post("/assembly/analyze")
async def analyze_video(
    item_id: str = Form(...),
    video_file: UploadFile = File(...),
    details_json_file: UploadFile | None = File(default=None),
):
    """Upload video + required consensus details JSON and run consensus analysis."""
    suffix = Path(video_file.filename or "upload.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        content = await video_file.read()
        tmp.write(content)

    if details_json_file is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Consensus details JSON must be provided as `details_json_file`."
            ),
        )

    uploaded_details_json = (
        details_json_file.filename or "uploaded_details.json",
        await details_json_file.read(),
    )

    try:
        result = analysis_service.run_analysis(
            item_id=item_id,
            uploaded_video_path=temp_path,
            original_filename=video_file.filename or f"upload{suffix}",
            details_json_file=uploaded_details_json,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Analysis failed for item '{item_id}': {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze video: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.get("/assembly/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get full analysis payload by id."""
    return analysis_service.get_analysis(analysis_id)


@router.get("/assembly/analysis/{analysis_id}/second/{second}")
async def get_analysis_second(analysis_id: str, second: int):
    """Get synced details for a specific second (carry-forward semantics)."""
    if second < 0:
        raise HTTPException(status_code=400, detail="second must be >= 0")
    return analysis_service.get_synced_second(analysis_id, second)


@router.get("/assembly/asset")
async def get_item_asset(path: str):
    """Safely serve item-related assets (anchors/manual pages) by path."""
    resolved = analysis_service.resolve_asset_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")
    return FileResponse(path=str(resolved))
