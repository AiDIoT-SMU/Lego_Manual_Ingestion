"""
Steps Routes: API endpoints for accessing assembly step data.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from backend.services.data_service import DataService


router = APIRouter()
data_service = DataService()


@router.get("/manuals", response_model=List[Dict[str, Any]])
async def list_manuals():
    """
    List all processed manuals.

    Returns:
        List of manual metadata with id, step_count, and created_at
    """
    return data_service.list_manuals()


@router.get("/manuals/{manual_id}/steps", response_model=Dict[str, Any])
async def get_steps(manual_id: str):
    """
    Get all steps for a specific manual.

    Args:
        manual_id: Unique identifier for the manual

    Returns:
        Complete manual data with all steps, parts, and cropped images

    Raises:
        HTTPException: 404 if manual not found
    """
    return data_service.get_steps(manual_id)


@router.get("/manuals/{manual_id}/steps/{step_number}", response_model=Dict[str, Any])
async def get_step(manual_id: str, step_number: int):
    """
    Get a specific step from a manual.

    Args:
        manual_id: Unique identifier for the manual
        step_number: Step number to retrieve

    Returns:
        Step data with parts, subassemblies, actions, and images

    Raises:
        HTTPException: 404 if manual or step not found
    """
    return data_service.get_step(manual_id, step_number)
