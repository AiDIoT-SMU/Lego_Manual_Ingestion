"""
Parts Routes: API endpoints for accessing parts catalog data.
"""

from fastapi import APIRouter
from typing import Dict, Any

from backend.services.data_service import DataService


router = APIRouter()
data_service = DataService()


@router.get("/manuals/{manual_id}/parts", response_model=Dict[str, Any])
async def get_parts(manual_id: str):
    """
    Get the parts catalog for a manual.

    Returns all unique parts aggregated across all steps, with their
    cropped images and information about which steps they're used in.

    Args:
        manual_id: Unique identifier for the manual

    Returns:
        Dictionary with:
        - manual_id: str
        - parts: List of part dictionaries with description, images, and used_in_steps

    Raises:
        HTTPException: 404 if manual not found
    """
    return data_service.get_parts_catalog(manual_id)


@router.get("/manuals/{manual_id}/subassemblies", response_model=Dict[str, Any])
async def get_subassemblies(manual_id: str):
    """
    Get all subassemblies from a manual.

    Args:
        manual_id: Unique identifier for the manual

    Returns:
        Dictionary with:
        - manual_id: str
        - subassemblies: List of subassembly dictionaries

    Raises:
        HTTPException: 404 if manual not found
    """
    return data_service.get_subassemblies(manual_id)
