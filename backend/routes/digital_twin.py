"""
Digital Twin Routes: API endpoints for accessing digital twin data.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from backend.services.data_service import DataService


router = APIRouter()
data_service = DataService()


@router.get("/manuals/{manual_id}/digital-twin", response_model=Dict[str, Any])
async def get_digital_twin(manual_id: str):
    """
    Get digital twin data for all steps of a manual.

    Args:
        manual_id: Unique identifier for the manual

    Returns:
        Digital twin data with all steps and brick information

    Raises:
        HTTPException: 404 if manual or digital twin data not found
    """
    return data_service.get_digital_twin(manual_id)
