"""
Data schemas for LEGO Assembly ingestion pipeline.
Defines Pydantic models for structured data validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class BoundingBox(BaseModel):
    """Bounding box in pixels (x, y, width, height)."""

    x: int = Field(description="Top-left x coordinate in pixels")
    y: int = Field(description="Top-left y coordinate in pixels")
    width: int = Field(description="Width in pixels")
    height: int = Field(description="Height in pixels")

    class Config:
        json_schema_extra = {
            "example": {
                "x": 100,
                "y": 150,
                "width": 80,
                "height": 60
            }
        }


class PartInfo(BaseModel):
    """Single-line part description with bounding box."""

    description: str = Field(
        description="Single-line part description (e.g., 'red 2x4 brick')"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box coordinates for this part in the source image"
    )
    cropped_image_path: Optional[str] = Field(
        default=None,
        description="File path to the cropped image of this part"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "description": "red 2x4 brick",
                "bounding_box": {"x": 100, "y": 150, "width": 80, "height": 60},
                "cropped_image_path": "data/cropped/6262059/parts/step_1_part_0.png"
            }
        }


class SubassemblyInfo(BaseModel):
    """Subassembly description with bounding box."""

    description: str = Field(
        description="Description of the subassembly"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box coordinates for this subassembly in the source image"
    )
    cropped_image_path: Optional[str] = Field(
        default=None,
        description="File path to the cropped image of this subassembly"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "description": "base assembly with wheels",
                "bounding_box": {"x": 200, "y": 300, "width": 150, "height": 120},
                "cropped_image_path": "data/cropped/6262059/subassemblies/step_1_subassembly_0.png"
            }
        }


class Step(BaseModel):
    """Single assembly step with parts, subassemblies, and actions."""

    step_number: int = Field(description="Step number in the assembly sequence")
    parts_required: List[PartInfo] = Field(
        default_factory=list,
        description="List of parts required for this step"
    )
    subassemblies: List[SubassemblyInfo] = Field(
        default_factory=list,
        description="List of subassemblies shown in this step"
    )
    actions: List[str] = Field(
        default_factory=list,
        description="List of assembly actions to perform"
    )
    source_page_path: str = Field(
        description="Path to the source instruction page image"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes or special instructions for this step"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "parts_required": [
                    {
                        "description": "red 2x4 brick",
                        "bounding_box": {"x": 100, "y": 150, "width": 80, "height": 60}
                    }
                ],
                "subassemblies": [
                    {
                        "description": "base assembly",
                        "bounding_box": {"x": 200, "y": 300, "width": 150, "height": 120}
                    }
                ],
                "actions": [
                    "Attach red brick to base",
                    "Press firmly until it clicks"
                ],
                "source_page_path": "data/manuals/6262059/page_013.png",
                "notes": "Ensure alignment before pressing"
            }
        }


class ManualExtraction(BaseModel):
    """Complete manual extraction with all steps."""

    manual_id: str = Field(description="Unique identifier for this manual")
    steps: List[Step] = Field(
        default_factory=list,
        description="List of all assembly steps extracted from the manual"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "manual_id": "6262059",
                "steps": [
                    {
                        "step_number": 1,
                        "parts_required": [],
                        "subassemblies": [],
                        "actions": ["Place base piece"],
                        "source_page_path": "data/manuals/6262059/page_013.png"
                    }
                ]
            }
        }
