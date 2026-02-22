"""
Data Service: Handles JSON file operations for manual data.
Provides methods to load, query, and aggregate step and part information.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from fastapi import HTTPException

from config.settings import Settings


class DataService:
    """Service for accessing processed manual data from JSON files."""

    def __init__(self):
        """Initialize data service with settings."""
        self.settings = Settings()

    def list_manuals(self) -> List[Dict[str, Any]]:
        """
        List all processed manuals.

        Returns:
            List of manual metadata dictionaries
        """
        manuals = []
        processed_dir = self.settings.processed_dir

        if not processed_dir.exists():
            return manuals

        for manual_dir in processed_dir.iterdir():
            if manual_dir.is_dir():
                enhanced_file = manual_dir / "enhanced.json"
                if enhanced_file.exists():
                    try:
                        with open(enhanced_file, 'r') as f:
                            data = json.load(f)

                        manuals.append({
                            "id": manual_dir.name,
                            "step_count": len(data.get("steps", [])),
                            "created_at": enhanced_file.stat().st_mtime
                        })
                    except Exception as e:
                        logger.error(f"Failed to read manual {manual_dir.name}: {e}")

        return sorted(manuals, key=lambda x: x["created_at"], reverse=True)

    def get_steps(self, manual_id: str) -> Dict[str, Any]:
        """
        Get all steps for a manual.

        Args:
            manual_id: Manual identifier

        Returns:
            Dictionary with manual data including all steps

        Raises:
            HTTPException: If manual not found
        """
        json_path = self.settings.processed_dir / manual_id / "enhanced.json"

        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Manual '{manual_id}' not found"
            )

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load steps for {manual_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load manual data: {str(e)}"
            )

    def get_step(self, manual_id: str, step_number: int) -> Dict[str, Any]:
        """
        Get a specific step from a manual.

        Args:
            manual_id: Manual identifier
            step_number: Step number to retrieve

        Returns:
            Step dictionary

        Raises:
            HTTPException: If manual or step not found
        """
        data = self.get_steps(manual_id)

        for step in data.get("steps", []):
            if step.get("step_number") == step_number:
                return step

        raise HTTPException(
            status_code=404,
            detail=f"Step {step_number} not found in manual '{manual_id}'"
        )

    def get_parts_catalog(self, manual_id: str) -> Dict[str, Any]:
        """
        Get a catalog of all unique parts across all steps.

        Args:
            manual_id: Manual identifier

        Returns:
            Dictionary with parts catalog:
            {
                "manual_id": str,
                "parts": [
                    {
                        "description": str,
                        "images": [str],  # List of cropped image paths
                        "used_in_steps": [int]  # List of step numbers
                    }
                ]
            }

        Raises:
            HTTPException: If manual not found
        """
        data = self.get_steps(manual_id)

        # Aggregate parts by description
        parts_map: Dict[str, Dict[str, Any]] = {}

        for step in data.get("steps", []):
            step_num = step.get("step_number")

            for part in step.get("parts_required", []):
                desc = part.get("description", "")
                if not desc:
                    continue

                # Initialize part entry if not exists
                if desc not in parts_map:
                    parts_map[desc] = {
                        "description": desc,
                        "images": [],
                        "used_in_steps": []
                    }

                # Add cropped image if available
                cropped_path = part.get("cropped_image_path")
                if cropped_path and cropped_path not in parts_map[desc]["images"]:
                    parts_map[desc]["images"].append(cropped_path)

                # Add step number if not already added
                if step_num not in parts_map[desc]["used_in_steps"]:
                    parts_map[desc]["used_in_steps"].append(step_num)

        # Convert to list and sort by first appearance
        parts_list = list(parts_map.values())
        parts_list.sort(key=lambda p: p["used_in_steps"][0] if p["used_in_steps"] else 999)

        return {
            "manual_id": manual_id,
            "parts": parts_list
        }

    def get_subassemblies(self, manual_id: str) -> Dict[str, Any]:
        """
        Get all subassemblies across all steps.

        Args:
            manual_id: Manual identifier

        Returns:
            Dictionary with subassemblies:
            {
                "manual_id": str,
                "subassemblies": [
                    {
                        "description": str,
                        "images": [str],
                        "from_step": int
                    }
                ]
            }
        """
        data = self.get_steps(manual_id)

        subassemblies = []

        for step in data.get("steps", []):
            step_num = step.get("step_number")

            for sub in step.get("subassemblies", []):
                subassemblies.append({
                    "description": sub.get("description", ""),
                    "images": [sub.get("cropped_image_path")] if sub.get("cropped_image_path") else [],
                    "from_step": step_num
                })

        return {
            "manual_id": manual_id,
            "subassemblies": subassemblies
        }
