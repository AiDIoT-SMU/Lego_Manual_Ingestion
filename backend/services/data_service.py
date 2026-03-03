"""
Data Service: Handles JSON file operations for manual data.
Provides methods to load, query, and aggregate step and part information.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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

    def _parse_part_description(self, raw_desc: str) -> Tuple[str, int]:
        """
        Parse part description to extract clean name and quantity.

        Examples:
            "red brick 2x4 (3x)" → ("red brick 2x4", 3)
            "tan slope 1x2 (1x)" → ("tan slope 1x2", 1)
            "blue plate" → ("blue plate", 1)

        Args:
            raw_desc: Raw part description from extraction

        Returns:
            Tuple of (clean_description, quantity)
        """
        match = re.search(r'^(.+?)\s*\((\d+)x\)$', raw_desc.strip())
        if match:
            clean_desc = match.group(1).strip()
            quantity = int(match.group(2))
            return (clean_desc, quantity)
        return (raw_desc.strip(), 1)

    def get_parts_catalog(self, manual_id: str) -> Dict[str, Any]:
        """
        Get a catalog of all unique parts across all steps with aggregated quantities.

        Args:
            manual_id: Manual identifier

        Returns:
            Dictionary with parts catalog:
            {
                "manual_id": str,
                "total_unique_parts": int,
                "parts": [
                    {
                        "description": str,  # Clean description without "(1x)" suffix
                        "images": [str],  # List of cropped image paths
                        "used_in_steps": [int],  # List of step numbers
                        "total_quantity": int  # Total count across all steps
                    }
                ]
            }

        Raises:
            HTTPException: If manual not found
        """
        data = self.get_steps(manual_id)

        # Aggregate parts by clean description
        parts_map: Dict[str, Dict[str, Any]] = {}

        for step in data.get("steps", []):
            step_num = step.get("step_number")

            for part in step.get("parts_required", []):
                raw_desc = part.get("description", "")
                if not raw_desc:
                    continue

                # Parse description and quantity
                clean_desc, quantity = self._parse_part_description(raw_desc)

                # Initialize part entry if not exists
                if clean_desc not in parts_map:
                    parts_map[clean_desc] = {
                        "description": clean_desc,
                        "images": [],
                        "used_in_steps": [],
                        "total_quantity": 0
                    }

                # Add quantity
                parts_map[clean_desc]["total_quantity"] += quantity

                # Add cropped image if available
                cropped_path = part.get("cropped_image_path")
                if cropped_path and cropped_path not in parts_map[clean_desc]["images"]:
                    parts_map[clean_desc]["images"].append(cropped_path)

                # Add step number if not already added
                if step_num not in parts_map[clean_desc]["used_in_steps"]:
                    parts_map[clean_desc]["used_in_steps"].append(step_num)

        # Convert to list and sort by first appearance
        parts_list = list(parts_map.values())
        parts_list.sort(key=lambda p: p["used_in_steps"][0] if p["used_in_steps"] else 999)

        return {
            "manual_id": manual_id,
            "total_unique_parts": len(parts_list),
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

    def get_digital_twin(self, manual_id: str) -> Dict[str, Any]:
        """
        Get digital twin data for all steps of a manual.

        Args:
            manual_id: Manual identifier

        Returns:
            Dictionary with digital twin data:
            {
                "manual_id": str,
                "steps": [
                    {
                        "step_number": int,
                        "step_name": str,
                        "num_bricks": int,
                        "bricks": [...]
                    }
                ]
            }

        Raises:
            HTTPException: If manual not found or digital twin data doesn't exist
        """
        digital_twin_dir = self.settings.processed_dir / manual_id / "digital_twin"

        if not digital_twin_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Digital twin data not found for manual '{manual_id}'"
            )

        # Load all step files
        steps_data = []
        step_files = sorted(digital_twin_dir.glob("step*.json"))

        if not step_files:
            raise HTTPException(
                status_code=404,
                detail=f"No digital twin step files found for manual '{manual_id}'"
            )

        for step_file in step_files:
            try:
                with open(step_file, 'r') as f:
                    step_data = json.load(f)
                    steps_data.append(step_data)
            except Exception as e:
                logger.error(f"Failed to load digital twin step {step_file.name}: {e}")

        # Sort by step number
        steps_data.sort(key=lambda x: x.get("step_number", 999))

        return {
            "manual_id": manual_id,
            "steps": steps_data
        }

    def save_video_analysis(
        self,
        manual_id: str,
        video_id: str,
        analysis: Dict[str, Any]
    ) -> None:
        """
        Save video analysis results to JSON file.

        Args:
            manual_id: Manual identifier
            video_id: Video identifier
            analysis: Complete video analysis results

        Raises:
            HTTPException: If save fails
        """
        video_dir = self.settings.processed_dir / manual_id / "video_analysis"
        video_dir.mkdir(parents=True, exist_ok=True)

        output_path = video_dir / f"{video_id}_analysis.json"

        try:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved video analysis to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save video analysis: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save video analysis: {str(e)}"
            )

    def get_video_analysis(
        self,
        manual_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """
        Load video analysis results.

        Args:
            manual_id: Manual identifier
            video_id: Video identifier

        Returns:
            Complete video analysis results

        Raises:
            HTTPException: If analysis not found
        """
        analysis_path = (
            self.settings.processed_dir /
            manual_id /
            "video_analysis" /
            f"{video_id}_analysis.json"
        )

        if not analysis_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video analysis not found for video '{video_id}' in manual '{manual_id}'"
            )

        try:
            with open(analysis_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load video analysis: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load video analysis: {str(e)}"
            )

    def list_video_analyses(self, manual_id: str) -> List[Dict[str, Any]]:
        """
        List all video analyses for a manual.

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
        video_dir = self.settings.processed_dir / manual_id / "video_analysis"

        if not video_dir.exists():
            return []

        analyses = []

        for json_file in video_dir.glob("*_analysis.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Skip error states
                if data.get("status") == "failed":
                    continue

                analyses.append({
                    "video_id": data.get("video_id", ""),
                    "filename": data.get("video_filename", ""),
                    "duration_seconds": data.get("total_duration_seconds", 0),
                    "processed_at": data.get("processed_at", "")
                })
            except Exception as e:
                logger.error(f"Failed to read video analysis {json_file.name}: {e}")

        # Sort by processed_at descending (most recent first)
        analyses.sort(
            key=lambda x: x.get("processed_at", ""),
            reverse=True
        )

        return analyses
