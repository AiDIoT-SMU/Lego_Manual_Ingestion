"""
CAD Ground Truth Database.
Loads and serves CAD point clouds as ground truth for validation.
"""

import open3d as o3d
from pathlib import Path
from typing import Dict


class CADGroundTruth:
    """Load and serve CAD point clouds as ground truth."""

    def __init__(self, manual_id: str, base_path: Path = None):
        """
        Initialize CAD database.

        Args:
            manual_id: Manual/assembly identifier (e.g., "123456")
            base_path: Optional custom base path (default: data/processed/{manual_id}/point_clouds)
        """
        self.manual_id = manual_id

        if base_path is None:
            project_root = Path(__file__).parent.parent
            base_path = project_root / "data" / "processed" / manual_id / "point_clouds"

        self.base_path = base_path
        self.step_clouds: Dict[int, o3d.geometry.PointCloud] = {}

        if not self.base_path.exists():
            raise ValueError(f"Point cloud directory not found: {self.base_path}")

    def load_all_steps(self):
        """Load all point cloud steps from the directory."""
        pcd_files = sorted(self.base_path.glob("*.pcd"))

        if not pcd_files:
            raise ValueError(f"No .pcd files found in {self.base_path}")

        for pcd_file in pcd_files:
            # Extract step number from filename (e.g., sg50_step1.pcd → 1)
            try:
                step_num = self._extract_step_number(pcd_file.stem)
                pcd = o3d.io.read_point_cloud(str(pcd_file))
                self.step_clouds[step_num] = pcd
                print(f"Loaded step {step_num}: {len(pcd.points)} points from {pcd_file.name}")
            except Exception as e:
                print(f"Warning: Failed to load {pcd_file.name}: {e}")
                continue

        if not self.step_clouds:
            raise ValueError("No point clouds could be loaded")

        print(f"\nLoaded {len(self.step_clouds)} assembly steps")

    def get_step_cloud(self, step_number: int) -> o3d.geometry.PointCloud:
        """
        Get ground truth point cloud for a specific step.

        Args:
            step_number: Step number (1-indexed)

        Returns:
            Open3D PointCloud object
        """
        if step_number not in self.step_clouds:
            raise ValueError(
                f"Step {step_number} not loaded. Available steps: {sorted(self.step_clouds.keys())}"
            )
        return self.step_clouds[step_number]

    def get_all_steps(self) -> Dict[int, o3d.geometry.PointCloud]:
        """Get all loaded step point clouds."""
        return self.step_clouds

    def _extract_step_number(self, filename: str) -> int:
        """Extract step number from filename (e.g., 'sg50_step1' → 1)."""
        # Try to find 'step' followed by a number
        import re
        match = re.search(r'step(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Fallback: try to find any number in filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))

        raise ValueError(f"Could not extract step number from filename: {filename}")
