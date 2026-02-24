"""
Per-Brick Error Detection System.
Compares camera observations against digital twin to identify individual brick errors.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BrickObservation:
    """Observed brick from camera."""
    brick_id: int                      # Detected brick ID (matched to expected)
    position: np.ndarray               # Observed position (x, y, z)
    rotation_matrix: np.ndarray        # Observed rotation (3x3)
    point_cloud: o3d.geometry.PointCloud = None  # Optional: brick point cloud
    confidence: float = 1.0            # Detection confidence (0-1)


class DigitalTwinLoader:
    """Load and query digital twin database."""

    def __init__(self, digital_twin_dir: Path, brick_library_dir: Path):
        """
        Initialize digital twin loader.

        Args:
            digital_twin_dir: Path to digital twin JSON files
            brick_library_dir: Path to brick library (meshes/point clouds)
        """
        self.digital_twin_dir = digital_twin_dir
        self.brick_library_dir = brick_library_dir

        # Load digital twin metadata
        twin_path = digital_twin_dir / "digital_twin.json"
        if not twin_path.exists():
            raise FileNotFoundError(f"Digital twin not found: {twin_path}")

        with open(twin_path, 'r') as f:
            self.database = json.load(f)

        self.steps = {step['step_number']: step for step in self.database['steps']}

    def get_step(self, step_number: int) -> dict:
        """Get all bricks for a specific step."""
        if step_number not in self.steps:
            raise ValueError(f"Step {step_number} not found in digital twin")

        return self.steps[step_number]

    def get_brick_geometry(self, brick_data: dict) -> Tuple[o3d.geometry.PointCloud, dict]:
        """
        Load 3D geometry for a brick from library.

        Args:
            brick_data: Brick metadata from digital twin

        Returns:
            Tuple of (point_cloud, mesh_info)
        """
        geometry_ref = brick_data['geometry_reference']
        pcd_file = geometry_ref['point_cloud_file']
        pcd_path = self.brick_library_dir / "point_clouds" / pcd_file

        if not pcd_path.exists():
            raise FileNotFoundError(f"Brick point cloud not found: {pcd_path}")

        pcd = o3d.io.read_point_cloud(str(pcd_path))

        # Apply brick's pose transformation
        pose = np.array(brick_data['pose_4x4'])
        pcd.transform(pose)

        return pcd, geometry_ref


class BrickErrorDetector:
    """
    Detect individual brick placement/orientation errors.
    """

    def __init__(self, digital_twin_loader: DigitalTwinLoader):
        """
        Initialize error detector.

        Args:
            digital_twin_loader: Digital twin database loader
        """
        self.twin = digital_twin_loader

        # Error thresholds (in LDraw units: 20 LDU = 1 stud = 8mm)
        self.position_threshold = 10.0    # 0.5 studs (4mm)
        self.rotation_threshold_deg = 15.0  # 15 degrees

    def detect_errors(
        self,
        step_number: int,
        observed_bricks: List[BrickObservation]
    ) -> List[Dict]:
        """
        Detect errors in observed brick placements for a specific step.

        Args:
            step_number: Current assembly step
            observed_bricks: List of detected bricks from camera

        Returns:
            List of error dictionaries
        """
        errors = []

        # Load expected bricks for this step
        step_data = self.twin.get_step(step_number)
        expected_bricks = step_data['bricks']

        print(f"\n=== Error Detection for Step {step_number} ===")
        print(f"Expected bricks: {len(expected_bricks)}")
        print(f"Observed bricks: {len(observed_bricks)}\n")

        # Check each expected brick
        for expected in expected_bricks:
            brick_id = expected['brick_id']

            # Find corresponding observed brick
            observed = self._find_observed_brick(brick_id, observed_bricks)

            if observed is None:
                errors.append({
                    'brick_id': brick_id,
                    'part_number': expected['part_number'],
                    'type': 'MISSING',
                    'severity': 'HIGH',
                    'message': f"Brick {brick_id} ({expected['part_number']}) not detected",
                    'expected_position': expected['position']
                })
                continue

            # Check position error
            position_error = self._check_position_error(expected, observed)
            if position_error:
                errors.append(position_error)

            # Check orientation error
            orientation_error = self._check_orientation_error(expected, observed)
            if orientation_error:
                errors.append(orientation_error)

        # Check for extra bricks (not in expected set)
        expected_ids = {brick['brick_id'] for brick in expected_bricks}
        for observed in observed_bricks:
            if observed.brick_id not in expected_ids:
                errors.append({
                    'brick_id': observed.brick_id,
                    'type': 'EXTRA_BRICK',
                    'severity': 'HIGH',
                    'message': f"Unexpected brick {observed.brick_id} detected",
                    'observed_position': observed.position.tolist()
                })

        return errors

    def _find_observed_brick(
        self,
        brick_id: int,
        observed_bricks: List[BrickObservation]
    ) -> BrickObservation:
        """Find observed brick matching expected brick ID."""
        for obs in observed_bricks:
            if obs.brick_id == brick_id:
                return obs
        return None

    def _check_position_error(
        self,
        expected: dict,
        observed: BrickObservation
    ) -> Dict:
        """Check if brick position is within tolerance."""
        expected_pos = np.array(expected['position'])
        observed_pos = observed.position

        position_error = np.linalg.norm(observed_pos - expected_pos)

        if position_error > self.position_threshold:
            # Calculate error in studs (20 LDU = 1 stud)
            error_studs = position_error / 20.0

            # Determine severity
            if position_error > 40.0:  # >2 studs
                severity = 'HIGH'
            elif position_error > 20.0:  # >1 stud
                severity = 'MEDIUM'
            else:
                severity = 'LOW'

            return {
                'brick_id': expected['brick_id'],
                'part_number': expected['part_number'],
                'type': 'POSITION_ERROR',
                'severity': severity,
                'message': f"Brick {expected['brick_id']} misplaced by {error_studs:.2f} studs ({position_error:.1f} LDU)",
                'expected_position': expected_pos.tolist(),
                'observed_position': observed_pos.tolist(),
                'error_magnitude_ldu': float(position_error),
                'error_magnitude_studs': float(error_studs),
                'error_vector': (observed_pos - expected_pos).tolist()
            }

        return None

    def _check_orientation_error(
        self,
        expected: dict,
        observed: BrickObservation
    ) -> Dict:
        """Check if brick orientation is within tolerance."""
        expected_rot = np.array(expected['rotation_matrix'])
        observed_rot = observed.rotation_matrix

        # Compute rotation difference using axis-angle representation
        R_diff = expected_rot.T @ observed_rot

        # Extract angle from rotation matrix
        # trace(R) = 1 + 2*cos(theta)
        trace = np.trace(R_diff)
        cos_theta = (trace - 1.0) / 2.0

        # Clamp to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        if angle_deg > self.rotation_threshold_deg:
            # Determine severity
            if angle_deg > 45.0:
                severity = 'HIGH'
            elif angle_deg > 30.0:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'

            return {
                'brick_id': expected['brick_id'],
                'part_number': expected['part_number'],
                'type': 'ORIENTATION_ERROR',
                'severity': severity,
                'message': f"Brick {expected['brick_id']} rotated by {angle_deg:.1f}°",
                'expected_rotation': expected_rot.tolist(),
                'observed_rotation': observed_rot.tolist(),
                'angle_error_degrees': float(angle_deg)
            }

        return None

    def print_error_report(self, errors: List[Dict]):
        """Print human-readable error report."""
        if not errors:
            print("✅ No errors detected! Assembly is correct.\n")
            return

        print(f"\n⚠️  Found {len(errors)} error(s):\n")
        print("=" * 80)

        # Group by severity
        high = [e for e in errors if e.get('severity') == 'HIGH']
        medium = [e for e in errors if e.get('severity') == 'MEDIUM']
        low = [e for e in errors if e.get('severity') == 'LOW']

        if high:
            print(f"\n🔴 HIGH SEVERITY ({len(high)} errors):")
            for i, error in enumerate(high, 1):
                self._print_error(i, error)

        if medium:
            print(f"\n🟡 MEDIUM SEVERITY ({len(medium)} errors):")
            for i, error in enumerate(medium, 1):
                self._print_error(i, error)

        if low:
            print(f"\n🟢 LOW SEVERITY ({len(low)} errors):")
            for i, error in enumerate(low, 1):
                self._print_error(i, error)

        print("\n" + "=" * 80 + "\n")

    def _print_error(self, index: int, error: Dict):
        """Print a single error."""
        print(f"{index}. [{error['type']}] {error['message']}")

        if error['type'] == 'POSITION_ERROR':
            print(f"   Expected: {error['expected_position']}")
            print(f"   Observed: {error['observed_position']}")
            print(f"   Error: {error['error_magnitude_studs']:.2f} studs\n")

        elif error['type'] == 'ORIENTATION_ERROR':
            print(f"   Rotation off by: {error['angle_error_degrees']:.1f}°\n")

        elif error['type'] == 'MISSING':
            print(f"   Expected at: {error['expected_position']}\n")

        elif error['type'] == 'EXTRA_BRICK':
            print(f"   Found at: {error['observed_position']}\n")
