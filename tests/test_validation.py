"""
Tests for Validation Module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.cad_database import CADGroundTruth
from validation.pose_matcher import PoseEstimator


PROJECT_ROOT = Path(__file__).parent.parent
POINT_CLOUD_DIR = PROJECT_ROOT / "data" / "processed" / "123456" / "point_clouds"


class TestCADGroundTruth:
    """Test CAD ground truth database."""

    def test_database_initialization(self):
        """Test database can be initialized."""
        # Skip if point clouds haven't been generated yet
        if not POINT_CLOUD_DIR.exists():
            pytest.skip("Point clouds not generated yet. Run process_ldraw_files.py first.")

        db = CADGroundTruth("123456")
        assert db.manual_id == "123456"
        assert db.base_path.exists()

    def test_load_all_steps(self):
        """Test loading all step point clouds."""
        if not POINT_CLOUD_DIR.exists():
            pytest.skip("Point clouds not generated yet.")

        db = CADGroundTruth("123456")
        db.load_all_steps()

        assert len(db.step_clouds) > 0
        print(f"Loaded {len(db.step_clouds)} steps")

    def test_get_step_cloud(self):
        """Test retrieving specific step cloud."""
        if not POINT_CLOUD_DIR.exists():
            pytest.skip("Point clouds not generated yet.")

        db = CADGroundTruth("123456")
        db.load_all_steps()

        # Get first step
        step_nums = sorted(db.step_clouds.keys())
        first_step = step_nums[0]

        pcd = db.get_step_cloud(first_step)
        assert pcd.has_points()
        print(f"Step {first_step}: {len(pcd.points)} points")


class TestPoseEstimator:
    """Test pose estimation."""

    def test_pose_estimator_initialization(self):
        """Test pose estimator initialization."""
        if not POINT_CLOUD_DIR.exists():
            pytest.skip("Point clouds not generated yet.")

        db = CADGroundTruth("123456")
        db.load_all_steps()

        estimator = PoseEstimator(db)
        assert estimator.cad_db == db

    def test_pose_estimation_with_identity_transform(self):
        """Test pose estimation with identical clouds (identity transform)."""
        if not POINT_CLOUD_DIR.exists():
            pytest.skip("Point clouds not generated yet.")

        db = CADGroundTruth("123456")
        db.load_all_steps()

        # Get a step cloud
        step_nums = sorted(db.step_clouds.keys())
        test_step = step_nums[0]

        # Use the same cloud as both source and target (should get identity)
        observed = db.get_step_cloud(test_step)

        estimator = PoseEstimator(db)
        transform = estimator.estimate_pose(observed, test_step)

        print(f"Estimated transformation:\n{transform}")

        # Check it's close to identity
        identity = np.eye(4)
        error = np.linalg.norm(transform - identity)
        print(f"Error from identity: {error:.6f}")

        # Should be very close to identity (within numerical precision)
        assert error < 0.1, "Identity transform should be recovered"

    def test_pose_estimation_with_translation(self):
        """Test pose estimation with known translation."""
        if not POINT_CLOUD_DIR.exists():
            pytest.skip("Point clouds not generated yet.")

        db = CADGroundTruth("123456")
        db.load_all_steps()

        step_nums = sorted(db.step_clouds.keys())
        test_step = step_nums[0]

        # Get ground truth cloud and apply known translation
        ground_truth = db.get_step_cloud(test_step)

        # Apply 10mm translation in X
        translation_matrix = np.array([
            [1, 0, 0, 10],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        observed = ground_truth.clone()
        observed.transform(translation_matrix)

        # Estimate pose
        estimator = PoseEstimator(db)
        estimated_transform = estimator.estimate_pose(observed, test_step)

        print(f"True translation: 10mm in X")
        print(f"Estimated transformation:\n{estimated_transform}")

        # Extract translation from estimated transform
        estimated_translation = estimated_transform[:3, 3]
        print(f"Estimated translation: {estimated_translation}")

        # Check if close to [10, 0, 0]
        expected = np.array([10, 0, 0])
        error = np.linalg.norm(estimated_translation - expected)
        print(f"Translation error: {error:.2f}mm")

        assert error < 5.0, "Should estimate translation within 5mm"


def test_fpfh_computation():
    """Test FPFH feature computation."""
    if not POINT_CLOUD_DIR.exists():
        pytest.skip("Point clouds not generated yet.")

    db = CADGroundTruth("123456")
    db.load_all_steps()

    step_nums = sorted(db.step_clouds.keys())
    pcd = db.get_step_cloud(step_nums[0])

    estimator = PoseEstimator(db)
    fpfh = estimator.compute_fpfh(pcd)

    assert fpfh.num() > 0
    assert fpfh.dimension() == 33  # FPFH features are 33-dimensional
    print(f"FPFH features: {fpfh.num()} points, {fpfh.dimension()} dimensions")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
