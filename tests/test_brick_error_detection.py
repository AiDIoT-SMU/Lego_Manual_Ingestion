"""
Test the complete per-brick error detection pipeline.
Demonstrates Option 2: Individual brick identification.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.brick_error_detector import (
    DigitalTwinLoader,
    BrickErrorDetector,
    BrickObservation
)


# Paths
DIGITAL_TWIN_DIR = Path(__file__).parent.parent / "data" / "processed" / "123456" / "digital_twin"
BRICK_LIBRARY_DIR = Path(__file__).parent.parent / "data" / "brick_library"


class TestBrickErrorDetection:
    """Test per-brick error detection system."""

    def test_digital_twin_loads(self):
        """Test that digital twin database loads correctly."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet. Run build_digital_twin.py first.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)

        assert len(twin.steps) > 0, "No steps loaded"
        print(f"✓ Loaded {len(twin.steps)} steps")

    def test_perfect_assembly(self):
        """Test that perfect assembly (no errors) is detected correctly."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)
        detector = BrickErrorDetector(twin)

        # Get step 1
        step_data = twin.get_step(1)

        # Simulate perfect camera observations (exact match to digital twin)
        observed_bricks = []
        for brick in step_data['bricks']:
            obs = BrickObservation(
                brick_id=brick['brick_id'],
                position=np.array(brick['position']),
                rotation_matrix=np.array(brick['rotation_matrix']),
                confidence=1.0
            )
            observed_bricks.append(obs)

        # Detect errors
        errors = detector.detect_errors(1, observed_bricks)

        assert len(errors) == 0, f"Expected no errors, but found {len(errors)}"
        print("✓ Perfect assembly correctly identified")

    def test_position_error_detection(self):
        """Test that position errors are detected."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)
        detector = BrickErrorDetector(twin)

        # Get step 1
        step_data = twin.get_step(1)

        # Simulate observations with brick 0 misplaced by 30 LDU (1.5 studs)
        observed_bricks = []
        for brick in step_data['bricks']:
            pos = np.array(brick['position'])

            # Misplace brick 0
            if brick['brick_id'] == 0:
                pos = pos + np.array([30.0, 0.0, 0.0])  # Shift 30 LDU in X

            obs = BrickObservation(
                brick_id=brick['brick_id'],
                position=pos,
                rotation_matrix=np.array(brick['rotation_matrix']),
                confidence=1.0
            )
            observed_bricks.append(obs)

        # Detect errors
        errors = detector.detect_errors(1, observed_bricks)

        # Should detect 1 position error
        position_errors = [e for e in errors if e['type'] == 'POSITION_ERROR']
        assert len(position_errors) == 1, f"Expected 1 position error, found {len(position_errors)}"
        assert position_errors[0]['brick_id'] == 0, "Error should be for brick 0"
        assert position_errors[0]['error_magnitude_ldu'] == pytest.approx(30.0, abs=0.1)

        print(f"✓ Position error detected: {position_errors[0]['message']}")

    def test_orientation_error_detection(self):
        """Test that orientation errors are detected."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)
        detector = BrickErrorDetector(twin)

        # Get step 1
        step_data = twin.get_step(1)

        # Simulate observations with brick 1 rotated 90° around Y axis
        observed_bricks = []
        for brick in step_data['bricks']:
            rot = np.array(brick['rotation_matrix'])

            # Rotate brick 1 by 90° around Y
            if brick['brick_id'] == 1:
                # 90° rotation matrix around Y axis
                rot_90_y = np.array([
                    [0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0]
                ])
                rot = rot @ rot_90_y

            obs = BrickObservation(
                brick_id=brick['brick_id'],
                position=np.array(brick['position']),
                rotation_matrix=rot,
                confidence=1.0
            )
            observed_bricks.append(obs)

        # Detect errors
        errors = detector.detect_errors(1, observed_bricks)

        # Should detect 1 orientation error
        orientation_errors = [e for e in errors if e['type'] == 'ORIENTATION_ERROR']
        assert len(orientation_errors) == 1, f"Expected 1 orientation error, found {len(orientation_errors)}"
        assert orientation_errors[0]['brick_id'] == 1, "Error should be for brick 1"

        angle_error = orientation_errors[0]['angle_error_degrees']
        assert angle_error > 80 and angle_error < 100, f"Expected ~90° rotation, got {angle_error:.1f}°"

        print(f"✓ Orientation error detected: {orientation_errors[0]['message']}")

    def test_missing_brick_detection(self):
        """Test that missing bricks are detected."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)
        detector = BrickErrorDetector(twin)

        # Get step 1
        step_data = twin.get_step(1)

        # Simulate observations with brick 2 missing
        observed_bricks = []
        for brick in step_data['bricks']:
            if brick['brick_id'] == 2:
                continue  # Skip brick 2 (missing)

            obs = BrickObservation(
                brick_id=brick['brick_id'],
                position=np.array(brick['position']),
                rotation_matrix=np.array(brick['rotation_matrix']),
                confidence=1.0
            )
            observed_bricks.append(obs)

        # Detect errors
        errors = detector.detect_errors(1, observed_bricks)

        # Should detect 1 missing brick
        missing_errors = [e for e in errors if e['type'] == 'MISSING']
        assert len(missing_errors) == 1, f"Expected 1 missing brick, found {len(missing_errors)}"
        assert missing_errors[0]['brick_id'] == 2, "Error should be for brick 2"

        print(f"✓ Missing brick detected: {missing_errors[0]['message']}")

    def test_extra_brick_detection(self):
        """Test that extra bricks are detected."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)
        detector = BrickErrorDetector(twin)

        # Get step 1
        step_data = twin.get_step(1)

        # Simulate observations with all expected bricks + one extra
        observed_bricks = []
        for brick in step_data['bricks']:
            obs = BrickObservation(
                brick_id=brick['brick_id'],
                position=np.array(brick['position']),
                rotation_matrix=np.array(brick['rotation_matrix']),
                confidence=1.0
            )
            observed_bricks.append(obs)

        # Add an extra brick
        extra_brick = BrickObservation(
            brick_id=999,  # ID not in expected set
            position=np.array([100.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3),
            confidence=0.8
        )
        observed_bricks.append(extra_brick)

        # Detect errors
        errors = detector.detect_errors(1, observed_bricks)

        # Should detect 1 extra brick
        extra_errors = [e for e in errors if e['type'] == 'EXTRA_BRICK']
        assert len(extra_errors) == 1, f"Expected 1 extra brick, found {len(extra_errors)}"
        assert extra_errors[0]['brick_id'] == 999, "Error should be for brick 999"

        print(f"✓ Extra brick detected: {extra_errors[0]['message']}")

    def test_complete_error_report(self):
        """Test complete error report with multiple error types."""
        if not DIGITAL_TWIN_DIR.exists():
            pytest.skip("Digital twin not built yet.")

        twin = DigitalTwinLoader(DIGITAL_TWIN_DIR, BRICK_LIBRARY_DIR)
        detector = BrickErrorDetector(twin)

        # Get step 1
        step_data = twin.get_step(1)

        # Simulate complex scenario
        observed_bricks = []
        for brick in step_data['bricks']:
            # Skip brick 2 (missing)
            if brick['brick_id'] == 2:
                continue

            pos = np.array(brick['position'])
            rot = np.array(brick['rotation_matrix'])

            # Misplace brick 0
            if brick['brick_id'] == 0:
                pos = pos + np.array([25.0, 0.0, 0.0])

            # Misrotate brick 1
            if brick['brick_id'] == 1:
                rot_45 = np.array([
                    [0.707, 0, 0.707],
                    [0, 1, 0],
                    [-0.707, 0, 0.707]
                ])
                rot = rot @ rot_45

            obs = BrickObservation(
                brick_id=brick['brick_id'],
                position=pos,
                rotation_matrix=rot,
                confidence=1.0
            )
            observed_bricks.append(obs)

        # Detect errors
        errors = detector.detect_errors(1, observed_bricks)

        # Print full report
        print("\n" + "=" * 80)
        detector.print_error_report(errors)

        # Should have at least 3 errors: 1 position, 1 orientation, 1 missing
        assert len(errors) >= 3, f"Expected at least 3 errors, found {len(errors)}"


if __name__ == "__main__":
    # Run tests manually
    tester = TestBrickErrorDetection()

    print("Testing per-brick error detection system...\n")

    try:
        tester.test_digital_twin_loads()
        tester.test_perfect_assembly()
        tester.test_position_error_detection()
        tester.test_orientation_error_detection()
        tester.test_missing_brick_detection()
        tester.test_extra_brick_detection()
        tester.test_complete_error_report()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
