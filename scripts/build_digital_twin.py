#!/usr/bin/env python3
"""
EXPERIMENTAL FEATURE

Build digital twin database with per-brick identification.
Maps each brick instance to its library geometry + pose.

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_processing.ldraw_parser import LDrawParser


def build_digital_twin(
    input_dir: Path,
    output_dir: Path,
    ldraw_library: Path,
    brick_library_dir: Path
):
    """
    Build structured digital twin database preserving individual brick information.

    Args:
        input_dir: Directory containing .ldr files
        output_dir: Output directory for digital twin JSON files
        ldraw_library: Path to LDraw library
        brick_library_dir: Path to brick library (from build_brick_library.py)
    """
    parser = LDrawParser(ldraw_library)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load brick library metadata
    library_metadata_path = brick_library_dir / "library_metadata.json"
    if not library_metadata_path.exists():
        print(f"Error: Brick library not found at {brick_library_dir}")
        print("Please run build_brick_library.py first!")
        return

    with open(library_metadata_path, 'r') as f:
        brick_library = json.load(f)

    # Find all .ldr files
    ldr_files = sorted(input_dir.glob("*.ldr"))

    if not ldr_files:
        print(f"No .ldr files found in {input_dir}")
        return

    print(f"Building digital twin from {len(ldr_files)} assembly steps...")
    print(f"Output directory: {output_dir}\n")

    all_steps = []

    # Process each step
    for ldr_file in ldr_files:
        step_name = ldr_file.stem

        # Extract step number (e.g., "sg50_step1" -> 1)
        try:
            step_number = int(step_name.split('step')[-1])
        except:
            step_number = len(all_steps) + 1

        print(f"Processing {step_name} (Step {step_number})...")

        # Parse LDraw file
        parts = parser.parse_ldr_file(ldr_file)
        print(f"  Found {len(parts)} bricks")

        # Build structured representation
        step_data = {
            "step_number": step_number,
            "step_name": step_name,
            "num_bricks": len(parts),
            "bricks": []
        }

        for brick_id, part in enumerate(parts):
            # Link to brick library
            library_key = f"{part.part_id}_{part.color_id}"

            if library_key not in brick_library["parts"]:
                print(f"  Warning: {library_key} not in brick library, skipping...")
                continue

            library_entry = brick_library["parts"][library_key]

            brick_data = {
                # Identification
                "brick_id": brick_id,
                "part_number": part.part_id,
                "color_id": part.color_id,

                # Pose (expected/ground truth)
                "position": part.position.tolist(),
                "rotation_matrix": part.rotation_matrix.tolist(),
                "pose_4x4": get_pose_matrix(part.rotation_matrix, part.position).tolist(),

                # Human-readable rotation
                "rotation_angles_deg": get_rotation_angles(part.rotation_matrix),

                # Link to brick library
                "geometry_reference": {
                    "mesh_file": library_entry["mesh_file"],
                    "library_key": library_key
                }
            }

            step_data["bricks"].append(brick_data)

        # Save step-specific JSON
        step_json_path = output_dir / f"step{step_number}.json"
        with open(step_json_path, 'w') as f:
            json.dump(step_data, f, indent=2)

        print(f"  ✓ Saved: {step_json_path.name}")

        all_steps.append(step_data)

    # Save combined database
    database = {
        "assembly_name": input_dir.name,
        "total_steps": len(all_steps),
        "brick_library_path": str(brick_library_dir),
        "steps": all_steps
    }

    database_path = output_dir / "digital_twin.json"
    with open(database_path, 'w') as f:
        json.dump(database, f, indent=2)

    print(f"\n✓ Digital twin database created: {database_path}")
    print(f"  Total steps: {len(all_steps)}")
    print(f"  Total bricks: {sum(step['num_bricks'] for step in all_steps)}")

    # Print summary
    print("\n=== Digital Twin Summary ===")
    for step in all_steps:
        print(f"Step {step['step_number']}: {step['num_bricks']} bricks")

        # Show unique part types
        part_types = {}
        for brick in step['bricks']:
            part = brick['part_number']
            part_types[part] = part_types.get(part, 0) + 1

        print(f"  Part types: {len(part_types)}")
        for part, count in sorted(part_types.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {part}: {count}x")


def get_pose_matrix(rotation_matrix: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Combine rotation and translation into 4x4 pose matrix."""
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = position
    return pose


def get_rotation_angles(rotation_matrix: np.ndarray) -> dict:
    """
    Extract rotation angles (Euler angles) from rotation matrix.
    Returns angles in degrees for human readability.
    """
    # Convert to numpy if not already
    R = np.array(rotation_matrix)

    # Extract Euler angles (ZYX convention)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return {
        "roll_deg": float(np.degrees(x)),
        "pitch_deg": float(np.degrees(y)),
        "yaw_deg": float(np.degrees(z))
    }


if __name__ == "__main__":
    # Paths
    input_dir = Path(__file__).parent.parent / "data" / "Lego Studio" / "123456"
    output_dir = Path(__file__).parent.parent / "data" / "processed" / "123456" / "digital_twin"
    ldraw_library = Path(__file__).parent.parent / "data" / "ldraw_library" / "ldraw"
    brick_library_dir = Path(__file__).parent.parent / "data" / "brick_library"

    build_digital_twin(input_dir, output_dir, ldraw_library, brick_library_dir)
