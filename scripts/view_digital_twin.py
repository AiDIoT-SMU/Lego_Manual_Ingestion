#!/usr/bin/env python3
"""
EXPERIMENTAL FEATURE

Visualize the digital twin assembly - similar to LEGO Studio.
Loads meshes for each brick and displays them with proper positioning.

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
import sys


def load_digital_twin_step(step_number: int, digital_twin_dir: Path, brick_library_dir: Path):
    """
    Load and visualize a specific assembly step.

    Args:
        step_number: Step number to visualize (1-7)
        digital_twin_dir: Path to digital twin JSON files
        brick_library_dir: Path to brick library with meshes
    """
    # Load step data
    step_file = digital_twin_dir / f"step{step_number}.json"
    if not step_file.exists():
        print(f"Error: Step file not found: {step_file}")
        return None

    with open(step_file, 'r') as f:
        step_data = json.load(f)

    print(f"Loading Step {step_number}: {step_data['step_name']}")
    print(f"  Bricks: {step_data['num_bricks']}")

    # Create list to hold all meshes
    meshes = []
    mesh_dir = brick_library_dir / "meshes"

    # Load each brick
    for brick in step_data['bricks']:
        brick_id = brick['brick_id']
        mesh_file = mesh_dir / brick['geometry_reference']['mesh_file']

        if not mesh_file.exists():
            print(f"  Warning: Mesh file not found: {mesh_file.name}")
            continue

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))

        if len(mesh.vertices) == 0:
            print(f"  Warning: Empty mesh for brick {brick_id}")
            continue

        # Apply pose transformation
        pose_matrix = np.array(brick['pose_4x4'])
        mesh.transform(pose_matrix)

        # Compute normals for better rendering
        mesh.compute_vertex_normals()

        meshes.append(mesh)

        if (brick_id + 1) % 10 == 0:
            print(f"  Loaded {brick_id + 1}/{step_data['num_bricks']} bricks...")

    print(f"  ✓ Loaded {len(meshes)} bricks successfully")

    return meshes, step_data


def visualize_digital_twin(step_number: int = 1):
    """
    Main visualization function.

    Args:
        step_number: Assembly step to visualize (default: 1)
    """
    # Setup paths
    base_dir = Path(__file__).parent.parent
    digital_twin_dir = base_dir / "data" / "processed" / "123456" / "digital_twin"
    brick_library_dir = base_dir / "data" / "brick_library"

    # Verify paths exist
    if not digital_twin_dir.exists():
        print(f"Error: Digital twin directory not found: {digital_twin_dir}")
        print("Please run build_digital_twin.py first!")
        return

    if not brick_library_dir.exists():
        print(f"Error: Brick library not found: {brick_library_dir}")
        print("Please run build_brick_library.py first!")
        return

    # Load the assembly step
    result = load_digital_twin_step(step_number, digital_twin_dir, brick_library_dir)
    if result is None:
        return

    meshes, step_data = result

    if not meshes:
        print("No meshes to display!")
        return

    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=50.0, origin=[0, 0, 0]
    )

    # Setup visualization
    print("\n" + "="*60)
    print(f"LEGO Assembly Visualization - Step {step_number}")
    print("="*60)
    print("\nControls:")
    print("  - Mouse drag: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Ctrl + Mouse: Pan")
    print("  - Q or ESC: Close window")
    print("\nNote: X-axis=Red, Y-axis=Green, Z-axis=Blue")
    print("="*60 + "\n")

    # Visualize all meshes together
    o3d.visualization.draw_geometries(
        meshes + [coord_frame],
        window_name=f"Digital Twin - Step {step_number} ({step_data['num_bricks']} bricks)",
        width=1600,
        height=900,
        left=50,
        top=50,
        mesh_show_back_face=True
    )


def visualize_all_steps():
    """Visualize all assembly steps sequentially."""
    base_dir = Path(__file__).parent.parent
    digital_twin_dir = base_dir / "data" / "processed" / "123456" / "digital_twin"
    brick_library_dir = base_dir / "data" / "brick_library"

    # Find all step files
    step_files = sorted(digital_twin_dir.glob("step*.json"))

    print(f"Found {len(step_files)} assembly steps")
    print("Press Q or ESC to move to the next step\n")

    for step_file in step_files:
        # Extract step number
        step_num = int(step_file.stem.replace('step', ''))

        # Load and visualize
        result = load_digital_twin_step(step_num, digital_twin_dir, brick_library_dir)
        if result is None:
            continue

        meshes, step_data = result

        if not meshes:
            continue

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50.0, origin=[0, 0, 0]
        )

        print(f"\nShowing Step {step_num}/{len(step_files)}...")

        # Visualize
        o3d.visualization.draw_geometries(
            meshes + [coord_frame],
            window_name=f"Digital Twin - Step {step_num}/{len(step_files)} ({step_data['num_bricks']} bricks)",
            width=1600,
            height=900,
            left=50,
            top=50,
            mesh_show_back_face=True
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg.lower() == "all":
            # Visualize all steps sequentially
            visualize_all_steps()
        else:
            # Visualize specific step
            try:
                step = int(arg)
                if step < 1 or step > 7:
                    print("Error: Step number must be between 1 and 7")
                    sys.exit(1)
                visualize_digital_twin(step)
            except ValueError:
                print(f"Error: Invalid step number '{arg}'")
                print("Usage: python view_digital_twin.py [step_number|all]")
                sys.exit(1)
    else:
        # Default: show step 1
        print("No step specified, showing Step 1")
        print("Usage: python view_digital_twin.py [step_number|all]\n")
        visualize_digital_twin(1)
