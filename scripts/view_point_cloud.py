#!/usr/bin/env python3
"""Simple point cloud viewer for LEGO assembly steps."""
import open3d as o3d
import sys
from pathlib import Path

def view_point_cloud(step_number: int):
    """
    View a specific assembly step point cloud.

    Args:
        step_number: Step number to view (1-7)
    """
    base_path = Path(__file__).parent.parent / "data" / "processed" / "123456" / "point_clouds"
    pcd_file = base_path / f"sg50_step{step_number}.pcd"

    if not pcd_file.exists():
        print(f"Error: Point cloud file not found: {pcd_file}")
        return

    print(f"Loading point cloud: {pcd_file.name}")
    pcd = o3d.io.read_point_cloud(str(pcd_file))

    print(f"Point cloud loaded:")
    print(f"  - Points: {len(pcd.points):,}")
    print(f"  - Has colors: {pcd.has_colors()}")
    print(f"  - Has normals: {pcd.has_normals()}")

    # Get bounding box info
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"  - Bounding box: {bbox.get_extent()}")

    print("\nControls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Ctrl+Mouse: Pan")
    print("  - Q or ESC: Close window")

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"LEGO Assembly - Step {step_number}",
        width=1280,
        height=720,
        left=50,
        top=50
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        step = int(sys.argv[1])
    else:
        step = 1

    if step < 1 or step > 7:
        print("Please provide a step number between 1 and 7")
        print("Usage: python view_point_cloud.py [step_number]")
        sys.exit(1)

    view_point_cloud(step)
