"""
Process LDraw files to generate meshes and point clouds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_processing.ldraw_parser import LDrawParser
from cad_processing.mesh_builder import MeshBuilder
from cad_processing.point_cloud_generator import PointCloudGenerator


def process_all_steps(
    input_dir: Path,
    output_dir: Path,
    ldraw_library: Path,
    num_points: int = 50000
):
    """
    Process all LDraw files in a directory.

    Args:
        input_dir: Directory containing .ldr files
        output_dir: Output directory for meshes and point clouds
        ldraw_library: Path to LDraw library (ldraw/ directory)
        num_points: Number of points to sample per point cloud
    """
    # Initialize processors
    parser = LDrawParser(ldraw_library)
    mesh_builder = MeshBuilder(ldraw_library)
    pcd_generator = PointCloudGenerator()

    # Create output directories
    unity_dir = output_dir / "unity_models"
    pcd_dir = output_dir / "point_clouds"
    unity_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    # Find all .ldr files
    ldr_files = sorted(input_dir.glob("*.ldr"))

    if not ldr_files:
        print(f"No .ldr files found in {input_dir}")
        return

    print(f"Found {len(ldr_files)} LDraw files to process")
    print(f"Output directory: {output_dir}")
    print()

    # Process each file
    for ldr_file in ldr_files:
        step_name = ldr_file.stem
        print(f"Processing {step_name}...")

        try:
            # 1. Parse LDraw file
            parts = parser.parse_ldr_file(ldr_file)
            print(f"  - Parsed {len(parts)} parts")

            # 2. Build mesh
            mesh = mesh_builder.build_mesh(parts, combine=True)
            print(f"  - Built mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            # 3. Export mesh to OBJ (for Unity)
            obj_path = unity_dir / f"{step_name}.obj"
            mesh.export(str(obj_path))
            print(f"  - Saved mesh: {obj_path}")

            # 4. Generate point cloud
            pcd = pcd_generator.mesh_to_point_cloud(
                mesh,
                num_points=num_points,
                use_poisson=True
            )
            print(f"  - Generated point cloud: {len(pcd.points)} points")

            # 5. Save point clouds (both PCD and PLY formats)
            pcd_path = pcd_dir / f"{step_name}.pcd"
            ply_path = pcd_dir / f"{step_name}.ply"

            pcd_generator.save_point_cloud(pcd, pcd_path, format='pcd')
            pcd_generator.save_point_cloud(pcd, ply_path, format='ply')

            print(f"  ✓ {step_name} complete\n")

        except Exception as e:
            print(f"  ✗ Error processing {step_name}: {e}\n")
            continue

    print("=" * 60)
    print("Processing complete!")
    print(f"Unity meshes: {unity_dir}")
    print(f"Point clouds: {pcd_dir}")


if __name__ == "__main__":
    # Configuration
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "Lego Studio" / "123456"
    output_dir = project_root / "data" / "processed" / "123456"
    ldraw_library = project_root / "data" / "ldraw_library" / "ldraw"

    # Process all files
    process_all_steps(
        input_dir=input_dir,
        output_dir=output_dir,
        ldraw_library=ldraw_library,
        num_points=50000
    )
