#!/usr/bin/env python3
"""
EXPERIMENTAL FEATURE

Build a library of 3D models for unique LEGO parts.
Process each unique part once, then reuse for efficiency.

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

import json
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_processing.ldraw_parser import LDrawParser
from cad_processing.mesh_builder import MeshBuilder


def scan_unique_parts(input_dir: Path, ldraw_library: Path) -> dict:
    """
    Scan all LDR files to find unique part types used.

    Returns:
        Dict mapping part_id -> list of color_ids used
    """
    parser = LDrawParser(ldraw_library)
    ldr_files = sorted(input_dir.glob("*.ldr"))

    unique_parts = defaultdict(set)

    print(f"Scanning {len(ldr_files)} files for unique parts...")

    for ldr_file in ldr_files:
        parts = parser.parse_ldr_file(ldr_file)
        for part in parts:
            unique_parts[part.part_id].add(part.color_id)

    print(f"\nFound {len(unique_parts)} unique part types")
    print(f"Total part+color combinations: {sum(len(colors) for colors in unique_parts.values())}")

    return {part_id: list(colors) for part_id, colors in unique_parts.items()}


def build_brick_library(
    unique_parts: dict,
    output_dir: Path,
    ldraw_library: Path
):
    """
    Build 3D models for each unique part type.

    Args:
        unique_parts: Dict of part_id -> list of color_ids
        output_dir: Output directory for brick library
        ldraw_library: Path to LDraw library
    """
    mesh_builder = MeshBuilder(ldraw_library)

    # Create output directories
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    library_metadata = {
        "parts": {}
    }

    print(f"\nBuilding brick library...")
    print(f"Output: {output_dir}\n")

    total_parts = sum(len(colors) for colors in unique_parts.values())
    processed = 0

    for part_id, color_ids in unique_parts.items():
        part_name = part_id.replace('.dat', '')

        for color_id in color_ids:
            processed += 1
            print(f"[{processed}/{total_parts}] Processing {part_name} (color {color_id})...")

            try:
                # Create a dummy part instance to use mesh builder
                from cad_processing.ldraw_parser import PartInstance
                dummy_part = PartInstance(
                    part_id=part_id,
                    color_id=color_id,
                    position=np.array([0.0, 0.0, 0.0]),
                    rotation_matrix=np.eye(3)
                )

                # Build mesh for this part
                mesh = mesh_builder.build_mesh([dummy_part], combine=True)

                # Save mesh
                mesh_filename = f"{part_name}_c{color_id}.obj"
                mesh_path = meshes_dir / mesh_filename
                mesh.export(str(mesh_path))

                # Store metadata
                key = f"{part_id}_{color_id}"
                library_metadata["parts"][key] = {
                    "part_id": part_id,
                    "color_id": color_id,
                    "mesh_file": mesh_filename,
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces)
                }

                print(f"  ✓ {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

    # Save library metadata
    metadata_path = output_dir / "library_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(library_metadata, f, indent=2)

    print(f"\n✓ Brick library created!")
    print(f"  Parts processed: {len(library_metadata['parts'])}")
    print(f"  Meshes: {meshes_dir}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    # Paths
    input_dir = Path(__file__).parent.parent / "data" / "Lego Studio" / "123456"
    output_dir = Path(__file__).parent.parent / "data" / "brick_library"
    ldraw_library = Path(__file__).parent.parent / "data" / "ldraw_library" / "ldraw"

    # Step 1: Scan for unique parts
    unique_parts = scan_unique_parts(input_dir, ldraw_library)

    # Step 2: Build library
    build_brick_library(unique_parts, output_dir, ldraw_library)
