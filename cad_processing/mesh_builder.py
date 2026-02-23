"""
Mesh Builder.
Loads LDraw part geometries and builds combined meshes.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import trimesh
from .ldraw_parser import PartInstance


class MeshBuilder:
    """Build 3D meshes from LDraw part instances."""

    # LDraw color codes to RGB (subset - extend as needed)
    LDRAW_COLORS = {
        0: [0.13, 0.13, 0.13],     # Black
        1: [0.0, 0.2, 0.7],        # Blue
        2: [0.0, 0.55, 0.33],      # Green
        4: [0.8, 0.0, 0.0],        # Red
        7: [0.5, 0.5, 0.5],        # Dark Gray
        8: [0.3, 0.3, 0.3],        # Dark Gray (alternate)
        14: [1.0, 0.84, 0.0],      # Yellow
        15: [1.0, 1.0, 1.0],       # White
        71: [0.7, 0.7, 0.7],       # Light Gray
        72: [0.4, 0.4, 0.4],       # Dark Gray
    }

    def __init__(self, ldraw_library_path: Path):
        """
        Initialize mesh builder.

        Args:
            ldraw_library_path: Path to ldraw/ directory
        """
        self.ldraw_library = ldraw_library_path
        self.parts_dir = ldraw_library_path / "parts"
        self.cache = {}  # Cache loaded part geometries

    def build_mesh(
        self,
        parts: List[PartInstance],
        combine: bool = True
    ) -> trimesh.Trimesh:
        """
        Build a mesh from part instances.

        Args:
            parts: List of PartInstance objects
            combine: If True, combine all parts into single mesh

        Returns:
            Trimesh object
        """
        meshes = []

        for part in parts:
            try:
                # Load part geometry
                vertices, faces = self._load_part_geometry(part.part_id)

                # Apply transformation
                transformed_verts = self._apply_transform(
                    vertices,
                    part.rotation_matrix,
                    part.position
                )

                # Create mesh
                mesh = trimesh.Trimesh(
                    vertices=transformed_verts,
                    faces=faces,
                    process=False  # Don't auto-process to preserve structure
                )

                # Apply color
                color = self._get_color(part.color_id)
                mesh.visual.vertex_colors = np.tile(color, (len(vertices), 1))

                meshes.append(mesh)

            except Exception as e:
                print(f"Warning: Failed to load part {part.part_id}: {e}")
                continue

        if not meshes:
            raise ValueError("No valid parts could be loaded")

        # Combine or return as scene
        if combine:
            combined = trimesh.util.concatenate(meshes)
            return combined
        else:
            return trimesh.Scene(meshes)

    def _load_part_geometry(self, part_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load part geometry from LDraw library.

        Returns:
            Tuple of (vertices, faces)
        """
        # Check cache
        if part_id in self.cache:
            return self.cache[part_id]

        part_path = self.parts_dir / part_id
        if not part_path.exists():
            raise FileNotFoundError(f"Part not found: {part_id}")

        vertices = []
        faces = []

        with open(part_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('0'):
                    continue

                tokens = line.split()
                line_type = tokens[0]

                # Type 3: Triangle
                if line_type == '3':
                    v1 = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
                    v2 = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
                    v3 = [float(tokens[8]), float(tokens[9]), float(tokens[10])]

                    idx = len(vertices)
                    vertices.extend([v1, v2, v3])
                    faces.append([idx, idx+1, idx+2])

                # Type 4: Quad (split into two triangles)
                elif line_type == '4':
                    v1 = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
                    v2 = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
                    v3 = [float(tokens[8]), float(tokens[9]), float(tokens[10])]
                    v4 = [float(tokens[11]), float(tokens[12]), float(tokens[13])]

                    idx = len(vertices)
                    vertices.extend([v1, v2, v3, v4])
                    # Split quad into two triangles
                    faces.append([idx, idx+1, idx+2])
                    faces.append([idx, idx+2, idx+3])

        if not vertices:
            raise ValueError(f"No geometry found in part: {part_id}")

        result = (np.array(vertices), np.array(faces))
        self.cache[part_id] = result
        return result

    def _apply_transform(
        self,
        vertices: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray
    ) -> np.ndarray:
        """Apply rotation and translation to vertices."""
        # Apply rotation
        rotated = vertices @ rotation.T
        # Apply translation
        transformed = rotated + translation
        return transformed

    def _get_color(self, color_id: int) -> np.ndarray:
        """Get RGB color for LDraw color code."""
        rgb = self.LDRAW_COLORS.get(color_id, [0.5, 0.5, 0.5])  # Default gray
        # Return as RGBA with full opacity
        return np.array(rgb + [1.0]) * 255
