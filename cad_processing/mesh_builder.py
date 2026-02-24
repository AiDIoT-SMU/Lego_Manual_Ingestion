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
        Load part geometry from LDraw library with recursive sub-part loading.

        Returns:
            Tuple of (vertices, faces)
        """
        # Check cache
        if part_id in self.cache:
            return self.cache[part_id]

        part_path = self.parts_dir / part_id

        # Also check primitives and parts/s subdirectories
        if not part_path.exists():
            # Try primitives directory (p/)
            part_path = self.parts_dir.parent / 'p' / part_id
        if not part_path.exists():
            # Try sub-parts directory (parts/s/)
            part_path = self.parts_dir / 's' / part_id
        if not part_path.exists():
            # Try with backslash path (Windows-style in LDraw files)
            if '\\' in part_id:
                clean_id = part_id.replace('\\', '/')
                part_path = self.parts_dir.parent / clean_id

        if not part_path.exists():
            raise FileNotFoundError(f"Part not found: {part_id}")

        vertices = []
        faces = []

        with open(part_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('0'):
                    continue

                tokens = line.split()
                if len(tokens) < 2:
                    continue

                line_type = tokens[0]

                # Type 1: Sub-file reference (RECURSIVE!)
                if line_type == '1':
                    try:
                        # Parse transformation
                        # Format: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <file>
                        sub_part_id = tokens[14]

                        # Position
                        position = np.array([
                            float(tokens[2]),
                            float(tokens[3]),
                            float(tokens[4])
                        ])

                        # Rotation matrix (3x3)
                        rotation = np.array([
                            [float(tokens[5]), float(tokens[6]), float(tokens[7])],
                            [float(tokens[8]), float(tokens[9]), float(tokens[10])],
                            [float(tokens[11]), float(tokens[12]), float(tokens[13])]
                        ])

                        # Recursively load sub-part
                        sub_vertices, sub_faces = self._load_part_geometry(sub_part_id)

                        # Transform sub-part vertices
                        transformed_verts = self._apply_transform(
                            sub_vertices,
                            rotation,
                            position
                        )

                        # Add to current part with face index offset
                        idx_offset = len(vertices)
                        vertices.extend(transformed_verts.tolist())

                        # Offset face indices
                        offset_faces = sub_faces + idx_offset
                        faces.extend(offset_faces.tolist())

                    except (IndexError, ValueError, FileNotFoundError) as e:
                        # Skip problematic sub-parts (optional parts, etc.)
                        pass

                # Type 3: Triangle
                elif line_type == '3' and len(tokens) >= 11:
                    v1 = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
                    v2 = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
                    v3 = [float(tokens[8]), float(tokens[9]), float(tokens[10])]

                    idx = len(vertices)
                    vertices.extend([v1, v2, v3])
                    faces.append([idx, idx+1, idx+2])

                # Type 4: Quad (split into two triangles)
                elif line_type == '4' and len(tokens) >= 14:
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
