"""
LDraw File Parser.
Parses .ldr files and extracts part instances with transformations.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PartInstance:
    """Single LEGO part placement in an assembly."""
    part_id: str          # e.g., "3004.dat"
    color_id: int         # LDraw color code
    position: np.ndarray  # (x, y, z) in LDraw units
    rotation_matrix: np.ndarray  # 3x3 rotation matrix


class LDrawParser:
    """Parser for LDraw .ldr files."""

    def __init__(self, ldraw_library_path: Path):
        """
        Initialize parser with path to LDraw parts library.

        Args:
            ldraw_library_path: Path to ldraw/ directory (e.g., data/ldraw_library/ldraw)
        """
        self.ldraw_library = ldraw_library_path
        self.parts_dir = ldraw_library_path / "parts"
        self.primitives_dir = ldraw_library_path / "p"

        if not self.parts_dir.exists():
            raise ValueError(f"LDraw parts directory not found: {self.parts_dir}")

    def parse_ldr_file(self, ldr_path: Path) -> List[PartInstance]:
        """
        Parse an LDraw .ldr file and extract all part instances.

        Args:
            ldr_path: Path to .ldr file

        Returns:
            List of PartInstance objects
        """
        parts = []

        with open(ldr_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('0'):
                    continue

                # Parse part reference lines (type 1)
                if line.startswith('1'):
                    try:
                        part = self._parse_part_line(line)
                        parts.append(part)
                    except Exception as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue

        return parts

    def _parse_part_line(self, line: str) -> PartInstance:
        """
        Parse a line type 1 (part reference).

        Format: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part.dat>
        """
        tokens = line.split()

        if len(tokens) < 15:
            raise ValueError(f"Invalid part line: {line}")

        # Extract data
        color_id = int(tokens[1])
        x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])

        # Rotation matrix (3x3, stored as 9 consecutive values)
        rot_values = [float(tokens[i]) for i in range(5, 14)]
        rotation_matrix = np.array(rot_values).reshape(3, 3)

        part_id = tokens[14]

        return PartInstance(
            part_id=part_id,
            color_id=color_id,
            position=np.array([x, y, z]),
            rotation_matrix=rotation_matrix
        )

    def get_part_path(self, part_id: str) -> Path:
        """Get the full path to a part file in the LDraw library."""
        # Try parts directory first
        part_path = self.parts_dir / part_id
        if part_path.exists():
            return part_path

        # Try primitives directory
        prim_path = self.primitives_dir / part_id
        if prim_path.exists():
            return prim_path

        raise FileNotFoundError(f"Part file not found: {part_id}")
