"""
Tests for CAD Processing Pipeline.
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_processing.ldraw_parser import LDrawParser, PartInstance
from cad_processing.mesh_builder import MeshBuilder
from cad_processing.point_cloud_generator import PointCloudGenerator


# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent
LDRAW_LIBRARY = PROJECT_ROOT / "data" / "ldraw_library" / "ldraw"
TEST_LDR_FILE = PROJECT_ROOT / "data" / "Lego Studio" / "123456" / "sg50_step1.ldr"


class TestLDrawParser:
    """Test LDraw file parsing."""

    def test_parser_initialization(self):
        """Test parser can be initialized with LDraw library."""
        parser = LDrawParser(LDRAW_LIBRARY)
        assert parser.ldraw_library == LDRAW_LIBRARY
        assert parser.parts_dir.exists()

    def test_parse_ldr_file(self):
        """Test parsing a real LDraw file."""
        parser = LDrawParser(LDRAW_LIBRARY)
        parts = parser.parse_ldr_file(TEST_LDR_FILE)

        assert len(parts) > 0, "Should parse at least one part"
        assert all(isinstance(p, PartInstance) for p in parts)

        # Check first part structure
        part = parts[0]
        assert part.part_id.endswith('.dat')
        assert isinstance(part.color_id, int)
        assert part.position.shape == (3,)
        assert part.rotation_matrix.shape == (3, 3)

    def test_part_path_resolution(self):
        """Test part file path resolution."""
        parser = LDrawParser(LDRAW_LIBRARY)
        # Common part that should exist
        path = parser.get_part_path("3001.dat")  # 2x4 brick
        assert path.exists()


class TestMeshBuilder:
    """Test mesh building from parts."""

    def test_mesh_builder_initialization(self):
        """Test mesh builder initialization."""
        builder = MeshBuilder(LDRAW_LIBRARY)
        assert builder.ldraw_library == LDRAW_LIBRARY

    def test_build_mesh_from_parts(self):
        """Test building mesh from parsed parts."""
        parser = LDrawParser(LDRAW_LIBRARY)
        builder = MeshBuilder(LDRAW_LIBRARY)

        # Parse test file
        parts = parser.parse_ldr_file(TEST_LDR_FILE)

        # Build mesh
        mesh = builder.build_mesh(parts, combine=True)

        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        print(f"Built mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    def test_color_mapping(self):
        """Test LDraw color to RGB conversion."""
        builder = MeshBuilder(LDRAW_LIBRARY)

        # Test known colors
        white = builder._get_color(15)
        assert white[0] == 255  # White should be [255, 255, 255, 255]

        red = builder._get_color(4)
        assert red[0] > red[1]  # Red should have more red than green/blue


class TestPointCloudGenerator:
    """Test point cloud generation."""

    def test_point_cloud_generation(self):
        """Test generating point cloud from mesh."""
        # Build a test mesh
        parser = LDrawParser(LDRAW_LIBRARY)
        builder = MeshBuilder(LDRAW_LIBRARY)
        generator = PointCloudGenerator()

        parts = parser.parse_ldr_file(TEST_LDR_FILE)
        mesh = builder.build_mesh(parts, combine=True)

        # Generate point cloud
        pcd = generator.mesh_to_point_cloud(mesh, num_points=10000)

        assert len(pcd.points) > 0
        assert pcd.has_points()
        print(f"Generated point cloud: {len(pcd.points)} points")

    def test_point_cloud_with_colors(self):
        """Test point cloud preserves colors."""
        parser = LDrawParser(LDRAW_LIBRARY)
        builder = MeshBuilder(LDRAW_LIBRARY)
        generator = PointCloudGenerator()

        parts = parser.parse_ldr_file(TEST_LDR_FILE)
        mesh = builder.build_mesh(parts, combine=True)
        pcd = generator.mesh_to_point_cloud(mesh, num_points=5000)

        assert pcd.has_colors(), "Point cloud should have colors"


def test_end_to_end_pipeline():
    """Test complete pipeline from LDraw to point cloud."""
    parser = LDrawParser(LDRAW_LIBRARY)
    builder = MeshBuilder(LDRAW_LIBRARY)
    generator = PointCloudGenerator()

    # Process test file
    parts = parser.parse_ldr_file(TEST_LDR_FILE)
    print(f"Parsed {len(parts)} parts")

    mesh = builder.build_mesh(parts, combine=True)
    print(f"Built mesh: {len(mesh.vertices)} vertices")

    pcd = generator.mesh_to_point_cloud(mesh, num_points=20000)
    print(f"Generated point cloud: {len(pcd.points)} points")

    assert len(parts) > 0
    assert len(mesh.vertices) > 0
    assert len(pcd.points) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
