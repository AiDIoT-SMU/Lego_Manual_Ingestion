"""
Point Cloud Generator.
Samples point clouds from meshes.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh


class PointCloudGenerator:
    """Generate point clouds from meshes."""

    def __init__(self):
        """Initialize point cloud generator."""
        pass

    def mesh_to_point_cloud(
        self,
        mesh: trimesh.Trimesh,
        num_points: int = 50000,
        use_poisson: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        Convert trimesh to Open3D point cloud.

        Args:
            mesh: Input trimesh object
            num_points: Number of points to sample
            use_poisson: Use Poisson disk sampling (better distribution)

        Returns:
            Open3D PointCloud object
        """
        # Convert trimesh to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Transfer vertex colors if available
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Normalize to 0-1
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Compute normals for better sampling
        o3d_mesh.compute_vertex_normals()

        # Sample point cloud
        if use_poisson:
            pcd = o3d_mesh.sample_points_poisson_disk(
                number_of_points=num_points,
                init_factor=5
            )
        else:
            pcd = o3d_mesh.sample_points_uniformly(
                number_of_points=num_points
            )

        # Estimate normals for point cloud
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=10.0, max_nn=30
            )
        )

        return pcd

    def save_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        output_path: Path,
        format: str = 'auto'
    ):
        """
        Save point cloud to file.

        Args:
            pcd: Open3D PointCloud
            output_path: Output file path
            format: 'pcd', 'ply', 'xyz', or 'auto' (detect from extension)
        """
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Saved point cloud: {output_path}")

    def visualize(self, pcd: o3d.geometry.PointCloud):
        """Visualize point cloud (for debugging)."""
        o3d.visualization.draw_geometries([pcd])
