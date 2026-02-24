"""
6D Pose Estimation using FPFH + ICP.
Based on research paper methodology for CAD-guided pose estimation.
"""

import open3d as o3d
import numpy as np
from typing import Optional
from .cad_database import CADGroundTruth


class PoseEstimator:
    """
    Estimate 6DoF pose of observed assembly using CAD ground truth.

    Uses FPFH features for global registration, refined with ICP.
    """

    def __init__(self, cad_database: CADGroundTruth):
        """
        Initialize pose estimator.

        Args:
            cad_database: CAD ground truth database
        """
        self.cad_db = cad_database

    def estimate_pose(
        self,
        observed_cloud: o3d.geometry.PointCloud,
        expected_step: int,
        max_correspondence_distance: float = 5.0
    ) -> np.ndarray:
        """
        Estimate 6DoF pose of observed assembly.

        Args:
            observed_cloud: Point cloud from camera (e.g., DEFOM-Stereo)
            expected_step: Which assembly step user is on (1-indexed)
            max_correspondence_distance: Maximum distance for correspondences (mm)

        Returns:
            4x4 transformation matrix (rotation + translation)
        """
        # Get CAD ground truth for this step
        cad_cloud = self.cad_db.get_step_cloud(expected_step)

        # Compute FPFH features
        print(f"Computing FPFH features for step {expected_step}...")
        cad_fpfh = self.compute_fpfh(cad_cloud)
        obs_fpfh = self.compute_fpfh(observed_cloud)

        # Global registration using FPFH + RANSAC
        print("Performing RANSAC-based global registration...")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=observed_cloud,
            target=cad_cloud,
            source_feature=obs_fpfh,
            target_feature=cad_fpfh,
            mutual_filter=True,
            max_correspondence_distance=max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        print(f"RANSAC fitness: {result.fitness:.4f}")

        # ICP refinement for precise alignment
        print("Refining with ICP...")
        icp_result = o3d.pipelines.registration.registration_icp(
            source=observed_cloud,
            target=cad_cloud,
            max_correspondence_distance=2.0,
            init=result.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        print(f"ICP fitness: {icp_result.fitness:.4f}")

        return icp_result.transformation

    def compute_fpfh(
        self,
        cloud: o3d.geometry.PointCloud,
        radius_normal: float = 10.0,
        radius_feature: float = 20.0
    ) -> o3d.pipelines.registration.Feature:
        """
        Compute FPFH (Fast Point Feature Histograms) features.

        Args:
            cloud: Input point cloud
            radius_normal: Radius for normal estimation (mm)
            radius_feature: Radius for feature computation (mm)

        Returns:
            FPFH feature object
        """
        # Estimate normals if not present
        if not cloud.has_normals():
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_normal,
                    max_nn=30
                )
            )

        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            cloud,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature,
                max_nn=100
            )
        )

        return fpfh

    def visualize_registration(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transformation: Optional[np.ndarray] = None
    ):
        """
        Visualize registration result.

        Args:
            source: Source point cloud (observed)
            target: Target point cloud (CAD ground truth)
            transformation: Optional transformation to apply to source
        """
        source_temp = source.clone()
        target_temp = target.clone()

        # Color source red, target green
        source_temp.paint_uniform_color([1, 0, 0])  # Red
        target_temp.paint_uniform_color([0, 1, 0])  # Green

        if transformation is not None:
            source_temp.transform(transformation)

        o3d.visualization.draw_geometries(
            [source_temp, target_temp],
            window_name="Registration Result (Red: Observed, Green: CAD Ground Truth)"
        )
