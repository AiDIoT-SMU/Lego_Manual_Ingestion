"""
CAD Processing Module for LEGO Assembly.
Converts LDraw files to meshes and point clouds.
"""

from .ldraw_parser import LDrawParser
from .mesh_builder import MeshBuilder
from .point_cloud_generator import PointCloudGenerator

__all__ = ['LDrawParser', 'MeshBuilder', 'PointCloudGenerator']
