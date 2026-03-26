"""
Video Processor: Extracts frames from uploaded assembly videos.
Uses OpenCV to extract frames at specified intervals for VLM analysis.
"""

import cv2
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from config.settings import Settings


class VideoProcessor:
    """Processes videos to extract frames for analysis."""

    def __init__(self, settings: Settings):
        """
        Initialize video processor.

        Args:
            settings: Application settings
        """
        self.settings = settings

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        frame_interval: int = 50,
        max_frames: int = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame (default: 50)
            max_frames: Maximum number of frames to process (default: None = all frames)

        Returns:
            List of frame metadata:
            [
                {
                    "frame_number": 0,
                    "timestamp_seconds": 0.0,
                    "frame_path": "path/to/frame_0000.jpg"
                },
                ...
            ]

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video properties: {total_frames} frames, {fps:.2f} FPS, "
            f"{duration:.2f}s duration"
        )

        extracted_frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Stop if we've reached max_frames limit
            if max_frames is not None and frame_count >= max_frames:
                logger.info(f"Reached max_frames limit ({max_frames}), stopping extraction")
                break

            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps if fps > 0 else 0.0
                frame_filename = f"frame_{frame_count:04d}.jpg"
                frame_path = output_dir / frame_filename

                # Save frame as JPEG
                cv2.imwrite(str(frame_path), frame)

                extracted_frames.append({
                    "frame_number": frame_count,
                    "timestamp_seconds": round(timestamp, 2),
                    "frame_path": str(frame_path)
                })

                logger.debug(f"Extracted frame {frame_count} at {timestamp:.2f}s")

            frame_count += 1

        cap.release()

        logger.info(
            f"Extracted {len(extracted_frames)} frames from {total_frames} total frames "
            f"(limited to {max_frames} frames)" if max_frames else
            f"Extracted {len(extracted_frames)} frames from {total_frames} total frames"
        )

        return extracted_frames

    def get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """
        Get metadata about a video file.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata:
            {
                "fps": float,
                "total_frames": int,
                "duration_seconds": float,
                "width": int,
                "height": int
            }

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "width": width,
            "height": height
        }
