"""
EXPERIMENTAL FEATURE

Video Quality Filter: Detects blur, hand obstruction, and frame stability.

Uses computer vision (OpenCV) for fast quality metrics before sending frames to VLM.

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class VideoQualityFilter:
    """Filters video frames based on quality metrics."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        stability_threshold: float = 0.95,
        hand_detection_enabled: bool = True
    ):
        """
        Initialize quality filter.

        Args:
            blur_threshold: Laplacian variance threshold (lower = more blurry)
            stability_threshold: Frame similarity threshold for stability check
            hand_detection_enabled: Whether to use hand detection (requires MediaPipe)
        """
        self.blur_threshold = blur_threshold
        self.stability_threshold = stability_threshold
        self.hand_detection_enabled = hand_detection_enabled
        self.previous_frame = None

    def analyze_frame(self, frame_path: Path) -> Dict[str, Any]:
        """
        Analyze a single frame for quality metrics.

        Args:
            frame_path: Path to the frame image

        Returns:
            Dictionary with quality metrics:
            {
                "blur_score": float,
                "is_blurry": bool,
                "stability_score": float,
                "is_stable": bool,
                "has_hands": bool,
                "overall_quality": float (0.0-1.0)
            }
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning(f"Could not read frame: {frame_path}")
            return self._default_metrics()

        # Blur detection
        blur_score = self._detect_blur(img)
        is_blurry = blur_score < self.blur_threshold

        # Stability check (compare with previous frame)
        stability_score = self._check_stability(img)
        is_stable = stability_score >= self.stability_threshold

        # Hand detection
        has_hands = self._detect_hands(img) if self.hand_detection_enabled else False

        # Calculate overall quality (0.0-1.0)
        # High quality = not blurry, stable, no hands
        quality_components = []

        # Normalize blur score (assume max useful blur score is 500)
        quality_components.append(min(blur_score / 500.0, 1.0))

        # Stability score is already 0-1
        quality_components.append(stability_score)

        # Hand penalty: reduce quality by 30% if hands detected
        if has_hands:
            quality_components.append(0.0)
        else:
            quality_components.append(1.0)

        overall_quality = np.mean(quality_components)

        # Store current frame for next stability check
        self.previous_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return {
            "blur_score": float(blur_score),
            "is_blurry": is_blurry,
            "stability_score": float(stability_score),
            "is_stable": is_stable,
            "has_hands": has_hands,
            "overall_quality": float(overall_quality)
        }

    def _detect_blur(self, img: np.ndarray) -> float:
        """
        Detect blur using Laplacian variance.

        Args:
            img: OpenCV image (BGR)

        Returns:
            Blur score (higher = sharper)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def _check_stability(self, img: np.ndarray) -> float:
        """
        Check frame stability by comparing with previous frame.

        Args:
            img: OpenCV image (BGR)

        Returns:
            Similarity score (0.0-1.0, higher = more similar = more stable)
        """
        if self.previous_frame is None:
            return 1.0  # First frame is considered stable

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if needed for comparison
        if gray.shape != self.previous_frame.shape:
            gray = cv2.resize(gray, (self.previous_frame.shape[1], self.previous_frame.shape[0]))

        # Compute structural similarity
        # Use normalized cross-correlation as a simple similarity metric
        gray_norm = gray.astype(np.float32) / 255.0
        prev_norm = self.previous_frame.astype(np.float32) / 255.0

        # Calculate mean squared error
        mse = np.mean((gray_norm - prev_norm) ** 2)

        # Convert to similarity (0 = identical, higher = different)
        # Use exponential decay to map MSE to similarity
        similarity = np.exp(-mse * 20)

        return float(similarity)

    def _detect_hands(self, img: np.ndarray) -> bool:
        """
        Detect hands in frame using simple skin color detection.

        This is a lightweight heuristic. For production, consider MediaPipe Hands.

        Args:
            img: OpenCV image (BGR)

        Returns:
            True if hands likely present
        """
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        # These values work for typical indoor lighting
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Calculate percentage of skin pixels
        skin_percentage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

        # If more than 5% of the frame is skin color, likely has hands
        # This threshold may need tuning based on your videos
        has_hands = skin_percentage > 0.05

        return has_hands

    def _default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when frame cannot be read."""
        return {
            "blur_score": 0.0,
            "is_blurry": True,
            "stability_score": 0.0,
            "is_stable": False,
            "has_hands": True,
            "overall_quality": 0.0
        }

    def reset(self):
        """Reset internal state (e.g., previous frame for stability check)."""
        self.previous_frame = None


def batch_filter_frames(
    frame_paths: list[Path],
    blur_threshold: float = 100.0,
    stability_threshold: float = 0.95,
    quality_threshold: float = 0.5
) -> list[Dict[str, Any]]:
    """
    Filter a batch of frames and return quality metrics for each.

    Args:
        frame_paths: List of frame paths
        blur_threshold: Laplacian variance threshold
        stability_threshold: Frame similarity threshold
        quality_threshold: Minimum overall quality to pass

    Returns:
        List of dicts with frame_path and quality metrics
    """
    filter = VideoQualityFilter(blur_threshold, stability_threshold)
    results = []

    for frame_path in frame_paths:
        metrics = filter.analyze_frame(frame_path)
        results.append({
            "frame_path": frame_path,
            **metrics,
            "passes_quality": metrics["overall_quality"] >= quality_threshold
        })

    return results
