"""
Visualization tool for Video Enhancement Pipeline.

Shows which frames were selected at each VLM pass with annotations.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

from config.settings import get_settings


def draw_text_with_background(img, text, position, font_scale=0.6, thickness=2, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Draw text with a background box for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    # Draw background rectangle
    cv2.rectangle(img,
                  (x, y - text_height - 10),
                  (x + text_width + 10, y + baseline),
                  bg_color, -1)

    # Draw text
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, thickness)

    return text_height + baseline + 15


def visualize_pass1_frames(manual_id: str, video_id: str, output_dir: Path, max_frames: int = 20):
    """
    Visualize frames from Pass 1 (classification with quality metrics).

    Creates annotated images showing:
    - Frame number and timestamp
    - Classification (action/placement_candidate)
    - Quality score
    - Hand obstruction, stability flags
    """
    settings = get_settings()

    # Load Pass 1 cache
    cache_path = settings.data_dir / "processed" / manual_id / f"video_frame_quality_{video_id}.json"

    if not cache_path.exists():
        logger.error(f"Pass 1 cache not found: {cache_path}")
        return

    cache = json.loads(cache_path.read_text())

    # Find frame directory
    frames_dir = settings.data_dir / "videos" / manual_id / f"{video_id}_enhancement_frames_v2"
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return

    # Create output directory
    pass1_output = output_dir / "pass1_classification"
    pass1_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Visualizing Pass 1 frames from {cache_path}")
    logger.info(f"Total frames classified: {len(cache)}")

    # Separate by type
    placement_candidates = []
    action_frames = []
    irrelevant_frames = []

    for frame_num_str, data in cache.items():
        frame_num = int(frame_num_str)
        if data.get("is_relevant", True):
            if data.get("frame_type") == "placement_candidate":
                placement_candidates.append((frame_num, data))
            else:
                action_frames.append((frame_num, data))
        else:
            irrelevant_frames.append((frame_num, data))

    logger.info(f"  Placement candidates: {len(placement_candidates)}")
    logger.info(f"  Action frames: {len(action_frames)}")
    logger.info(f"  Irrelevant frames: {len(irrelevant_frames)}")

    # Visualize placement candidates
    logger.info(f"Creating annotated images for placement candidates (showing first {max_frames})...")
    for idx, (frame_num, data) in enumerate(placement_candidates[:max_frames]):
        frame_path = frames_dir / f"frame_{frame_num:04d}.jpg"
        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Resize for easier viewing
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            img = cv2.resize(img, (1200, int(height * scale)))

        # Add annotations
        timestamp = frame_num / 30.0
        quality_score = data.get("quality_score", 0.0)
        has_hands = data.get("has_hand_obstruction", False)
        is_stable = data.get("is_stable", True)
        confidence = data.get("confidence", 0.0)

        y_offset = 30

        # Title
        draw_text_with_background(img, f"PLACEMENT CANDIDATE #{idx + 1}", (10, y_offset),
                                   font_scale=0.8, thickness=2, bg_color=(0, 128, 0))
        y_offset += 35

        # Frame info
        draw_text_with_background(img, f"Frame: {frame_num} | Time: {timestamp:.2f}s", (10, y_offset))
        y_offset += 30

        # Quality metrics
        quality_color = (0, 255, 0) if quality_score >= 0.7 else (0, 165, 255) if quality_score >= 0.5 else (0, 0, 255)
        draw_text_with_background(img, f"Quality: {quality_score:.2f}", (10, y_offset),
                                   bg_color=quality_color, text_color=(0, 0, 0))
        y_offset += 30

        # Flags
        hand_text = "Hands: YES" if has_hands else "Hands: NO"
        hand_color = (0, 0, 255) if has_hands else (0, 255, 0)
        draw_text_with_background(img, hand_text, (10, y_offset), bg_color=hand_color, text_color=(0, 0, 0))
        y_offset += 30

        stable_text = "Stable: YES" if is_stable else "Stable: NO"
        stable_color = (0, 255, 0) if is_stable else (0, 0, 255)
        draw_text_with_background(img, stable_text, (10, y_offset), bg_color=stable_color, text_color=(0, 0, 0))
        y_offset += 30

        draw_text_with_background(img, f"Confidence: {confidence:.2f}", (10, y_offset))

        # Save
        output_path = pass1_output / f"placement_candidate_{idx+1:03d}_frame_{frame_num:04d}.jpg"
        cv2.imwrite(str(output_path), img)

    # Create summary grid for placement candidates
    create_summary_grid(placement_candidates[:20], frames_dir, pass1_output / "placement_candidates_grid.jpg",
                       "PLACEMENT CANDIDATES (Pass 1)")

    # Visualize some action frames too
    logger.info(f"Creating annotated images for action frames (showing first {max_frames//2})...")
    for idx, (frame_num, data) in enumerate(action_frames[:max_frames//2]):
        frame_path = frames_dir / f"frame_{frame_num:04d}.jpg"
        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Resize
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            img = cv2.resize(img, (1200, int(height * scale)))

        timestamp = frame_num / 30.0
        quality_score = data.get("quality_score", 0.0)

        y_offset = 30
        draw_text_with_background(img, f"ACTION FRAME #{idx + 1}", (10, y_offset),
                                   font_scale=0.8, thickness=2, bg_color=(0, 0, 255))
        y_offset += 35
        draw_text_with_background(img, f"Frame: {frame_num} | Time: {timestamp:.2f}s", (10, y_offset))
        y_offset += 30
        draw_text_with_background(img, f"Quality: {quality_score:.2f}", (10, y_offset))

        output_path = pass1_output / f"action_{idx+1:03d}_frame_{frame_num:04d}.jpg"
        cv2.imwrite(str(output_path), img)

    logger.info(f"Pass 1 visualization complete: {pass1_output}")
    logger.info(f"  Saved {min(len(placement_candidates), max_frames)} placement candidate images")
    logger.info(f"  Saved {min(len(action_frames), max_frames//2)} action frame images")


def visualize_pass2_placements(manual_id: str, video_id: str, output_dir: Path, max_placements: int = 30):
    """
    Visualize validated placements from Pass 2.

    Shows:
    - Placement index
    - Action description
    - Parts added
    - Spatial position
    - Confidence
    """
    settings = get_settings()

    # Load Pass 2 cache
    cache_path = settings.data_dir / "processed" / manual_id / f"video_validated_placements_{video_id}.json"

    if not cache_path.exists():
        logger.error(f"Pass 2 cache not found: {cache_path}")
        return

    cache = json.loads(cache_path.read_text())
    placements = cache.get("placements", [])

    # Find frame directory
    frames_dir = settings.data_dir / "videos" / manual_id / f"{video_id}_enhancement_frames_v2"
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return

    # Create output directory
    pass2_output = output_dir / "pass2_validated_placements"
    pass2_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Visualizing Pass 2 validated placements from {cache_path}")
    logger.info(f"Total validated placements: {len(placements)}")
    logger.info(f"Total candidates processed: {cache.get('total_placement_candidates', 'unknown')}")

    if cache.get('total_placement_candidates', 0) > 0:
        duplicate_rate = 1.0 - (len(placements) / cache.get('total_placement_candidates', 1))
        logger.info(f"Duplicate filtering: {duplicate_rate * 100:.1f}% filtered out")

    # Visualize each placement
    for idx, placement in enumerate(placements[:max_placements]):
        frame_num = placement["frame_number"]
        frame_path = frames_dir / f"frame_{frame_num:04d}.jpg"

        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Resize
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            img = cv2.resize(img, (1200, int(height * scale)))

        # Add annotations
        timestamp = placement.get("timestamp", frame_num / 30.0)
        action_desc = placement.get("action_description", "No description")
        confidence = placement.get("confidence", 0.0)
        new_parts = placement.get("new_parts", [])
        spatial = placement.get("spatial_position", {})
        subassembly = placement.get("current_subassembly", "main")

        y_offset = 30

        # Title
        draw_text_with_background(img, f"VALIDATED PLACEMENT #{idx + 1}", (10, y_offset),
                                   font_scale=0.8, thickness=2, bg_color=(255, 128, 0))
        y_offset += 35

        # Frame info
        draw_text_with_background(img, f"Frame: {frame_num} | Time: {timestamp:.2f}s", (10, y_offset))
        y_offset += 30

        # Action description (split into multiple lines if too long)
        desc_lines = wrap_text(action_desc, 60)
        for line in desc_lines[:2]:  # Show max 2 lines
            draw_text_with_background(img, line, (10, y_offset), font_scale=0.5)
            y_offset += 25

        # Parts added
        if new_parts:
            parts_text = f"Parts: {len(new_parts)} added"
            draw_text_with_background(img, parts_text, (10, y_offset), font_scale=0.5)
            y_offset += 25

            for part in new_parts[:2]:  # Show first 2 parts
                part_desc = part.get("part_description", "unknown")
                matched = part.get("matched_from_manual", False)
                match_text = " [MATCHED]" if matched else " [UNMATCHED]"
                color = (0, 255, 0) if matched else (0, 165, 255)
                draw_text_with_background(img, f"  - {part_desc}{match_text}", (10, y_offset),
                                          font_scale=0.4, bg_color=color, text_color=(0, 0, 0))
                y_offset += 22

        # Spatial position
        if spatial:
            location = spatial.get("location", "")
            reference = spatial.get("reference_object", "")
            if location and reference:
                draw_text_with_background(img, f"Position: {location} of {reference}", (10, y_offset),
                                          font_scale=0.5)
                y_offset += 25

        # Confidence
        conf_color = (0, 255, 0) if confidence >= 0.8 else (0, 165, 255) if confidence >= 0.6 else (0, 0, 255)
        draw_text_with_background(img, f"Confidence: {confidence:.2f}", (10, y_offset),
                                  bg_color=conf_color, text_color=(0, 0, 0))

        # Save
        output_path = pass2_output / f"placement_{idx+1:03d}_frame_{frame_num:04d}.jpg"
        cv2.imwrite(str(output_path), img)

    # Create summary grid
    placement_tuples = [(p["frame_number"], p) for p in placements[:20]]
    create_summary_grid(placement_tuples, frames_dir, pass2_output / "validated_placements_grid.jpg",
                       "VALIDATED PLACEMENTS (Pass 2 - After Duplicate Removal)")

    logger.info(f"Pass 2 visualization complete: {pass2_output}")
    logger.info(f"  Saved {min(len(placements), max_placements)} validated placement images")


def create_summary_grid(frames_data: List[tuple], frames_dir: Path, output_path: Path, title: str):
    """Create a grid of thumbnail images."""
    if not frames_data:
        return

    # Load images
    images = []
    for frame_num, _ in frames_data:
        frame_path = frames_dir / f"frame_{frame_num:04d}.jpg"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None:
                # Resize to thumbnail
                img = cv2.resize(img, (300, 200))

                # Add frame number
                draw_text_with_background(img, f"Frame {frame_num}", (5, 25),
                                          font_scale=0.5, thickness=1)
                images.append(img)

    if not images:
        return

    # Create grid (4 columns)
    cols = 4
    rows = (len(images) + cols - 1) // cols

    grid_height = rows * 200
    grid_width = cols * 300

    # Add title space
    title_height = 60
    grid = np.zeros((grid_height + title_height, grid_width, 3), dtype=np.uint8)

    # Draw title
    cv2.putText(grid, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Place images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y_start = row * 200 + title_height
        x_start = col * 300
        grid[y_start:y_start + 200, x_start:x_start + 300] = img

    cv2.imwrite(str(output_path), grid)
    logger.info(f"  Created summary grid: {output_path}")


def wrap_text(text: str, max_chars: int) -> List[str]:
    """Wrap text to multiple lines."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_chars:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def main():
    """Visualize both Pass 1 and Pass 2 results."""
    manual_id = "111111"
    video_id = "changi_airport"

    settings = get_settings()
    output_dir = settings.data_dir / "visualization" / manual_id / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Video Pipeline Visualization")
    logger.info("=" * 80)
    logger.info(f"Manual ID: {manual_id}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Visualize Pass 1
    logger.info("PASS 1: Frame Classification with Quality Metrics")
    logger.info("-" * 80)
    visualize_pass1_frames(manual_id, video_id, output_dir, max_frames=20)
    logger.info("")

    # Visualize Pass 2
    logger.info("PASS 2: Validated Placements (After Duplicate Filtering)")
    logger.info("-" * 80)
    visualize_pass2_placements(manual_id, video_id, output_dir, max_placements=30)
    logger.info("")

    logger.info("=" * 80)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Check outputs in: {output_dir}")
    logger.info("")
    logger.info("Files created:")
    logger.info(f"  - pass1_classification/placement_candidate_*.jpg (annotated frames)")
    logger.info(f"  - pass1_classification/placement_candidates_grid.jpg (grid overview)")
    logger.info(f"  - pass1_classification/action_*.jpg (action frames)")
    logger.info(f"  - pass2_validated_placements/placement_*.jpg (annotated frames)")
    logger.info(f"  - pass2_validated_placements/validated_placements_grid.jpg (grid overview)")


if __name__ == "__main__":
    main()
