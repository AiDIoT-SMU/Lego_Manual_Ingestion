# YOLO-World + SAM3 Archive

This directory contains code and documentation for an experimental object detection pipeline using YOLO-World and SAM3 (Segment Anything Model 3) from Roboflow.

## Archived Date
2026-03-28

## What Was Archived

### Code Files
- `yolo_world_sam3_detector.py` - Main detector module implementing the two-stage detection pipeline
- `test_video_enhancer_v2.py` - Test script for the video enhancement pipeline with YOLO-World + SAM3 integration

### Documentation
- `HALLUCINATION_FIX_v2.md` - Documentation about the Frame 150 hallucination issue and attempted fixes

## What This Code Did

The YOLO-World + SAM3 pipeline was an attempt to prevent VLM (Vision Language Model) hallucination during video frame analysis by:

1. **Stage 1 - YOLO-World**: Zero-shot object detection with text prompts to identify LEGO bricks and return bounding boxes
2. **Stage 2 - SAM3**: Refine segmentation using detected bboxes as prompts to get accurate object masks
3. **Object Tracking**: Compare masks between frames using IoU (Intersection over Union) to detect new objects vs duplicates

The goal was to provide accurate object counts to the VLM to prevent it from hallucinating parts that weren't actually visible in the frame.

## Why It Was Archived

1. **Temporarily Disabled**: The code was already commented out in `video_enhancer_v2.py` (lines 827-895), indicating it was disabled for "VLM-ONLY TESTING"
2. **Code Cleanup**: The user requested removal of all yolo-world + sam3 related code
3. **Preserved for Reference**: Archived rather than deleted in case the approach is useful in the future

## Dependencies That Were Removed

- Roboflow API key configuration (`roboflow_api_key` in `config/settings.py`)
- Imports from `backend.services.yolo_world_sam3_detector`
- NumPy usage for mask operations (removed from `video_enhancer_v2.py`)

## Related Data

Note: The file `data/changi/input/sg50_dependencies.json` still contains `sam3_prompt` fields for each LEGO part. These are just descriptive text strings and were left in place since they don't cause any issues.

## How to Restore (If Needed)

If you want to restore this functionality:

1. Move the files back to their original locations:
   - `yolo_world_sam3_detector.py` → `backend/services/`
   - `test_video_enhancer_v2.py` → project root

2. Restore the imports in `backend/services/video_enhancer_v2.py`:
   ```python
   from backend.services.yolo_world_sam3_detector import (
       detect_objects_yolo_world_sam3,
       annotate_frame_with_objects,
       get_generic_lego_query,
       find_new_masks
   )
   import numpy as np
   ```

3. Add back the Roboflow API key to `config/settings.py`:
   ```python
   roboflow_api_key: str
   ```

4. Uncomment the YOLO-World + SAM3 detection code in the `_validate_placements_with_context` method

## Additional Notes

This was an interesting experiment in using object detection to ground VLM analysis, but it introduced complexity and API costs. The current VLM-only approach with perceptual hashing for duplicate detection may be sufficient for the use case.
