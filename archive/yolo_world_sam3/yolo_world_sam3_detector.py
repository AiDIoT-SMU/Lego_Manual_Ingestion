"""
YOLO-World + SAM3 Object Detection Pipeline.

Two-stage object detection for LEGO assembly:
1. YOLO-World: Zero-shot object detection with text prompts, returns bounding boxes
2. SAM3: Refines segmentation using detected bboxes as prompts

This provides accurate object counting to prevent VLM hallucination.

Note: Originally intended to use Grounding DINO, but Roboflow's hosted Grounding DINO
service is broken ("Model package is broken" error). YOLO-World is a suitable alternative
as it's also a zero-shot object detector that works with text prompts.
"""

import base64
import cv2
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from PIL import Image as PILImage, ImageDraw, ImageFont
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ── API Configuration ─────────────────────────────────────────────────────────

YOLO_WORLD_ENDPOINT = "https://infer.roboflow.com/yolo_world/infer"
SAM3_ENDPOINT = "https://serverless.roboflow.com/sam3/concept_segment"
YOLO_WORLD_CONFIDENCE_THRESHOLD = 0.02  # YOLO-World needs very low threshold for LEGO detection
SAM3_CONFIDENCE_THRESHOLD = 0.5

# Distinct colors for object annotations (RGB)
OBJECT_COLORS = [
    (255, 80, 80),    # red
    (80, 200, 80),    # green
    (80, 120, 255),   # blue
    (255, 200, 0),    # yellow
    (200, 80, 255),   # purple
    (0, 200, 220),    # cyan
    (255, 130, 0),    # orange
    (180, 255, 80),   # lime
    (255, 80, 180),   # pink
    (80, 255, 200),   # teal
]


# ── API Calls ─────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, requests.exceptions.Timeout)
    ),
)
def call_yolo_world_api(
    image_path: str,
    text_query: str,
    api_key: str,
    confidence_threshold: float = YOLO_WORLD_CONFIDENCE_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    Call YOLO-World API for zero-shot object detection with text prompts.

    Args:
        image_path: Path to image file
        text_query: Comma-separated text queries (e.g., "LEGO brick, LEGO plate")
        api_key: Roboflow API key
        confidence_threshold: Detection confidence threshold

    Returns:
        Full response dict with predictions containing bboxes, or None on failure.
        Response format:
        {
            "predictions": [
                {
                    "x": center_x,
                    "y": center_y,
                    "width": w,
                    "height": h,
                    "class": "LEGO brick",
                    "confidence": 0.95
                },
                ...
            ]
        }
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Convert comma-separated text_query to list of individual prompts
    text_prompts = [prompt.strip() for prompt in text_query.split(",")]

    payload = {
        "image": {"type": "base64", "value": image_b64},
        "text": text_prompts,
        "confidence": confidence_threshold,
    }

    try:
        response = requests.post(
            f"{YOLO_WORLD_ENDPOINT}?api_key={api_key}",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"YOLO-World API HTTP error: {e}")
        logger.error(f"Response status: {response.status_code}")
        logger.error(f"Response body: {response.text[:500]}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, requests.exceptions.Timeout)
    ),
)
def call_sam3_api(
    image_path: str,
    text_prompts: Optional[List[str]] = None,
    bbox_prompts: Optional[List[List[int]]] = None,
    api_key: str = "",
    confidence_threshold: float = SAM3_CONFIDENCE_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    Call SAM3 API for object segmentation.

    Args:
        image_path: Path to image file
        text_prompts: List of text prompts (e.g., ["LEGO brick"])
        bbox_prompts: List of bboxes in [x, y, width, height] format (center + size)
        api_key: Roboflow API key
        confidence_threshold: Segmentation confidence threshold

    Returns:
        Full response dict or None on failure.

    Note: Must provide either text_prompts OR bbox_prompts, not both.
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompts = []
    if text_prompts:
        prompts = [{"type": "text", "text": p} for p in text_prompts]
    elif bbox_prompts:
        # Convert bbox format: [x_center, y_center, width, height] -> SAM3 bbox prompt
        prompts = [{"type": "bbox", "bbox": bbox} for bbox in bbox_prompts]
    else:
        raise ValueError("Must provide either text_prompts or bbox_prompts")

    payload = {
        "image": {"type": "base64", "value": image_b64},
        "prompts": prompts,
        "output_prob_thresh": confidence_threshold,
        "format": "json",
    }

    try:
        response = requests.post(
            f"{SAM3_ENDPOINT}?api_key={api_key}",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"SAM3 API HTTP error: {e}")
        logger.error(f"Response status: {response.status_code}")
        logger.error(f"Response body: {response.text[:500]}")
        raise


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_sam3_results(
    response: Optional[Dict[str, Any]],
    prompts: List[str]
) -> List[Optional[Dict[str, Any]]]:
    """
    Parse SAM3 response to extract one result per prompt.

    Returns list where each element is either:
    - {"polygon": [[x,y],...], "confidence": float} for successful detection
    - None for no detection
    """
    if not response:
        return [None] * len(prompts)

    prompt_results = response.get("prompt_results", [])
    results = []

    for i in range(len(prompts)):
        if i >= len(prompt_results):
            results.append(None)
            continue

        preds = prompt_results[i].get("predictions", [])
        if not preds:
            results.append(None)
            continue

        pred = preds[0]  # highest-confidence prediction
        masks = pred.get("masks", [])
        if not masks:
            results.append(None)
            continue

        results.append({
            "polygon": masks[0],   # [[x,y], [x,y], ...]
            "confidence": pred.get("confidence", SAM3_CONFIDENCE_THRESHOLD),
        })

    return results


# ── Mask Utilities ────────────────────────────────────────────────────────────

def polygon_to_mask(polygon: List[List[float]], img_shape: Tuple[int, int]) -> np.ndarray:
    """Convert SAM3 polygon to binary mask."""
    h, w = img_shape
    points = np.array(polygon, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """Convert binary mask to bounding box [x1, y1, x2, y2]."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: Binary mask (numpy array)
        mask2: Binary mask (numpy array)

    Returns:
        IoU score between 0 and 1
    """
    if mask1.shape != mask2.shape:
        return 0.0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def find_new_masks(
    current_masks: List[np.ndarray],
    previous_masks: List[np.ndarray],
    iou_threshold: float = 0.3
) -> List[int]:
    """
    Find indices of masks in current_masks that are NEW (don't match any previous mask).

    A mask is considered "new" if its IoU with all previous masks is below the threshold.
    This helps detect when a new brick is added vs when the frame is a duplicate.

    Args:
        current_masks: List of masks from current frame
        previous_masks: List of masks from previous accepted frame
        iou_threshold: Minimum IoU to consider masks as matching (default 0.3)

    Returns:
        List of indices into current_masks that represent new objects
    """
    if not previous_masks:
        # First frame - all masks are new
        return list(range(len(current_masks)))

    new_mask_indices = []

    for i, curr_mask in enumerate(current_masks):
        # Check if this mask matches any previous mask
        is_new = True
        for prev_mask in previous_masks:
            iou = compute_mask_iou(curr_mask, prev_mask)
            if iou >= iou_threshold:
                # This mask matches a previous mask - not new
                is_new = False
                break

        if is_new:
            new_mask_indices.append(i)

    return new_mask_indices


# ── Two-Stage Detection Pipeline ──────────────────────────────────────────────

def detect_objects_yolo_world_sam3(
    image_path: str,
    text_query: str,
    api_key: str,
    img_shape: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """
    Two-stage object detection: YOLO-World → SAM3.

    Stage 1: YOLO-World detects objects and returns bounding boxes
    Stage 2: SAM3 refines segmentation using detected bboxes as prompts

    Args:
        image_path: Path to image file
        text_query: Comma-separated query (e.g., "LEGO brick, LEGO plate")
        api_key: Roboflow API key
        img_shape: (height, width) of the image

    Returns:
        List of detections with:
        - label: Object class from YOLO-World
        - bbox: [x1, y1, x2, y2] in pixel coordinates
        - mask: Binary mask from SAM3
        - confidence: Detection confidence
    """
    img_h, img_w = img_shape

    # Stage 1: YOLO-World detection
    try:
        yolo_response = call_yolo_world_api(
            image_path,
            text_query,
            api_key,
            YOLO_WORLD_CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        logger.warning(f"YOLO-World detection failed: {e}")
        return []

    if not yolo_response or "predictions" not in yolo_response:
        logger.debug("YOLO-World returned no predictions")
        return []

    predictions = yolo_response["predictions"]
    if not predictions:
        return []

    logger.debug(f"YOLO-World detected {len(predictions)} objects")

    # Stage 2: SAM3 segmentation using detected bboxes
    # Convert YOLO-World bboxes (x_center, y_center, w, h) to SAM3 bbox format
    bbox_prompts = []
    for pred in predictions:
        x_center = pred["x"]
        y_center = pred["y"]
        width = pred["width"]
        height = pred["height"]
        bbox_prompts.append([x_center, y_center, width, height])

    try:
        sam3_response = call_sam3_api(
            image_path,
            text_prompts=None,
            bbox_prompts=bbox_prompts,
            api_key=api_key,
            confidence_threshold=SAM3_CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        logger.warning(f"SAM3 segmentation failed: {e}")
        # Fallback: return detections with bboxes but no masks
        detections = []
        for pred in predictions:
            x_center = pred["x"]
            y_center = pred["y"]
            width = pred["width"]
            height = pred["height"]

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            detections.append({
                "label": pred.get("class", "LEGO object"),
                "bbox": [x1, y1, x2, y2],
                "mask": None,
                "confidence": pred.get("confidence", 0.0)
            })
        return detections

    # Parse SAM3 results and combine with Grounding DINO labels
    sam3_results = parse_sam3_results(sam3_response, [""] * len(predictions))

    detections = []
    for i, (pred, sam3_result) in enumerate(zip(predictions, sam3_results)):
        if sam3_result is None:
            # SAM3 failed for this bbox, use bbox only
            x_center = pred["x"]
            y_center = pred["y"]
            width = pred["width"]
            height = pred["height"]

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            detections.append({
                "label": pred.get("class", "LEGO object"),
                "bbox": [x1, y1, x2, y2],
                "mask": None,
                "confidence": pred.get("confidence", 0.0)
            })
        else:
            # Convert SAM3 polygon to mask
            polygon = sam3_result["polygon"]
            mask = polygon_to_mask(polygon, img_shape)
            bbox = mask_to_bbox(mask)

            if bbox:
                detections.append({
                    "label": pred.get("class", "LEGO object"),
                    "bbox": bbox,
                    "mask": mask,
                    "confidence": pred.get("confidence", 0.0)
                })

    return detections


# ── Frame Annotation ──────────────────────────────────────────────────────────

def annotate_frame_with_objects(
    frame_path: Path,
    object_detections: List[Dict[str, Any]],
    output_path: Path,
    previous_placements: Optional[List[Dict[str, Any]]] = None
) -> Path:
    """
    Annotate frame with detected objects and counts.

    Args:
        frame_path: Path to the frame image
        object_detections: List of detected objects with labels, masks, bboxes, counts
        output_path: Where to save the annotated frame
        previous_placements: Optional list of previous placement results to show context

    Returns:
        Path to annotated frame
    """
    # Load image
    img = cv2.imread(str(frame_path))
    if img is None:
        logger.warning(f"Could not load frame {frame_path}")
        return frame_path

    img_h, img_w = img.shape[:2]

    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()

    # Draw each detected object
    for idx, obj in enumerate(object_detections):
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        label = obj.get("label", "object")
        count = obj.get("count", 1)
        bbox = obj.get("bbox")

        if bbox:
            x1, y1, x2, y2 = bbox
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label with count
            text = f"{label} (x{count})" if count > 1 else label
            text_y = max(0, y1 - 20)

            # Draw background for text
            text_bbox = draw.textbbox((x1, text_y), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, text_y), text, fill=(0, 0, 0), font=font)

    # Add previous placement context at the top if available
    if previous_placements:
        context_y = 10
        draw.rectangle([5, context_y, img_w - 5, context_y + 25], fill=(50, 50, 50), outline=(200, 200, 200), width=2)

        last_placement = previous_placements[-1] if previous_placements else None
        if last_placement:
            context_text = f"Previous: {last_placement.get('action_description', 'N/A')[:80]}"
            draw.text((10, context_y + 5), context_text, fill=(255, 255, 255), font=font)

    # Save annotated frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(output_path), format="JPEG", quality=90)

    return output_path


# ── Prompt Utilities ──────────────────────────────────────────────────────────

def get_generic_lego_query() -> str:
    """
    Return generic LEGO object query for YOLO-World detection.

    Using generic prompts instead of specific parts from enhanced.json to:
    1. Avoid returning no results for overly-specific prompts
    2. Prevent VLM hallucination from seeing the parts list

    Returns comma-separated query string for YOLO-World.
    """
    return "LEGO brick, LEGO plate, LEGO baseplate, LEGO piece"
