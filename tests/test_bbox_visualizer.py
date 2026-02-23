"""
Bounding Box Visualizer: Sends a single instruction page to Gemini
Robotics-ER 1.5 and draws the returned bounding boxes on the image.

Follows the official Gemini Robotics-ER cookbook exactly:
- Uses google-genai SDK directly (not litellm)
- Resizes image to 800px wide before sending (standardised input size)
- thinking_budget=0 for spatial tasks, temperature=0.5
- box_2d [ymin, xmin, ymax, xmax] normalized to 0-1000

Usage:
    uv run python tests/test_bbox_visualizer.py
    uv run python tests/test_bbox_visualizer.py data/manuals/6262059/page_014.png
"""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont, ImageColor
from google import genai
from google.genai import types
from loguru import logger
from config.settings import get_settings

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "output"
RESIZE_WIDTH = 800   # matches cookbook get_image_resized()

COLORS = [
    "red", "green", "blue", "yellow", "orange", "pink", "purple",
    "brown", "gray", "turquoise", "cyan", "magenta", "lime", "navy",
    "maroon", "teal", "olive", "coral", "lavender", "violet", "gold",
] + [name for name, _ in ImageColor.colormap.items()]

# ─── Helpers (mirrors cookbook) ───────────────────────────────────────────────

def get_image_resized(img_path: Path) -> Image.Image:
    """Resize to 800px wide, preserving aspect ratio — matches cookbook."""
    img = Image.open(img_path)
    img = img.resize(
        (RESIZE_WIDTH, int(RESIZE_WIDTH * img.size[1] / img.size[0])),
        Image.Resampling.LANCZOS,
    )
    return img


def parse_json(json_output: str) -> str:
    """Strip markdown fencing — matches cookbook parse_json()."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def plot_bounding_boxes(img: Image.Image, bounding_boxes_json: str) -> Image.Image:
    """
    Mirrors cookbook plot_bounding_boxes() exactly.
    Expects box_2d = [ymin, xmin, ymax, xmax] normalized 0-1000.
    Draws on the image that was sent to the model (already resized).
    """
    width, height = img.size
    logger.info(f"Drawing on image: {width}×{height} px")

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    parsed = parse_json(bounding_boxes_json)
    boxes = json.loads(parsed)

    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]

        # Convert normalized 0-1000 → absolute pixels (cookbook formula)
        abs_y1 = int(box["box_2d"][0] / 1000 * height)
        abs_x1 = int(box["box_2d"][1] / 1000 * width)
        abs_y2 = int(box["box_2d"][2] / 1000 * height)
        abs_x2 = int(box["box_2d"][3] / 1000 * width)

        # Swap if inverted (cookbook does this too)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)

        if "label" in box:
            draw.text((abs_x1 + 8, abs_y1 + 6), box["label"], fill=color, font=font)

    return img

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path("data/manuals/6262059/page_013.png")

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    # Strip litellm provider prefix ("gemini/") — google-genai SDK uses bare model names
    model_id = settings.vlm_model.split("/", 1)[-1]

    # Resize to 800px wide — same as cookbook get_image_resized()
    img = get_image_resized(image_path)
    logger.info(f"Resized image: {img.size[0]}×{img.size[1]} px")

    # Focused prompt for LEGO instruction pages: parts panel + subassembly only
    prompt = (
        "This is a page from a LEGO instruction manual. "
        "Return bounding boxes for EXACTLY these two things:\n\n"
        "1. PARTS PANEL — a small inset rectangle with a blue or light-grey background "
        "and a dark/black outline, usually in a corner of the page, containing one or more "
        "brick images with a quantity label (e.g. '1x', '2x'). "
        'Label it "parts".\n\n'
        "2. SUBASSEMBLY — the main 3D-rendered LEGO piece or partially-built model that is "
        "being assembled in this step, shown larger and outside the parts panel. "
        'Label it "subassembly".\n\n'
        "Ignore: background graphics, page numbers, arrows, decorative images, "
        "and any large hero image of the fully finished product.\n\n"
        "Never return masks or code fencing. "
        'The format must be: [{"box_2d": [ymin, xmin, ymax, xmax], "label": <label>}] '
        "normalized to 0-1000. Values in box_2d must be integers only."
    )

    logger.info(f"Calling {model_id}…")
    response = client.models.generate_content(
        model=model_id,
        contents=[img, prompt],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    raw = response.text
    logger.info("Response received")

    print("\n─── Raw model response ───")
    print(raw)

    # Draw boxes on the resized image (same image the model processed)
    annotated = plot_bounding_boxes(img.copy(), raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{image_path.stem}_bbox.png"
    annotated.save(out_path)
    logger.success(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
