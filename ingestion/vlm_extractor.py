"""
VLM Extractor: Two-call pipeline per page.

Call 1 — Semantic (litellm, with retry):
    Extracts step numbers, part descriptions, subassembly descriptions,
    actions and notes.  No coordinates.

Call 2 — Spatial (google-genai SDK, with retry):
    Locates bounding boxes for the parts panel and the subassembly render
    for each step on the page.  Mirrors the official Gemini Robotics-ER
    cookbook (800 px resize, thinking_budget=0, temperature=0.5).

The two results are merged by step index (left-to-right page order).
"""

import io
import json
import base64
import time
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
from loguru import logger

import litellm
from PIL import Image as PILImage
from google import genai
from google.genai import types

from .schemas import Step, PartInfo, SubassemblyInfo, BoundingBox


# ── constants ────────────────────────────────────────────────────────────────

SEND_WIDTH = 800   # resize width sent to both calls (cookbook standard)


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Any:
    """Strip markdown fencing and parse JSON."""
    if not text or not text.strip():
        raise ValueError("Empty response from VLM")
    if "```json" in text:
        start = text.find("```json") + 7
        end   = text.find("```", start)
        text  = text[start:end]
    elif "```" in text:
        start = text.find("```") + 3
        end   = text.find("```", start)
        text  = text[start:end]
    return json.loads(text.strip())


def _box2d_to_bbox(box_2d: List[int], img_width: int, img_height: int) -> BoundingBox:
    """Convert normalized [ymin, xmin, ymax, xmax] (0-1000) to pixel BoundingBox."""
    ymin, xmin, ymax, xmax = box_2d
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return BoundingBox(
        x=round(xmin / 1000 * img_width),
        y=round(ymin / 1000 * img_height),
        width=round((xmax - xmin) / 1000 * img_width),
        height=round((ymax - ymin) / 1000 * img_height),
    )


def _resize_image(image_path: str) -> PILImage.Image:
    """Resize to SEND_WIDTH px wide, preserving aspect ratio (cookbook standard)."""
    with PILImage.open(image_path) as img:
        orig_w, orig_h = img.size
        new_h = int(SEND_WIDTH * orig_h / orig_w)
        return img.resize((SEND_WIDTH, new_h), PILImage.Resampling.LANCZOS)


def _image_to_b64(img: PILImage.Image, mime: str = "image/png") -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_spatial_prompt(n_steps: int) -> str:
    step_labels = ", ".join(
        f'"parts_{i}" / "subassembly_{i}"' for i in range(n_steps)
    )
    return (
        f"This LEGO instruction page contains {n_steps} step(s) "
        f"in left-to-right order on the page.\n\n"
        "For EACH step find EXACTLY TWO bounding boxes:\n\n"
        "PARTS PANEL — a small inset rectangle with a blue or light-grey background "
        "and a dark/black outline, usually in a corner of the page (or its half for "
        "multi-step pages), containing brick images with quantity labels (1x, 2x, …). "
        "Label it parts_0 for the first step, parts_1 for the second, etc.\n\n"
        "SUBASSEMBLY — the main 3D-rendered LEGO model shown OUTSIDE and below/beside "
        "the parts panel, representing the build state after completing that step. "
        "Label it subassembly_0 for the first step, subassembly_1 for the second, etc.\n\n"
        "Ignore: background graphics, page numbers, step-number digits, arrows, "
        "decorative illustrations, yellow attachment-hint panels.\n\n"
        "Never return masks or code fencing.\n"
        f"Return: [{{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": <one of {step_labels}>}}] "
        "normalized to 0-1000. Values in box_2d must be integers only."
    )


# ── main class ───────────────────────────────────────────────────────────────

class VLMExtractor:
    """
    Extracts structured step information from LEGO instruction pages.

    Uses a two-call approach per page:
      1. Semantic call  — litellm (retry-enabled) — text only, no coordinates
      2. Spatial call   — google-genai SDK        — bounding boxes only
    Results are merged by step index.
    """

    def __init__(self, vlm_model: str, api_key: str, max_retries: int = 3):
        self.litellm_model = vlm_model
        # google-genai uses bare model name without the "gemini/" litellm prefix
        self.genai_model = vlm_model.split("/", 1)[-1]
        self.max_retries = max_retries
        self.genai_client = genai.Client(api_key=api_key)
        os.environ["GEMINI_API_KEY"] = api_key
        litellm.drop_params = True
        logger.info(f"VLMExtractor ready — semantic: {self.litellm_model} | spatial: {self.genai_model}")

    # ── public ───────────────────────────────────────────────────────────────

    def extract_steps(
        self,
        image_paths: List[str],
        prompt_template: str,
    ) -> List[Step]:
        """
        Extract all steps from a list of instruction page images.

        Args:
            image_paths:     Paths to instruction page images.
            prompt_template: Semantic-only prompt (from step_extraction.txt).

        Returns:
            Flat list of Step objects across all pages.
        """
        logger.info(f"Extracting steps from {len(image_paths)} image(s)")
        all_steps: List[Step] = []

        for img_path in image_paths:
            logger.info(f"Processing {Path(img_path).name}")
            try:
                page_steps = self._extract_from_page(img_path, prompt_template)
                all_steps.extend(page_steps)
                logger.info(f"  → {len(page_steps)} step(s)")
            except Exception as e:
                logger.error(f"Failed on {Path(img_path).name}: {e}")

        logger.info(f"Total steps extracted: {len(all_steps)}")
        return all_steps

    # ── per-page orchestration ────────────────────────────────────────────────

    def _extract_from_page(
        self,
        image_path: str,
        semantic_prompt: str,
    ) -> List[Step]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Capture original dimensions for coordinate rescaling
        with PILImage.open(image_path) as raw:
            orig_w, orig_h = raw.size

        # Shared resized image for both calls
        img_resized = _resize_image(image_path)

        # ── Call 1: Semantic ──────────────────────────────────────────────
        logger.debug("  Call 1 — semantic")
        semantic_data = self._semantic_call(img_resized, semantic_prompt)
        n_steps = len(semantic_data)
        logger.debug(f"  Semantic found {n_steps} step(s)")

        if n_steps == 0:
            return []

        # ── Call 2: Spatial ───────────────────────────────────────────────
        logger.debug("  Call 2 — spatial")
        spatial_prompt = _build_spatial_prompt(n_steps)
        spatial_data = self._spatial_call(img_resized, spatial_prompt)
        logger.debug(f"  Spatial returned {len(spatial_data)} box(es)")

        # ── Merge ─────────────────────────────────────────────────────────
        return self._merge_to_steps(
            semantic_data, spatial_data, orig_w, orig_h, image_path
        )

    # ── call 1: semantic ─────────────────────────────────────────────────────

    def _semantic_call(
        self,
        img: PILImage.Image,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """litellm call — returns list of raw step dicts (text only)."""
        b64 = _image_to_b64(img)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]
        raw = self._litellm_with_retry(messages)
        result = _parse_json(raw)
        return result if isinstance(result, list) else [result]

    # ── call 2: spatial ──────────────────────────────────────────────────────

    def _spatial_call(
        self,
        img: PILImage.Image,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """google-genai call — returns list of {box_2d, label} dicts."""
        retry_delay = 2
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=self.genai_model,
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                result = _parse_json(response.text)
                return result if isinstance(result, list) else [result]

            except Exception as e:
                last_exc = e
                err = str(e).lower()
                retryable = any(k in err for k in ("503", "429", "overloaded", "rate limit", "timeout", "500"))
                if retryable and attempt < self.max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(f"  Spatial call transient error, retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    break

        logger.error(f"  Spatial call failed after {self.max_retries} attempts: {last_exc}")
        return []   # degrade gracefully — semantic data is still used, boxes are None

    # ── litellm retry wrapper ─────────────────────────────────────────────────

    def _litellm_with_retry(self, messages: List[Dict]) -> str:
        retry_delay = 2
        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.litellm_model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=4096,
                )
                text = response.choices[0].message.content
                if text is None:
                    raise ValueError("VLM returned None content")
                return text
            except Exception as e:
                err = str(e).lower()
                retryable = any(k in err for k in ("503", "429", "overloaded", "rate limit", "timeout", "500"))
                if retryable and attempt < self.max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(f"  Semantic call transient error, retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"  Semantic call failed: {e}")
                    raise

    # ── merge ─────────────────────────────────────────────────────────────────

    def _merge_to_steps(
        self,
        semantic_data: List[Dict[str, Any]],
        spatial_data: List[Dict[str, Any]],
        orig_w: int,
        orig_h: int,
        source_page_path: str,
    ) -> List[Step]:
        """
        Merge semantic text with spatial bounding boxes.

        Semantic steps are in page order (index 0, 1, …).
        Spatial boxes use labels "parts_0", "subassembly_0", "parts_1", … to
        identify which step they belong to.
        """
        # Index spatial boxes by (type, step_index)
        boxes: Dict[str, List[int]] = {}
        for item in spatial_data:
            label = item.get("label", "")
            box   = item.get("box_2d")
            if label and box and len(box) == 4:
                boxes[label] = box

        steps: List[Step] = []
        for i, sem in enumerate(semantic_data):
            # Parts: one PartInfo per described part, all sharing the panel bbox
            parts_bbox = boxes.get(f"parts_{i}")
            bbox_parts = _box2d_to_bbox(parts_bbox, orig_w, orig_h) if parts_bbox else None

            parts = [
                PartInfo(
                    description=p.get("description", ""),
                    bounding_box=bbox_parts,
                )
                for p in sem.get("parts_required", [])
            ]

            # Subassembly: one entry per step
            sub_bbox = boxes.get(f"subassembly_{i}")
            bbox_sub = _box2d_to_bbox(sub_bbox, orig_w, orig_h) if sub_bbox else None

            sub_desc = sem.get("subassembly_description", "")
            subassemblies = [SubassemblyInfo(description=sub_desc, bounding_box=bbox_sub)] if sub_desc else []

            step = Step(
                step_number=sem.get("step_number", i + 1),
                parts_required=parts,
                subassemblies=subassemblies,
                actions=sem.get("actions", []),
                source_page_path=source_page_path,
                notes=sem.get("notes"),
            )
            steps.append(step)

        return steps
