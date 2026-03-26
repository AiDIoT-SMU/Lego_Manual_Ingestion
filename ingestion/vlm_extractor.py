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
import httpx
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


def _patch_litellm_timeout(timeout_seconds: int) -> None:
    """
    Patch litellm's default HTTP timeout.

    litellm uses a hard-coded _DEFAULT_TIMEOUT of 5 seconds for all HTTP requests.
    The timeout parameter in litellm.completion() does NOT update the httpx client's
    read/write timeout, only affects API-level retries.

    This function monkey-patches the _DEFAULT_TIMEOUT module variable before any
    HTTPHandler instances are created, ensuring all HTTP clients use the correct timeout.

    Args:
        timeout_seconds: Timeout in seconds for read/write operations
    """
    import litellm.llms.custom_httpx.http_handler as http_handler_module

    http_handler_module._DEFAULT_TIMEOUT = httpx.Timeout(
        connect=30.0,  # Connection establishment timeout
        read=float(timeout_seconds),  # Response read timeout
        write=float(timeout_seconds),  # Request write timeout
        pool=5.0  # Connection pool timeout
    )
    logger.debug(f"Patched litellm _DEFAULT_TIMEOUT to: {http_handler_module._DEFAULT_TIMEOUT}")


def _build_spatial_prompt(template: str, semantic_data: List[Dict[str, Any]]) -> str:
    """
    Inject n_steps and parts list into spatial prompt template.

    Args:
        template: Spatial prompt template from step_spatial.txt
        semantic_data: List of step dicts with parts_required

    Returns:
        Populated prompt string
    """
    n_steps = len(semantic_data)

    # Build parts list section
    parts_lines = ["PARTS BY STEP:"]
    for i, sem in enumerate(semantic_data):
        parts = sem.get("parts_required", [])
        if parts:
            part_descs = [f'"{p.get("description", "")}"' for p in parts]
            parts_lines.append(f"Step {i}: {', '.join(part_descs)}")
        else:
            parts_lines.append(f"Step {i}: (no parts)")

    parts_list = "\n".join(parts_lines)

    # Replace both placeholders
    prompt = template.replace("{n_steps}", str(n_steps))
    prompt = prompt.replace("{parts_list}", parts_list)

    return prompt


# ── main class ───────────────────────────────────────────────────────────────

class VLMExtractor:
    """
    Extracts structured step information from LEGO instruction pages.

    Uses a two-call approach per page:
      1. Semantic call  — litellm (retry-enabled) — text only, no coordinates
      2. Spatial call   — google-genai SDK        — bounding boxes only
    Results are merged by step index.
    """

    def __init__(
        self,
        vlm_model: str,
        api_key: str,
        max_retries: int = 3,
        timeout: int = 60,
        spatial_prompt_template: str = "",
    ):
        self.litellm_model = vlm_model
        # google-genai uses bare model name without the "gemini/" litellm prefix
        self.genai_model = vlm_model.split("/", 1)[-1]
        self.max_retries = max_retries
        self.timeout = timeout
        self.spatial_prompt_template = spatial_prompt_template
        self.genai_client = genai.Client(api_key=api_key)
        os.environ["GEMINI_API_KEY"] = api_key

        # Configure litellm
        litellm.drop_params = True
        _patch_litellm_timeout(self.timeout)

        logger.info(f"VLMExtractor ready — semantic: {self.litellm_model} | spatial: {self.genai_model} | timeout: {self.timeout}s")

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

        # ── Call 2: Enhanced Spatial (includes individual parts) ──────────
        logger.debug("  Call 2 — enhanced spatial")
        spatial_prompt = _build_spatial_prompt(self.spatial_prompt_template, semantic_data)
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
                # response.text concatenates ALL parts including thinking tokens.
                # Extract only non-thinking parts to avoid passing reasoning text
                # into the JSON parser.
                parts = response.candidates[0].content.parts
                response_text = "".join(
                    p.text for p in parts
                    if not getattr(p, "thought", False) and getattr(p, "text", None)
                )
                result = _parse_json(response_text)
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

    def _litellm_with_retry(self, messages: List[Dict], timeout: Optional[int] = None) -> str:
        """
        Call litellm with retry logic.

        Args:
            messages: List of message dicts for the VLM
            timeout: Optional timeout in seconds. If None, uses self.timeout.
                     Note: The httpx client timeout is configured via _DEFAULT_TIMEOUT monkey-patch in __init__.
        """
        retry_delay = 2
        effective_timeout = timeout if timeout is not None else self.timeout

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.litellm_model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=65535,
                    timeout=effective_timeout,
                )
                msg = response.choices[0].message
                text = msg.content
                if text is None:
                    finish_reason = response.choices[0].finish_reason
                    extra_fields = {k: v for k, v in vars(msg).items() if v is not None and k != "content"}
                    logger.warning(
                        f"  content=None | finish_reason={finish_reason} | "
                        f"other fields: {list(extra_fields.keys())} | "
                        f"usage={response.usage}"
                    )
                    if attempt < self.max_retries - 1:
                        wait = retry_delay * (2 ** attempt)
                        logger.warning(f"  Semantic call returned None content, retry in {wait}s")
                        time.sleep(wait)
                        continue
                    raise ValueError("VLM returned None content after all retries")
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

        Spatial boxes now include:
        - "parts_panel_X" - entire parts panel for step X (optional, for reference)
        - "subassembly_X" - subassembly for step X
        - "step_X_part_Y" - individual part Y in step X
        """
        # Index spatial boxes by label
        boxes: Dict[str, List[int]] = {}
        for item in spatial_data:
            label = item.get("label", "")
            box   = item.get("box_2d")
            if label and box and len(box) == 4:
                boxes[label] = box

        steps: List[Step] = []
        for i, sem in enumerate(semantic_data):
            # Build parts list with individual boxes
            parts = []
            for j, p in enumerate(sem.get("parts_required", [])):
                desc = p.get("description", "")

                # Try to get individual part box first
                part_label = f"step_{i}_part_{j}"
                part_bbox = boxes.get(part_label)

                if part_bbox:
                    # Individual part detected
                    bbox = _box2d_to_bbox(part_bbox, orig_w, orig_h)
                else:
                    # Fallback: no individual box detected, set to None (no image)
                    bbox = None

                parts.append(PartInfo(description=desc, bounding_box=bbox))

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
