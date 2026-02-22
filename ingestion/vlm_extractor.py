"""
VLM Extractor: Uses Vision-Language Model to extract structured information
from LEGO instruction steps with bounding boxes.
"""

import json
import base64
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
import litellm

from .schemas import Step, PartInfo, SubassemblyInfo, BoundingBox


class VLMExtractor:
    """Extracts structured step information using VLM (Gemini Robotic ER 1.5 preview)."""

    def __init__(self, vlm_model: str, api_key: str, max_retries: int = 3):
        """
        Initialize VLM extractor.

        Args:
            vlm_model: LiteLLM model identifier (e.g., "gemini/gemini-robotics-er-1.5-preview")
            api_key: API key for the VLM provider
            max_retries: Maximum number of retry attempts for transient errors
        """
        self.model = vlm_model
        self.max_retries = max_retries

        # Set API key in environment for LiteLLM
        os.environ["GEMINI_API_KEY"] = api_key

        # Configure LiteLLM
        litellm.drop_params = True  # Drop unsupported parameters

        logger.info(f"VLMExtractor initialized with model: {self.model}")

    def extract_steps(
        self,
        image_paths: List[str],
        prompt_template: str
    ) -> List[Step]:
        """
        Extract all steps from instruction page images.

        Args:
            image_paths: List of paths to instruction page images
            prompt_template: VLM prompt template for extraction

        Returns:
            List of Step objects with bounding boxes

        Raises:
            Exception: If extraction fails after all retries
        """
        logger.info(f"Extracting steps from {len(image_paths)} images")

        all_steps = []

        for img_path in image_paths:
            logger.info(f"Processing {img_path}")

            try:
                # Extract steps from this page
                page_steps = self._extract_from_page(img_path, prompt_template)

                # Add source page path to each step
                for step_data in page_steps:
                    step_data["source_page_path"] = img_path

                # Convert to Step objects
                steps = self._convert_to_steps(page_steps)
                all_steps.extend(steps)

                logger.info(f"Extracted {len(steps)} step(s) from {Path(img_path).name}")

            except Exception as e:
                logger.error(f"Failed to extract from {img_path}: {e}")
                # Continue with other images rather than failing completely
                continue

        logger.info(f"Total steps extracted: {len(all_steps)}")
        return all_steps

    def _extract_from_page(
        self,
        image_path: str,
        prompt_template: str
    ) -> List[Dict[str, Any]]:
        """
        Extract step data from a single page image.

        Args:
            image_path: Path to page image
            prompt_template: VLM prompt template

        Returns:
            List of dictionaries containing raw step data

        Raises:
            Exception: If all retries fail
        """
        # Prepare message with image
        content = [{"type": "text", "text": prompt_template}]

        # Encode image
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Determine mime type
        suffix = path.suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
        }.get(suffix, 'image/png')

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        })

        messages = [{"role": "user", "content": content}]

        # Call VLM with retry logic
        response_text = self._call_with_retry(messages)

        # Parse JSON response
        result = self._parse_json_response(response_text)

        # Normalize to array format
        if isinstance(result, dict):
            return [result]
        elif isinstance(result, list):
            return result
        else:
            raise ValueError(f"Invalid response format: {type(result)}")

    def _call_with_retry(self, messages: List[Dict]) -> str:
        """
        Call litellm.completion with exponential backoff retry logic.

        Args:
            messages: Messages to send to the model

        Returns:
            Response text from the model

        Raises:
            Exception: If all retries fail or non-retryable error occurs
        """
        retry_delay = 2  # Initial delay in seconds

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=60000
                )

                result_text = response.choices[0].message.content

                if result_text is None:
                    raise ValueError("VLM returned None content")

                return result_text

            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__

                # Check if it's a retryable error
                is_retryable = any([
                    "503" in error_str,
                    "overloaded" in error_str.lower(),
                    "unavailable" in error_str.lower(),
                    "429" in error_str,  # Rate limit
                    "rate limit" in error_str.lower(),
                    "timeout" in error_str.lower(),
                    "500" in error_str,  # Internal server error
                ])

                if is_retryable and attempt < self.max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Transient error ({error_type}), retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {str(e)[:200]}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"VLM call failed: {error_type} - {e}")
                    raise

    def _parse_json_response(self, response_text: str) -> Any:
        """
        Parse JSON response from VLM, handling markdown code blocks.

        Args:
            response_text: Raw response text from VLM

        Returns:
            Parsed JSON object (dict or list)

        Raises:
            json.JSONDecodeError: If parsing fails
        """
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from VLM")

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            json_str = response_text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            raise

    def _convert_to_steps(self, step_data_list: List[Dict[str, Any]]) -> List[Step]:
        """
        Convert raw step data dictionaries to Step objects with validation.

        Args:
            step_data_list: List of raw step data dictionaries

        Returns:
            List of validated Step objects
        """
        steps = []

        for step_data in step_data_list:
            try:
                # Convert parts
                parts = []
                for part_data in step_data.get("parts_required", []):
                    bbox_data = part_data.get("bounding_box")
                    bbox = BoundingBox(**bbox_data) if bbox_data else None

                    part = PartInfo(
                        description=part_data.get("description", ""),
                        bounding_box=bbox
                    )
                    parts.append(part)

                # Convert subassemblies
                subassemblies = []
                for sub_data in step_data.get("subassemblies", []):
                    bbox_data = sub_data.get("bounding_box")
                    bbox = BoundingBox(**bbox_data) if bbox_data else None

                    subassembly = SubassemblyInfo(
                        description=sub_data.get("description", ""),
                        bounding_box=bbox
                    )
                    subassemblies.append(subassembly)

                # Create Step object
                step = Step(
                    step_number=step_data.get("step_number", 0),
                    parts_required=parts,
                    subassemblies=subassemblies,
                    actions=step_data.get("actions", []),
                    source_page_path=step_data.get("source_page_path", ""),
                    notes=step_data.get("notes")
                )

                steps.append(step)

            except Exception as e:
                logger.error(f"Failed to convert step data to Step object: {e}")
                logger.debug(f"Step data: {step_data}")
                # Skip malformed steps rather than failing completely
                continue

        return steps
