#!/usr/bin/env python3
"""
Check available Gemini models and save to documentation.

This script queries the Google GenAI API to list all available models,
filters for Gemini models with vision capabilities, and saves the results
to a markdown file for reference.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from loguru import logger
from config.settings import get_settings


def main():
    """Check available Gemini models and save results."""
    logger.info("=" * 80)
    logger.info("CHECKING AVAILABLE GEMINI MODELS")
    logger.info("=" * 80)

    # Load settings
    settings = get_settings()

    # Initialize GenAI client
    logger.info("\nInitializing Google GenAI client...")
    client = genai.Client(api_key=settings.gemini_api_key)

    # List all models
    logger.info("Fetching available models...")
    try:
        models = client.models.list()
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return

    # Filter for Gemini models
    gemini_models = []
    for model in models:
        if "gemini" in model.name.lower():
            # Extract model info
            model_info = {
                "name": model.name,
                "display_name": getattr(model, "display_name", "N/A"),
                "description": getattr(model, "description", "N/A"),
                "supported_generation_methods": getattr(model, "supported_generation_methods", []),
                "input_token_limit": getattr(model, "input_token_limit", "N/A"),
                "output_token_limit": getattr(model, "output_token_limit", "N/A"),
            }
            gemini_models.append(model_info)

    logger.info(f"\nFound {len(gemini_models)} Gemini models")

    # Filter out embedding models (not for generation)
    generation_models = [
        m for m in gemini_models
        if "embedding" not in m["name"].lower()
        and "tts" not in m["name"].lower()
    ]

    logger.info(f"Found {len(generation_models)} generative Gemini models")

    # Print models to console
    logger.info("\n" + "=" * 80)
    logger.info("AVAILABLE GEMINI MODELS")
    logger.info("=" * 80)

    for model in generation_models:
        logger.info(f"\nModel: {model['name']}")
        logger.info(f"  Display Name: {model['display_name']}")
        logger.info(f"  Description: {model['description']}")
        logger.info(f"  Input Token Limit: {model['input_token_limit']}")
        logger.info(f"  Output Token Limit: {model['output_token_limit']}")
        logger.info(f"  Methods: {', '.join(model['supported_generation_methods'])}")

    # Save to markdown file
    output_path = Path(__file__).parent.parent / "docs" / "available_gemini_models.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving results to: {output_path}")

    with open(output_path, "w") as f:
        f.write("# Available Gemini Models\n\n")
        f.write(f"*Last updated: {Path(__file__).name} script*\n\n")
        f.write("## Generative Models for VLM Tasks\n\n")
        f.write("These models can be used for vision-language tasks. All Gemini models (except embeddings/TTS) support multimodal inputs including images.\n\n")

        for model in generation_models:
            # Extract model name without "models/" prefix for litellm
            model_short_name = model['name'].replace('models/', '')

            f.write(f"### {model['display_name']}\n\n")
            f.write(f"**Model ID:** `{model['name']}`\n\n")
            f.write(f"**litellm format:** `gemini/{model_short_name}`\n\n")
            f.write(f"**Description:** {model['description']}\n\n")
            f.write(f"**Token Limits:**\n")
            f.write(f"- Input: {model['input_token_limit']:,}\n")
            f.write(f"- Output: {model['output_token_limit']:,}\n\n")
            f.write(f"**Supported Methods:** {', '.join(model['supported_generation_methods'])}\n\n")
            f.write("---\n\n")

        # Add configuration example
        f.write("## Configuration\n\n")
        f.write("To use a different model, update your `.env` file:\n\n")
        f.write("```bash\n")
        f.write("# Recommended models (as of script run):\n")
        f.write("VLM_MODEL=gemini/gemini-2.5-flash       # Stable, balanced performance\n")
        f.write("VLM_MODEL=gemini/gemini-2.5-pro         # High-quality, more expensive\n")
        f.write("VLM_MODEL=gemini/gemini-2.5-flash-lite  # Fast, cost-effective\n")
        f.write("VLM_MODEL=gemini/gemini-robotics-er-1.5-preview  # Specialized for robotics\n")
        f.write("\n")
        f.write("# Alternative models:\n")
        f.write("VLM_MODEL=gemini/gemini-3-pro-preview   # Preview of next generation\n")
        f.write("VLM_MODEL=gemini/gemini-3.1-flash-lite-preview  # Lightweight preview\n")
        f.write("```\n\n")
        f.write("**Note:** Model availability may change over time. Run `python scripts/check_available_gemini_models.py` to get the latest list.\n")

    # Also save JSON for programmatic access
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "generative_models": generation_models,
            "all_gemini_models": gemini_models
        }, f, indent=2)

    logger.info(f"Saved JSON data to: {json_path}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nDocumentation saved to: {output_path}")
    logger.info(f"JSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
