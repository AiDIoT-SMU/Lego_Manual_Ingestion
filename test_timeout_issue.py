"""
Test to debug the persistent timeout issue.
"""

import httpx
from loguru import logger

# Test 1: Check what happens when we import VLMExtractor
logger.info("=" * 80)
logger.info("DEBUGGING TIMEOUT ISSUE")
logger.info("=" * 80)

logger.info("\nStep 1: Check litellm HTTPHandler before import")
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _DEFAULT_TIMEOUT
logger.info(f"_DEFAULT_TIMEOUT before import: {_DEFAULT_TIMEOUT}")

handler1 = HTTPHandler()
logger.info(f"New HTTPHandler timeout: {handler1.client.timeout}")

logger.info("\nStep 2: Import config and create VLMExtractor")
from config.settings import get_settings
from ingestion.vlm_extractor import VLMExtractor

settings = get_settings()
logger.info(f"Settings VLM timeout: {settings.vlm_timeout}")

logger.info("\nStep 3: Check _DEFAULT_TIMEOUT after VLMExtractor import")
import litellm.llms.custom_httpx.http_handler as http_handler_module
logger.info(f"_DEFAULT_TIMEOUT after import: {http_handler_module._DEFAULT_TIMEOUT}")

logger.info("\nStep 4: Create VLMExtractor instance")
vlm = VLMExtractor(
    vlm_model=settings.vlm_model,
    api_key=settings.gemini_api_key,
    max_retries=3,
    timeout=settings.vlm_timeout
)

logger.info("\nStep 5: Check _DEFAULT_TIMEOUT after VLMExtractor creation")
logger.info(f"_DEFAULT_TIMEOUT: {http_handler_module._DEFAULT_TIMEOUT}")

logger.info("\nStep 6: Create new HTTPHandler and check timeout")
handler2 = HTTPHandler()
logger.info(f"New HTTPHandler timeout: {handler2.client.timeout}")

logger.info("\nStep 7: Make a test litellm call")
test_messages = [{
    "role": "user",
    "content": "Say 'timeout test' in exactly those two words."
}]

try:
    result = vlm._litellm_with_retry(test_messages)
    logger.info(f"SUCCESS: {result}")
except Exception as e:
    logger.error(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n" + "=" * 80)
