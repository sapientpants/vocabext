"""Central LLM client with rate limiting and retry logic for OpenAI API requests."""

import asyncio
import json
import logging
import random
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Global semaphore to limit concurrent LLM requests
_semaphore: asyncio.Semaphore | None = None
_CONCURRENCY = 100

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 30.0  # seconds

# Global client instance
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get or create the global OpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=120.0)
    return _client


def get_semaphore() -> asyncio.Semaphore:
    """Get or create the global LLM semaphore (lazy init for event loop)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_CONCURRENCY)
    return _semaphore


def _is_retryable(error: Exception) -> bool:
    """Check if an error is retryable (transient)."""
    if isinstance(error, APITimeoutError):
        return True
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        # Retry on server errors (5xx) and rate limits (429)
        return bool(error.status_code >= 500 or error.status_code == 429)
    return False


async def chat_completion(
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Make a request to OpenAI Responses API with structured JSON output.

    Uses the OpenAI Responses API (client.responses.create) which provides
    structured output with JSON schema validation. This is distinct from the
    older Chat Completions API.

    See: https://platform.openai.com/docs/api-reference/responses/create

    Uses global semaphore to limit concurrent requests to 100.
    Retries transient errors with exponential backoff.

    Args:
        prompt: The user prompt to send
        schema: JSON schema for structured output
        schema_name: Name for the schema
        model: Optional model override (defaults to settings.openai_model)

    Returns:
        Parsed JSON response as dict

    Raises:
        APITimeoutError: Request timed out (after retries exhausted)
        APIStatusError: HTTP error from API (after retries exhausted)
        APIConnectionError: Cannot connect to API (after retries exhausted)
        ValueError: Empty or invalid response
    """
    client = get_client()
    model = model or settings.openai_model
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with get_semaphore():
                # OpenAI Responses API - uses 'input' (not 'messages') and 'text.format'
                # for structured JSON output. Response uses 'output_text' property.
                response = await client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )

            content = response.output_text
            if not content:
                raise ValueError("Empty response from OpenAI")

            return cast(dict[str, Any], json.loads(content))

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES and _is_retryable(e):
                # Exponential backoff with non-cryptographic jitter for timing only
                jitter = random.uniform(0, 1)  # nosec B311 - safe for timing jitter
                delay = min(_BASE_DELAY * (2**attempt) + jitter, _MAX_DELAY)
                logger.debug(f"Retry {attempt + 1}/{_MAX_RETRIES} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
            else:
                raise

    # Should not reach here, but satisfy type checker and static analysis
    if last_error is not None:
        raise last_error
    raise RuntimeError("chat_completion failed without capturing an exception")
