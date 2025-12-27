"""Central LLM client with rate limiting for all OpenAI API requests."""

import asyncio
import json
import logging
from typing import Any, cast

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Global semaphore to limit concurrent LLM requests
# Allows up to 20 parallel requests to balance throughput and API rate limits
_semaphore: asyncio.Semaphore | None = None

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
        _semaphore = asyncio.Semaphore(100)
    return _semaphore


async def chat_completion(
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Make a chat completion request with structured JSON output.

    Uses global semaphore to limit concurrent requests to 20.

    Args:
        prompt: The user prompt to send
        schema: JSON schema for structured output
        schema_name: Name for the schema
        model: Optional model override (defaults to settings.openai_model)

    Returns:
        Parsed JSON response as dict

    Raises:
        APITimeoutError: Request timed out
        APIStatusError: HTTP error from API
        APIConnectionError: Cannot connect to API
        ValueError: Empty or invalid response
    """
    client = get_client()
    model = model or settings.openai_model

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from OpenAI")

    return cast(dict[str, Any], json.loads(content))
