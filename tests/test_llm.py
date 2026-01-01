"""Tests for LLM client module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.services.llm import (
    _is_retryable,
    chat_completion,
    get_client,
    get_semaphore,
)


class TestGetClient:
    """Tests for get_client function."""

    def test_get_client_returns_client(self):
        """Should return an AsyncOpenAI client."""
        with patch("app.services.llm._client", None):
            client = get_client()
            assert client is not None

    def test_get_client_returns_same_instance(self):
        """Should return the same client instance."""
        with patch("app.services.llm._client", None):
            client1 = get_client()
            client2 = get_client()
            assert client1 is client2


class TestGetSemaphore:
    """Tests for get_semaphore function."""

    def test_get_semaphore_returns_semaphore(self):
        """Should return an asyncio Semaphore."""
        with patch("app.services.llm._semaphore", None):
            sem = get_semaphore()
            assert isinstance(sem, asyncio.Semaphore)

    def test_get_semaphore_returns_same_instance(self):
        """Should return the same semaphore instance."""
        with patch("app.services.llm._semaphore", None):
            sem1 = get_semaphore()
            sem2 = get_semaphore()
            assert sem1 is sem2


class TestIsRetryable:
    """Tests for _is_retryable function."""

    def test_timeout_error_is_retryable(self):
        """APITimeoutError should be retryable."""
        error = APITimeoutError(request=MagicMock())
        assert _is_retryable(error) is True

    def test_connection_error_is_retryable(self):
        """APIConnectionError should be retryable."""
        error = APIConnectionError(request=MagicMock())
        assert _is_retryable(error) is True

    def test_rate_limit_error_is_retryable(self):
        """HTTP 429 should be retryable."""
        response = MagicMock()
        response.status_code = 429
        error = APIStatusError(message="Rate limited", response=response, body=None)
        assert _is_retryable(error) is True

    def test_server_error_is_retryable(self):
        """HTTP 5xx should be retryable."""
        response = MagicMock()
        response.status_code = 500
        error = APIStatusError(message="Server error", response=response, body=None)
        assert _is_retryable(error) is True

    def test_client_error_not_retryable(self):
        """HTTP 4xx (except 429) should not be retryable."""
        response = MagicMock()
        response.status_code = 400
        error = APIStatusError(message="Bad request", response=response, body=None)
        assert _is_retryable(error) is False

    def test_other_error_not_retryable(self):
        """Other exceptions should not be retryable."""
        error = ValueError("Some error")
        assert _is_retryable(error) is False


class TestChatCompletion:
    """Tests for chat_completion function."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(self):
        """Should return parsed JSON response on success."""
        mock_response = MagicMock()
        mock_response.output_text = '{"key": "value"}'

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("app.services.llm.get_client", return_value=mock_client):
            with patch("app.services.llm._semaphore", asyncio.Semaphore(10)):
                result = await chat_completion(
                    prompt="test prompt",
                    schema={"type": "object"},
                    schema_name="test_schema",
                )

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_chat_completion_empty_response_raises(self):
        """Should raise ValueError on empty response."""
        mock_response = MagicMock()
        mock_response.output_text = ""

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("app.services.llm.get_client", return_value=mock_client):
            with patch("app.services.llm._semaphore", asyncio.Semaphore(10)):
                with pytest.raises(ValueError, match="Empty response"):
                    await chat_completion(
                        prompt="test",
                        schema={"type": "object"},
                        schema_name="test",
                    )

    @pytest.mark.asyncio
    async def test_chat_completion_retries_on_timeout(self):
        """Should retry on timeout error."""
        mock_response = MagicMock()
        mock_response.output_text = '{"success": true}'

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        # First call times out, second succeeds
        mock_client.responses.create = AsyncMock(
            side_effect=[
                APITimeoutError(request=MagicMock()),
                mock_response,
            ]
        )

        with patch("app.services.llm.get_client", return_value=mock_client):
            with patch("app.services.llm._semaphore", asyncio.Semaphore(10)):
                with patch("app.services.llm._BASE_DELAY", 0.01):
                    result = await chat_completion(
                        prompt="test",
                        schema={"type": "object"},
                        schema_name="test",
                    )

        assert result == {"success": True}
        assert mock_client.responses.create.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_completion_raises_after_max_retries(self):
        """Should raise after exhausting retries."""
        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=APITimeoutError(request=MagicMock()))

        with patch("app.services.llm.get_client", return_value=mock_client):
            with patch("app.services.llm._semaphore", asyncio.Semaphore(10)):
                with patch("app.services.llm._BASE_DELAY", 0.01):
                    with patch("app.services.llm._MAX_RETRIES", 2):
                        with pytest.raises(APITimeoutError):
                            await chat_completion(
                                prompt="test",
                                schema={"type": "object"},
                                schema_name="test",
                            )

        # Initial + 2 retries = 3 calls
        assert mock_client.responses.create.call_count == 3

    @pytest.mark.asyncio
    async def test_chat_completion_no_retry_on_client_error(self):
        """Should not retry on 4xx errors."""
        response = MagicMock()
        response.status_code = 400

        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(
            side_effect=APIStatusError(message="Bad request", response=response, body=None)
        )

        with patch("app.services.llm.get_client", return_value=mock_client):
            with patch("app.services.llm._semaphore", asyncio.Semaphore(10)):
                with pytest.raises(APIStatusError):
                    await chat_completion(
                        prompt="test",
                        schema={"type": "object"},
                        schema_name="test",
                    )

        # Should only be called once (no retries)
        assert mock_client.responses.create.call_count == 1
