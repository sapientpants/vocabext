"""Async runner utilities for CLI commands."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous CLI code."""
    return asyncio.run(coro)
