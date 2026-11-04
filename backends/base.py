from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class Backend(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Send messages and return the assistant reply as plain text."""

    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        """Start a streamed reply and return an async iterator of text chunks."""

    @abstractmethod
    def model_card(self) -> dict[str, Any]:
        """Return an OpenAI-compatible model object."""
