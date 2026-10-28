from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Send messages and return the assistant reply as plain text."""

    @abstractmethod
    def model_card(self) -> dict[str, Any]:
        """Return an OpenAI-compatible model object."""
