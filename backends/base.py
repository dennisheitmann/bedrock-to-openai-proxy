from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


class BadRequestError(ValueError):
    """The request cannot be sent to the backend (e.g. no usable message content).

    The server turns this into an HTTP 400 instead of a 500.
    """


@dataclass
class Completion:
    """A finished reply plus why it ended (an OpenAI finish_reason value)."""

    text: str
    finish_reason: str = "stop"


class TextStream:
    """Async iterator of text chunks that also records why the reply ended.

    `finish_reason` is an OpenAI value ("stop", "length", "content_filter") and is only
    set once the backend has reported it, which is normally just before iteration ends.
    """

    def __init__(self) -> None:
        self.finish_reason: str | None = None
        # Set when the backend failed after the reply had already started streaming.
        self.error: str | None = None
        self._source: AsyncIterator[str] | None = None

    def attach(self, source: AsyncIterator[str]) -> "TextStream":
        self._source = source
        return self

    def __aiter__(self) -> "TextStream":
        return self

    async def __anext__(self) -> str:
        if self._source is None:
            raise StopAsyncIteration
        return await self._source.__anext__()

    async def aclose(self) -> None:
        close = getattr(self._source, "aclose", None)
        if close is not None:
            await close()


class Backend(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Send messages and return the assistant reply as plain text."""

    async def complete_full(self, messages: list[dict[str, Any]], **kwargs) -> Completion:
        """Like complete(), but also reports why the reply ended."""
        return Completion(await self.complete(messages, **kwargs))

    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]], **kwargs) -> TextStream:
        """Start a streamed reply and return an async iterator of text chunks."""

    @abstractmethod
    def model_card(self) -> dict[str, Any]:
        """Return an OpenAI-compatible model object."""
