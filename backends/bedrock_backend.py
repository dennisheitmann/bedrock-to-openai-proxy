import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import boto3
from botocore.config import Config

from backends.base import Backend


class BedrockBackend(Backend):
    def __init__(
        self,
        model_id: str,
        region: str = "eu-north-1",
        max_tokens: int = 64000,
        timeout: int = 300,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                # Longest silence (seconds) allowed while waiting on Bedrock. For streamed
                # replies this applies between chunks; for non-streamed replies it covers
                # the whole generation.
                read_timeout=timeout,
                # Retries after the first attempt. Keep this low so one slow call cannot
                # hang the proxy for many minutes: 1 retry means at most 2 x read_timeout.
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._invoke, messages)

    async def stream(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        # The Bedrock stream is opened before this returns, so errors such as missing
        # credentials or an unknown model are raised here, before the HTTP response starts.
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self._open_stream, messages)
        return self._iter_stream(response["stream"])

    def _request_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        bedrock_messages, system = _convert_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if system:
            kwargs["system"] = system
        return kwargs

    def _invoke(self, messages: list[dict[str, Any]]) -> str:
        response = self._client.converse(**self._request_kwargs(messages))
        print(f"[bedrock] completed, stopReason={response.get('stopReason')} usage={response.get('usage')}", flush=True)
        content = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(b.get("text", "") for b in content if "text" in b)

    def _open_stream(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self._client.converse_stream(**self._request_kwargs(messages))

    async def _iter_stream(self, events: Any) -> AsyncIterator[str]:
        # boto3 event streams block while waiting for data, so each event is read in a
        # worker thread to keep the event loop free for other requests.
        loop = asyncio.get_running_loop()
        iterator = iter(events)
        while True:
            event = await loop.run_in_executor(None, next, iterator, None)
            if event is None:
                return
            if "messageStop" in event:
                print(f"[bedrock] stream finished, stopReason={event['messageStop'].get('stopReason')}", flush=True)
            elif "metadata" in event:
                print(f"[bedrock] usage={event['metadata'].get('usage')}", flush=True)
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield delta["text"]

    def model_card(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "bedrock",
            "display_name": self.model_id,
            "name": self.model_id,
        }


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if b.get("type") in ("text", "input_text"))
    return ""


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict] | None]:
    """Convert OpenAI-style messages to Bedrock converse format."""
    bedrock_msgs = []
    system = []

    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or m.get("output", "")
        text = _extract_text(content)
        if not text:
            continue
        if role == "system":
            system.append({"text": text})
        elif role in ("user", "assistant"):
            # Bedrock requires alternating user/assistant turns — merge consecutive same-role messages
            if bedrock_msgs and bedrock_msgs[-1]["role"] == role:
                bedrock_msgs[-1]["content"][0]["text"] += "\n" + text
            else:
                bedrock_msgs.append({"role": role, "content": [{"text": text}]})

    # Bedrock requires the first message to be from the user
    if bedrock_msgs and bedrock_msgs[0]["role"] != "user":
        bedrock_msgs.insert(0, {"role": "user", "content": [{"text": ""}]})

    return bedrock_msgs, system if system else None
