import json
from typing import Any

import boto3
from botocore.config import Config

from backends.base import Backend


class BedrockBackend(Backend):
    def __init__(self, model_id: str, region: str = "us-east-1"):
        self.model_id = model_id
        self.region = region
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(read_timeout=300),
        )

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._invoke, messages)

    def _invoke(self, messages: list[dict[str, Any]]) -> str:
        bedrock_messages, system = _convert_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.converse(**kwargs)
        content = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(b.get("text", "") for b in content if "text" in b)

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
