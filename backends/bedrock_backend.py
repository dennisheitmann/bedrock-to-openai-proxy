import asyncio
from collections.abc import AsyncIterator
from typing import Any

import boto3
from botocore.config import Config

from backends.base import Backend, BadRequestError, Completion, TextStream

# Bedrock Converse stopReason -> OpenAI finish_reason. Unknown values map to "stop".
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
    "refusal": "content_filter",
}


def _finish_reason(stop_reason: str | None) -> str:
    return _FINISH_REASONS.get(stop_reason or "", "stop")


# Converse stream events that report a failure after the response has started.
_STREAM_ERROR_EVENTS = (
    "internalServerException",
    "modelStreamErrorException",
    "serviceUnavailableException",
    "throttlingException",
    "validationException",
)


class BedrockBackend(Backend):
    def __init__(
        self,
        model_id: str,
        region: str = "eu-north-1",
        max_tokens: int = 64000,
        timeout: int = 300,
        forward_sampling_params: bool = False,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        # Claude Sonnet 5.5 rejects temperature/top_p/top_k with a 400 error, so they are
        # dropped by default. Enable only for models that still accept them.
        self.forward_sampling_params = forward_sampling_params
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
        return (await self.complete_full(messages, **kwargs)).text

    async def complete_full(self, messages: list[dict[str, Any]], **kwargs) -> Completion:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._invoke, messages, kwargs)

    async def stream(self, messages: list[dict[str, Any]], **kwargs) -> TextStream:
        # The Bedrock stream is opened before this returns, so errors such as missing
        # credentials or an unknown model are raised here, before the HTTP response starts.
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self._open_stream, messages, kwargs)
        state = TextStream()
        return state.attach(self._iter_stream(response["stream"], state))

    def _inference_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """Map OpenAI-style sampling parameters to Bedrock's inferenceConfig."""
        max_tokens = params.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            # Never exceed the configured ceiling.
            max_tokens = min(max_tokens, self.max_tokens)
        else:
            max_tokens = self.max_tokens
        config: dict[str, Any] = {"maxTokens": max_tokens}

        if self.forward_sampling_params:
            temperature = params.get("temperature")
            top_p = params.get("top_p")
            if temperature is not None:
                # OpenAI allows 0-2, Bedrock Claude models only 0-1.
                config["temperature"] = min(max(float(temperature), 0.0), 1.0)
            elif top_p is not None:
                # Some Claude models reject temperature and topP together, so topP is only
                # sent when no temperature was given.
                config["topP"] = min(max(float(top_p), 0.0), 1.0)

        stop = params.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if isinstance(stop, list):
            sequences = [s for s in stop if isinstance(s, str) and s.strip()]
            if sequences:
                config["stopSequences"] = sequences[:4]
        return config

    def _request_kwargs(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> dict[str, Any]:
        bedrock_messages, system = _convert_messages(messages)
        if not bedrock_messages:
            raise BadRequestError("No usable text content found in the messages (only empty, tool or image content).")
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": self._inference_config(params or {}),
        }
        if system:
            kwargs["system"] = system
        return kwargs

    def _invoke(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> Completion:
        response = self._client.converse(**self._request_kwargs(messages, params))
        stop_reason = response.get("stopReason")
        print(f"[bedrock] completed, stopReason={stop_reason} usage={response.get('usage')}", flush=True)
        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in content if "text" in b)
        return Completion(text, _finish_reason(stop_reason))

    def _open_stream(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._client.converse_stream(**self._request_kwargs(messages, params))

    async def _iter_stream(self, events: Any, state: TextStream) -> AsyncIterator[str]:
        # boto3 event streams block while waiting for data, so each event is read in a
        # worker thread to keep the event loop free for other requests.
        loop = asyncio.get_running_loop()
        iterator = iter(events)
        try:
            while True:
                try:
                    event = await loop.run_in_executor(None, next, iterator, None)
                except Exception as exc:
                    # The HTTP 200 has already been sent, so the error cannot become a
                    # status code. Record it so the server can send an error frame.
                    print(f"[bedrock] stream error: {type(exc).__name__}: {exc}", flush=True)
                    state.error = f"{type(exc).__name__}: {exc}"
                    return
                if event is None:
                    return
                # Errors can also arrive as events instead of raised exceptions.
                for name in _STREAM_ERROR_EVENTS:
                    if name in event:
                        message = event[name].get("message", "") if isinstance(event[name], dict) else ""
                        print(f"[bedrock] stream error event {name}: {message}", flush=True)
                        state.error = f"{name}: {message}".strip()
                        return
                if "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")
                    state.finish_reason = _finish_reason(stop_reason)
                    print(f"[bedrock] stream finished, stopReason={stop_reason}", flush=True)
                elif "metadata" in event:
                    print(f"[bedrock] usage={event['metadata'].get('usage')}", flush=True)
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]
        finally:
            # Runs on normal end, errors and client disconnects. Closing the HTTP response
            # releases the connection and lets a blocked worker thread return early.
            close = getattr(events, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

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
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text"):
                text = b.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict] | None]:
    """Convert OpenAI-style messages to Bedrock converse format."""
    bedrock_msgs = []
    system = []

    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or m.get("output", "")
        text = _extract_text(content)
        if not text.strip():
            continue
        if role in ("system", "developer"):
            system.append({"text": text})
        elif role in ("user", "assistant"):
            # Bedrock requires alternating user/assistant turns — merge consecutive same-role messages
            if bedrock_msgs and bedrock_msgs[-1]["role"] == role:
                bedrock_msgs[-1]["content"][0]["text"] += "\n" + text
            else:
                bedrock_msgs.append({"role": role, "content": [{"text": text}]})

    # Some models reject a conversation that ends on an assistant turn (no prefill support),
    # which happens when the final user message was empty.
    while bedrock_msgs and bedrock_msgs[-1]["role"] == "assistant":
        bedrock_msgs.pop()

    # Bedrock requires the first message to be from the user
    if bedrock_msgs and bedrock_msgs[0]["role"] != "user":
        bedrock_msgs.insert(0, {"role": "user", "content": [{"text": "(continued)"}]})

    return bedrock_msgs, system if system else None
