import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from backends import BedrockBackend
from backends.base import Backend, BadRequestError

BACKENDMODEL = "global.anthropic.claude-sonnet-5-5"
BACKENDREGION = "eu-north-1"
BACKENDMAXTOKENS = 64000  # max output tokens per reply (Bedrock's default is much lower)
BACKENDTIMEOUT = 300  # seconds Bedrock may stay silent before the call is abandoned
API_KEY = "DENNIS_HEITMANN_PROXY_API_KEY"

_bearer = HTTPBearer()

def _verify_api_key(credentials: HTTPAuthorizationCredentials = Security(_bearer)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# --- Backend registry ---
# Add or remove backends here. Key = model id used in API requests.
_backends: dict[str, Backend] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _backends[BACKENDMODEL] = BedrockBackend(
        BACKENDMODEL,
        region=BACKENDREGION,
        max_tokens=BACKENDMAXTOKENS,
        timeout=BACKENDTIMEOUT,
    )
    yield
    for backend in _backends.values():
        if hasattr(backend, "close"):
            backend.close()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(BadRequestError)
async def _bad_request_handler(request: Request, exc: BadRequestError):
    return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})


def _get_backend(model: str) -> Backend:
    if model not in _backends:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found. Available: {list(_backends)}")
    return _backends[model]


# --- Request models ---
class Message(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = BACKENDMODEL
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None

    def backend_params(self) -> dict[str, Any]:
        """Sampling parameters forwarded to the backend (None values are left out)."""
        params = {
            "max_tokens": self.max_completion_tokens or self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
        }
        return {k: v for k, v in params.items() if v is not None}


class ResponsesRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = BACKENDMODEL
    input: list[dict]
    stream: bool = False
    temperature: float = 1.0
    store: bool = False


# --- Helpers ---
def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if b.get("type") in ("input_text", "text"))
    return ""


def _is_new_conversation(messages: list) -> bool:
    """True if no assistant message is present — signals a fresh conversation."""
    return not any(
        (m.get("role") if isinstance(m, dict) else m.role) == "assistant"
        for m in messages
    )


async def _sse_chat_chunks(chunks, model: str):
    """Wrap text deltas in OpenAI chat.completion.chunk server-sent events."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    started = time.monotonic()
    text_chunks = 0
    finished = False
    failed = False

    def frame(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    try:
        yield frame({"role": "assistant", "content": ""})
        async for text in chunks:
            text_chunks += 1
            yield frame({"content": text})
        error = getattr(chunks, "error", None)
        if error:
            # The HTTP 200 is already sent, so report the failure inside the stream.
            failed = True
            payload = {"error": {"message": f"Upstream error: {error}", "type": "upstream_error"}}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            yield frame({}, finish_reason=getattr(chunks, "finish_reason", None) or "stop")
            yield "data: [DONE]\n\n"
        finished = True
    finally:
        # Release the upstream stream even when the client disconnected mid-reply.
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
        if failed:
            state = "failed mid-stream"
        else:
            state = "completed" if finished else "ended early (client disconnected or error)"
        print(f"[stream] {completion_id} {state} after {time.monotonic() - started:.1f}s, {text_chunks} text chunks", flush=True)


def _responses_to_messages(input_messages: list[dict]) -> list[dict]:
    """Normalise /v1/responses input into a flat message list."""
    result = []
    for m in input_messages:
        role = m.get("role", "")
        msg_type = m.get("type", "message")
        content = m.get("content") or m.get("output", "")
        if msg_type == "function_call_output":
            role = "user"
            content = f"[{m.get('name', 'tool')}]\n{_extract_text(content)}"
        result.append({"role": role, "content": content})
    return result


# --- Endpoints ---
@app.get("/v1/models", dependencies=[Depends(_verify_api_key)])
async def list_models():
    return {"object": "list", "data": [b.model_card() for b in _backends.values()]}


@app.get("/v1/models/{model_id}", dependencies=[Depends(_verify_api_key)])
async def get_model(model_id: str):
    return _get_backend(model_id).model_card()


@app.post("/v1/chat/completions", dependencies=[Depends(_verify_api_key)])
async def chat_completions(req: ChatRequest):
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message provided")

    backend = _get_backend(req.model)
    messages = [m.model_dump() for m in req.messages]
    is_new = _is_new_conversation(req.messages)
    print(f"[request] {time.strftime('%H:%M:%S')} stream={req.stream} model={req.model} messages={len(messages)}", flush=True)

    if req.stream:
        chunks = await backend.stream(messages, is_new=is_new, **req.backend_params())
        return StreamingResponse(
            _sse_chat_chunks(chunks, req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    result = await backend.complete_full(messages, is_new=is_new, **req.backend_params())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": result.finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/responses", dependencies=[Depends(_verify_api_key)])
async def responses(req: ResponsesRequest):
    messages = _responses_to_messages(req.input)
    if not messages:
        raise HTTPException(status_code=400, detail="No prompt content found")

    backend = _get_backend(req.model)
    reply = await backend.complete(messages, is_new=_is_new_conversation(req.input))

    return {
        "id": f"resp-{uuid.uuid4().hex}",
        "object": "response",
        "status": "completed",
        "created_at": int(time.time()),
        "model": req.model,
        "output": [{
            "id": f"msg-{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": reply, "annotations": []}],
        }],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=42424)
