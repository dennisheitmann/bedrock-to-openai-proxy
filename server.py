import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from backends import BedrockBackend
from backends.base import Backend

BACKENDMODEL = "global.anthropic.claude-sonnet-4-6"
API_KEY = "secret_api_key_you_have_to_replace"

_bearer = HTTPBearer()

def _verify_api_key(credentials: HTTPAuthorizationCredentials = Security(_bearer)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# --- Backend registry ---
# Add or remove backends here. Key = model id used in API requests.
_backends: dict[str, Backend] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _backends[BACKENDMODEL] = BedrockBackend(BACKENDMODEL, region="us-east-1")
    yield
    for backend in _backends.values():
        if hasattr(backend, "close"):
            backend.close()


app = FastAPI(lifespan=lifespan)


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
    stream: bool = False  # streaming not supported


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
    reply = await backend.complete(messages, is_new=_is_new_conversation(req.messages))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
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
