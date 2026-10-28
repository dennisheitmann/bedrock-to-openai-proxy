# local-bedrock-proxy

A local OpenAI-compatible HTTP proxy that routes requests to AWS Bedrock. Allows any OpenAI SDK-compatible client to use Bedrock models without code changes.

## How it works

```
Client (OpenAI SDK) → local-bedrock-proxy → AWS Bedrock (Claude)
```

The proxy exposes `/v1/chat/completions`, `/v1/responses`, and `/v1/models` endpoints, translating OpenAI-style requests into Bedrock `converse` API calls.

## Requirements

- Python 3.11+
- AWS credentials configured (via `~/.aws/credentials`, environment variables, or IAM role)
- Access to the target Bedrock model enabled in your AWS account

Install dependencies:

```bash
pip install fastapi uvicorn boto3
```

## Configuration

| Environment variable | Required | Description |
|---|---|---|
| `PROXY_API_KEY` | Yes | Bearer token clients must send in the `Authorization` header |

Set the model and region directly in `server.py`:

```python
BACKENDMODEL = "global.anthropic.claude-sonnet-4-6"  # Bedrock model ID
# region is set in lifespan: BedrockBackend(BACKENDMODEL, region="us-east-1")
```

## Running

```bash
# CMD
python server.py
```

The server starts on `http://0.0.0.0:42424`.

## Usage

### OpenAI Python SDK

```python
import openai

client = openai.OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:42424/v1",
)

response = client.chat.completions.create(
    model="global.anthropic.claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### curl

```bash
curl http://localhost:42424/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "global.anthropic.claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{model_id}` | Get a specific model |
| `POST` | `/v1/chat/completions` | OpenAI chat completions |
| `POST` | `/v1/responses` | OpenAI responses API |

All endpoints require `Authorization: Bearer <key>`.

## Project structure

```
local-bedrock-proxy/
├── server.py               # FastAPI app, request models, endpoints
└── backends/
    ├── base.py             # Abstract Backend interface
    └── bedrock_backend.py  # AWS Bedrock implementation
```
