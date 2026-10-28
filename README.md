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

Set the model, region and API_KEY directly in `server.py`:

```python
BACKENDMODEL = "global.anthropic.claude-sonnet-4-6"
BACKENDREGION = "us-east-1"
API_KEY = "secret_api_key_you_have_to_replace"
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
    api_key="secret_api_key_you_have_to_replace",
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
  -H "Authorization: Bearer secret_api_key_you_have_to_replace" \
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
