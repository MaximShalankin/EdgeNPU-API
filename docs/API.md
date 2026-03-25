# API Reference

EdgeNPU-API implements a subset of the OpenAI API for compatibility with existing clients.

## Base URL

```
http://localhost:8082
```

## Endpoints

### GET /

Server information.

**Response:**
```json
{
  "name": "LLM HTTP Server",
  "version": "1.1.0",
  "model": "qwen3-4b",
  "type": "llm",
  "mode": "persistent"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "qwen3-4b",
  "type": "llm",
  "llm_loaded": true,
  "vision_loaded": false,
  "mode": "persistent"
}
```

| Field | Type | Description |
|-------|------|-------------|
| status | string | "healthy" or "unhealthy" |
| llm_loaded | bool | LLM model loaded |
| vision_loaded | bool | Vision encoder loaded (VLM only) |

### GET /v1/models

List available models (OpenAI-compatible).

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-4b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "local"
    }
  ]
}
```

### GET /v1/models/{model_id}

Get model information.

**Response:**
```json
{
  "id": "qwen3-4b",
  "object": "model",
  "created": 1700000000,
  "owned_by": "local"
}
```

### POST /v1/chat/completions

Create a chat completion.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| messages | array | yes | Conversation messages |
| model | string | no | Model name (ignored, uses loaded model) |
| max_tokens | number | no | Max tokens (ignored, uses CLI setting) |
| stream | bool | no | Streaming (not supported, ignored) |

**Message Formats:**

Text-only:
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

With image (VLM):
```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
```

**Image URL formats:**
- Data URL: `data:image/jpeg;base64,<base64_data>`
- Data URL: `data:image/png;base64,<base64_data>`

**Response:**
```json
{
  "id": "chatcmpl-169c262c4",
  "object": "chat.completion",
  "created": 1774346948,
  "model": "qwen3-4b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I assist you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

**Error Response:**
```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error"
  }
}
```

## Error Codes

| HTTP Code | Description |
|-----------|-------------|
| 400 | Invalid request (missing fields, parse error) |
| 404 | Model not found |
| 500 | Server error (inference failed) |
| 503 | Models not loaded |

## Limitations

- **No streaming** — Full response returned after completion
- **No conversation history** — Each request is independent
- **No batch processing** — Single message per request
- **Token counts are zero** — Not tracked by RKLLM runtime

## Compatibility

Tested with:
- OpenAI Python SDK (basic chat)
- LangChain (ChatOpenAI)
- Custom HTTP clients
