# EdgeNPU-API

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588-orange.svg)]()
[![API](https://img.shields.io/badge/API-OpenAI%20Compatible-green.svg)]()

OpenAI-compatible HTTP API server for running LLM and VLM models on Rockchip RK3588 NPU.

## Overview

EdgeNPU-API provides a persistent HTTP server for inference with RKNN-compiled language models:
- **LLM mode** — Text-only inference (Qwen3, Qwen2.5, Llama-3.2)
- **VLM mode** — Vision-Language inference with image support (Qwen3-VL)

Key features:
- OpenAI-compatible `/v1/chat/completions` endpoint
- Drop-in replacement for OpenAI API clients
- Persistent model loading (load once, serve many requests)
- Support for multiple prompt formats (Qwen, Llama)

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Platform | RK3588 (Orange Pi 5, Rock 5B, etc.) | |
| RAM | 8 GB | 16 GB |
| OS | Ubuntu 22.04/24.04 (aarch64) | Joshua Riek's Ubuntu |
| OpenCV | 4.x | |
| NPU Driver | 0.9.6+ | |

**Important system tweak:**
```bash
# Increase file descriptor limit (required for large models)
ulimit -n 16384
```

## Model Preparation

You need `.rkllm` model files compiled for RK3588. Options:

### Option 1: Download pre-converted models

| Model | Type | Size | Link |
|-------|------|------|------|
| Qwen3-4B | LLM | ~2.4 GB | [HuggingFace](https://huggingface.co/) |
| Qwen3-VL-4B | VLM | ~4.8 GB | [HuggingFace](https://huggingface.co/) |
| Qwen2.5-1.5B | LLM | ~1.5 GB | [HuggingFace](https://huggingface.co/) |

### Option 2: Convert models yourself

Use [rkllm-toolkit](https://github.com/airockchip/rknn-llm) on an x86 machine with 32GB+ RAM.

See [docs/CONVERT.md](docs/CONVERT.md) for conversion instructions.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/EdgeNPU-API.git
cd EdgeNPU-API

# Build
cmake -B build
cmake --build build

# Run (LLM mode)
LD_LIBRARY_PATH=./lib/aarch64 ./build/rkllm-server \
  --type llm \
  --llm ~/models/qwen3-4b-rk3588-w8a8.rkllm \
  --port 8082

# Run (VLM mode)
LD_LIBRARY_PATH=./lib/aarch64 ./build/rkllm-server \
  --type vlm \
  --llm ~/models/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm \
  --vision ~/models/qwen3-vl_vision_rk3588.rknn \
  --port 8082
```

## Usage

### Command Line Options

```
Usage: rkllm-server [options]

Options:
  --llm <path>          Path to LLM model (.rkllm)
  --vision <path>       Path to vision encoder (.rknn) [VLM only]
  --type <type>         Model type: 'vlm' or 'llm' (default: vlm)
  --prompt-family <f>   Prompt format: 'qwen' or 'llama' (default: qwen)
  --name <name>         Model name for API responses (default: qwen3-vl)
  --port <num>          HTTP server port (default: 8082)
  --max-tokens <num>    Max new tokens (default: 256)
  --context-len <num>   Max context length (default: 512)
  --npu-cores <num>     NPU core count (default: 3)
  --help                Show this help
```

### Examples

**LLM (Qwen3-4B):**
```bash
./build/rkllm-server --type llm --name qwen3-4b \
  --llm ~/models/qwen3-4b.rkllm
```

**LLM (Llama-3.2):**
```bash
./build/rkllm-server --type llm --name llama-3.2-1b \
  --prompt-family llama \
  --llm ~/models/llama32-1b.rkllm
```

**VLM (Qwen3-VL):**
```bash
./build/rkllm-server --type vlm --name qwen3-vl \
  --llm ~/models/qwen3-vl-4b.rkllm \
  --vision ~/models/qwen3-vl_vision.rknn
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server info |
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{id}` | Get model info |
| POST | `/v1/chat/completions` | Chat completion |

### Chat Completion

```bash
# Text-only request
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```bash
# With image (VLM mode)
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'
```

### Response Format

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

See [docs/API.md](docs/API.md) for full API documentation.

## Benchmarks

Tested on Orange Pi 5 (RK3588, 8GB RAM):

| Mode | Model | Load Time | Notes |
|------|-------|-----------|-------|
| LLM | Qwen3-4B | ~64s | Text-only |
| VLM | Qwen3-VL-4B | ~78s | 64s LLM + 14s vision |

Inference speed depends on `max_new_tokens` and prompt length.

## Limitations

- **Single request at a time** — Requests are queued and processed sequentially
- **No streaming** — Responses are returned after completion
- **No conversation history** — Each request is stateless (`keep_history=0`)
- **RAM requirement** — 4B models require 8GB+ RAM

## Project Structure

```
EdgeNPU-API/
├── src/main_server.cpp      # Main server code
├── include/base64.h         # Base64 utilities
├── lib/
│   ├── rkllm/rkllm.h        # RKLLM API headers
│   ├── rknn/rknn_api.h      # RKNN API headers
│   ├── image_enc/           # Vision encoder
│   └── aarch64/*.so         # Runtime libraries
├── third_party/             # Header-only deps (httplib, json)
├── examples/                # Usage examples
├── docs/                    # Documentation
└── CMakeLists.txt
```

## Troubleshooting

### "Too many open files"
```bash
ulimit -n 16384
```

### Model load fails
- Check model path is correct
- Ensure sufficient RAM (`free -h`)
- Verify NPU driver: `cat /sys/kernel/debug/rknpu/driver_version`

### Slow inference
- Reduce `--max-tokens`
- Use smaller model (e.g., Qwen2.5-1.5B)

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Rockchip rknn-llm](https://github.com/airockchip/rknn-llm) — RKLLM runtime
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) — HTTP library
- [nlohmann/json](https://github.com/nlohmann/json) — JSON library
