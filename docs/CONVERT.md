# Model Conversion Guide

This guide explains how to convert HuggingFace models to `.rkllm` format for RK3588.

## Prerequisites

Conversion must be done on an **x86 machine** (not on RK3588) with:
- Python 3.8+
- 32GB+ RAM (for 7B models)
- rkllm-toolkit installed

## Install rkllm-toolkit

```bash
pip install rkllm-toolkit
```

Or download from [rknn-llm releases](https://github.com/airockchip/rknn-llm/releases).

## Basic Conversion

```python
from rkllm.api import RKLLM

# Create RKLLM instance
llm = RKLLM()

# Load HuggingFace model
llm.load_huggingface(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    model_cvt="qwen2.5"
)

# Build for RK3588
llm.build(
    do_quantization=True,
    optimization_level=3,
    target_platform="rk3588"
)

# Export
llm.export_rkllm("qwen2.5-1.5b-instruct_rk3588.rkllm")
```

## Supported Model Families

| Family | `model_cvt` value | Examples |
|--------|-------------------|----------|
| Qwen2.5 | `qwen2.5` | Qwen2.5-1.5B, Qwen2.5-7B |
| Qwen3 | `qwen3` | Qwen3-4B, Qwen3-8B |
| Qwen3-VL | `qwen3-vl` | Qwen3-VL-2B, Qwen3-VL-4B |
| Llama-3.2 | `llama3.2` | Llama-3.2-1B, Llama-3.2-3B |

## Vision Encoder (VLM)

For VLM models, you also need to convert the vision encoder:

```python
from rknn.api import RKNN

# Vision encoder conversion
rknn = RKNN()
rknn.config(target_platform="rk3588")
rknn.load_onnx(model="vision_encoder.onnx")
rknn.build(do_quantization=True)
rknn.export_rknn("qwen3-vl_vision_rk3588.rknn")
```

## Quantization Options

| Option | Size | Speed | Quality |
|--------|------|-------|---------|
| W8A8 | Smallest | Fastest | Good |
| W4A16 | Small | Fast | Better |
| FP16 | Large | Slower | Best |

Default is W8A8 for best performance on edge devices.

## Pre-converted Models

If you don't want to convert yourself, check these resources:

- [Pelochus/ezrknn-llm](https://github.com/Pelochus/ezrknn-llm) — Collection of converted models
- [Qengineering](https://github.com/Qengineering) — Various RKNN models
- [HuggingFace rknn-llm](https://huggingface.co/models?search=rkllm) — Community models

## Troubleshooting

### Out of memory during conversion
- Use machine with more RAM
- Try smaller model variant
- Use `model_cvt` with lower precision

### Model doesn't load on RK3588
- Verify target platform is `rk3588`
- Check NPU driver version matches
- Ensure model architecture is supported

### Slow inference
- Use W8A8 quantization
- Reduce `max_context_len` in server config
- Check thermal throttling: `cat /sys/class/thermal/thermal_zone*/temp`
