#!/usr/bin/env python3
"""
EdgeNPU-API Python Client Example
OpenAI-compatible client for RK3588 LLM/VLM server
"""

import base64
import httpx


class EdgeNPUClient:
    """Simple client for EdgeNPU-API server."""

    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url

    def health(self) -> dict:
        """Check server health."""
        response = httpx.get(f"{self.base_url}/health")
        return response.json()

    def list_models(self) -> dict:
        """List available models."""
        response = httpx.get(f"{self.base_url}/v1/models")
        return response.json()

    def chat(self, message: str, model: str = None) -> str:
        """Send a text message and get response."""
        payload = {
            "messages": [{"role": "user", "content": message}]
        }
        if model:
            payload["model"] = model

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120.0
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def chat_with_image(self, message: str, image_path: str, model: str = None) -> str:
        """Send a message with image (VLM mode)."""
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }}
                ]
            }]
        }
        if model:
            payload["model"] = model

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120.0
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]


def main():
    client = EdgeNPUClient()

    # Health check
    print("Health:", client.health())

    # Simple chat
    print("\nChat:", client.chat("Hello! Who are you?"))

    # With image (if available)
    # print("\nVLM:", client.chat_with_image("Describe this", "test.jpg"))


if __name__ == "__main__":
    main()
