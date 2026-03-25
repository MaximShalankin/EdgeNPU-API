#!/bin/bash
# EdgeNPU-API curl examples
# Usage: ./curl_examples.sh [host]

HOST="${1:-localhost:8082}"

echo "=== EdgeNPU-API Examples ==="
echo "Host: $HOST"
echo

# 1. Server info
echo "1. Server info:"
curl -s "http://$HOST/" | jq .
echo

# 2. Health check
echo "2. Health check:"
curl -s "http://$HOST/health" | jq .
echo

# 3. List models
echo "3. List models:"
curl -s "http://$HOST/v1/models" | jq .
echo

# 4. Simple chat (LLM)
echo "4. Simple chat:"
curl -s -X POST "http://$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in 5 words"}]}' | jq .
echo

# 5. Chat with image (VLM)
echo "5. Chat with image (VLM):"
if [ -f "test_image.jpg" ]; then
  IMAGE_B64=$(base64 -w0 test_image.jpg)
  curl -s -X POST "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMAGE_B64\"}}]}]}" | jq .
else
  echo "Skip: test_image.jpg not found"
fi
echo

echo "=== Done ==="
