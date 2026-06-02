#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server with the LoRA adapter attached.
#
# Requires a CUDA GPU and `pip install vllm` (see docker/Dockerfile.serve).
# Both the base route and the adapter route are exposed:
#   - base id (e.g. mistralai/Mistral-7B-v0.3)  -> the un-tuned model
#   - <adapter_name> (e.g. mistral-ft)          -> base + LoRA adapter
#
# Usage:  bash serve/launch_vllm.sh configs/mistral7b_qlora.yaml
set -euo pipefail

CONFIG="${1:-configs/mistral7b_qlora.yaml}"

read_cfg() { uv run python -c "from common.config import load_config;c=load_config('$CONFIG');print($1)"; }

BASE_MODEL=$(read_cfg "c.base_model")
ADAPTER_NAME=$(read_cfg "c.serve.adapter_name")
ADAPTER_DIR=$(read_cfg "c.adapter_dir")
MAX_LORA_RANK=$(read_cfg "c.serve.max_lora_rank")
MAX_MODEL_LEN=$(read_cfg "c.serve.max_model_len")
GPU_UTIL=$(read_cfg "c.serve.gpu_memory_utilization")
PORT=$(read_cfg "c.serve.port")
DTYPE=$(read_cfg "c.serve.dtype")

echo "Serving base=$BASE_MODEL  adapter=$ADAPTER_NAME ($ADAPTER_DIR)  port=$PORT"

exec vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "$ADAPTER_NAME=$ADAPTER_DIR" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --port "$PORT"
