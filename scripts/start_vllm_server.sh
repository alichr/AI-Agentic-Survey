#!/bin/bash
# Start vLLM server for metadata extraction (Qwen3 Instruct model)
# Requires: pip install vllm>=0.8.5
#
# Model: Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, 30.5B total / 3.3B active)
# - FP8 quantized for memory efficiency
# - Non-thinking instruct model (direct responses, no <think> tags)
# - Native context: 262,144 tokens (configured to 64,000 by default)

set -e

MODEL_NAME="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
TP_SIZE="${VLLM_TP_SIZE:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-64000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"

echo "Starting vLLM server..."
echo "  Model: ${MODEL_NAME}"
echo "  Tensor Parallel Size: ${TP_SIZE}"
echo "  Max Model Length: ${MAX_MODEL_LEN}"
echo "  GPU Memory Utilization: ${GPU_UTIL}"
echo "  Serving on: ${HOST}:${PORT}"

vllm serve "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --dtype auto
