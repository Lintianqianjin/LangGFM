#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

# Run Llama-3.1-8B-Instruct on GPU 0 in the background
# nohup python scripts/batch_eval_structure.py Llama-3.1-8B-Instruct > logs/llama_pipeline.log 2>&1 &

# Run Qwen2.5-7B-Instruct on GPU 1 in the background
nohup python scripts/batch_eval_structure.py Qwen2.5-7B-Instruct > logs/qwen_pipeline.log 2>&1 &

echo "🚀 Pipelines started in the background."
echo "Check logs/llama_pipeline.log and logs/qwen_pipeline.log for logs."
