#!/usr/bin/env bash

mkdir -p slurm_logs
mkdir -p slurm_scripts


# 第一个参数作为 dataset
DATASET="$1"

# 若未传入 dataset，给出提示并退出
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi

# 这里通过 cat 的方式生成一个临时 SLURM 脚本
cat <<EOF > slurm_scripts/temp_openllm_${DATASET}.slurm
#!/bin/bash
#SBATCH --job-name=l-${DATASET}
#SBATCH --output=slurm_logs/openllm_${DATASET}.out
#SBATCH --error=slurm_logs/openllm_${DATASET}.err
#SBATCH --gres=gpu:2                     # Request GPUs (A100 or H100)
#SBATCH --constraint="A100|H100"        # Allow both A100 and H100 GPUs
#SBATCH --cpus-per-task=16               # Request CPUs
#SBATCH --mem=64G                       # Request memory
#SBATCH --time=3:00:00                  # Maximum runtime


# Load necessary modules (if required)
# module load cuda  # Adjust based on the actual CUDA version


# Activate the Conda environment
source ~/softwares/anaconda3/bin/activate
conda activate vllm

# Change to the working directory
cd ~/projects/LangGFM

# Execute Python scripts
python scripts/inference.py --model_name_or_path meta-llama/Llama-3.3-70B-Instruct --dataset experiments__langgfm_i__${DATASET}__test_200 --output_dir experiments/langgfm_i/${DATASET}/test_200/ckpts/openllm/Llama-3.3-70B-Instruct --top_k 1 --temperature 0.
python scripts/inference.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-70B --dataset experiments__langgfm_i__${DATASET}__test_200 --output_dir experiments/langgfm_i/${DATASET}/test_200/ckpts/openllm/DeepSeek-R1-Distill-Llama-70B --top_k 1 --temperature 0.
python scripts/inference.py --model_name_or_path Qwen/Qwen2.5-72B-Instruct --dataset experiments__langgfm_i__${DATASET}__test_200 --output_dir experiments/langgfm_i/${DATASET}/test_200/ckpts/openllm/Qwen2.5-72B-Instruct --top_k 1 --temperature 0.
EOF

# 使用临时脚本提交作业
sbatch slurm_scripts/temp_openllm_${DATASET}.slurm
