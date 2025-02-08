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
cat <<EOF > slurm_scripts/temp_langgfm_i_${DATASET}.slurm
#!/bin/bash
#SBATCH --job-name=i-${DATASET}
#SBATCH --output=slurm_logs/langgfm_i_${DATASET}.out
#SBATCH --error=slurm_logs/langgfm_i_${DATASET}.err
#SBATCH --gres=gpu:2                     # Request GPUs (A100 or H100)
#SBATCH --constraint="A100|H100"        # Allow both A100 and H100 GPUs
#SBATCH --cpus-per-task=16               # Request CPUs
#SBATCH --mem=64G                       # Request memory
#SBATCH --time=12:00:00                  # Maximum runtime


# Load necessary modules (if required)
module load cuda  # Adjust based on the actual CUDA version


# Activate the Conda environment
source ~/softwares/anaconda3/bin/activate
conda activate GFM

# Change to the working directory
cd ~/projects/LangGFM

# Execute Python scripts
python scripts/generate_instruction_dataset.py --job_path experiments/langgfm_i/$DATASET/train
python scripts/generate_instruction_dataset.py --job_path experiments/langgfm_i/$DATASET/test
python scripts/training.py --train_dir experiments/langgfm_i/$DATASET/train --eval_dir experiments/langgfm_i/$DATASET/test --model_name_or_path Qwen/Qwen2.5-7B-Instruct --lora_rank 64 --lora_alpha 256 --lora_dropout 0. --use_rslora True --learning_rate 2.0e-5 --batch_size 64 --num_train_epochs 50 --warmup_ratio 0.4 --eval_steps 25 --save_steps 25


EOF

# 使用临时脚本提交作业
sbatch slurm_scripts/temp_langgfm_i_${DATASET}.slurm
