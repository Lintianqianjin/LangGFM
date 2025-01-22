#!/usr/bin/env bash

# 第一个参数作为 dataset
DATASET="$1"

# 若未传入 dataset，给出提示并退出
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi

# 这里通过 cat 的方式生成一个临时 SLURM 脚本
cat <<EOF > temp_${DATASET}.slurm
#!/bin/bash
#SBATCH --job-name=${DATASET}
#SBATCH --output=${DATASET}.out
#SBATCH --error=${DATASET}.err
#SBATCH --partition=short                 # Partition name, ensure this supports A100 or H100 GPUs
#SBATCH --gres=gpu:1                     # Request 2 GPUs (A100 or H100)
#SBATCH --constraint="A100|H100"        # Allow both A100 and H100 GPUs
#SBATCH --cpus-per-task=32               # Request 32 CPUs
#SBATCH --mem=128G                       # Request 128GB of memory
#SBATCH --time=12:00:00                  # Maximum runtime


# Load necessary modules (if required)
module load cuda  # Adjust based on the actual CUDA version

# Activate the Conda environment
source ~/softwares/anaconda3/bin/activate
conda activate GFM

# Change to the working directory
cd ~/projects/LangGFM

# Execute Python scripts
python scripts/generate_instruction_dataset.py --job_path experiments/$DATASET/train_800
python scripts/generate_instruction_dataset.py --job_path experiments/$DATASET/test_200
python scripts/training.py --train_dir experiments/$DATASET/train_800/ --eval_dir experiments/$DATASET/test_200/ --lora_rank 64 --lora_alpha 128


EOF

# 使用临时脚本提交作业
sbatch temp_${DATASET}.slurm
