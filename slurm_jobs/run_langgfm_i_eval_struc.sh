#!/usr/bin/env bash

# # 第一个参数作为 dataset
# DATASET="$1"

# # 若未传入 dataset，给出提示并退出
# if [ -z "$DATASET" ]; then
#   echo "Usage: $0 <dataset>"
#   exit 1
# fi
job_name="i_eval_struc"

# Ensure required directories exist
mkdir -p slurm_scripts slurm_logs

# 这里通过 cat 的方式生成一个临时 SLURM 脚本
cat <<EOF > slurm_scripts/temp_langgfm_${job_name}.slurm
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=slurm_logs/${job_name}.out
#SBATCH --error=slurm_logs/${job_name}.err
#SBATCH --gres=gpu:1                   # Request GPUs (A100 or H100)
#SBATCH --constraint="A100|H100"        # Allow both A100 and H100 GPUs
#SBATCH --cpus-per-task=8              # Request CPUs
#SBATCH --mem=64G                       # Request memory
#SBATCH --time=12:00:00                  # Maximum runtime

# Load necessary modules (if required)
module load cuda  # Adjust based on the actual CUDA version

# Activate the Conda environment
source ~/miniconda/bin/activate
conda activate GFM

# Change to the working directory
cd ~/projects/LangGFM

# Execute Python scripts
python scripts/batch_eval_structure.py


EOF

# 使用临时脚本提交作业
sbatch slurm_scripts/temp_langgfm_${job_name}.slurm
