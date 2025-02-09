#!/usr/bin/env bash

mkdir -p slurm_logs
mkdir -p slurm_scripts

# 第一个参数作为 dataset
DATASET="$1"
# 第二个参数作为 model_name_or_path
MODEL="$2"

# 若未传入 dataset 或 model_name_or_path，给出提示并退出
if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
  echo "Usage: $0 <dataset> <model_name_or_path>"
  exit 1
fi

# 处理 MODEL 变量以创建安全的文件名（替换 / 为 _）
SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/\//_/g')

# 生成 SLURM 脚本的文件名
SLURM_SCRIPT="slurm_scripts/temp_langgfm_i_${DATASET}_${SAFE_MODEL_NAME}.slurm"

# 生成 SLURM 脚本
cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=i-${DATASET}-${SAFE_MODEL_NAME}
#SBATCH --output=slurm_logs/langgfm_i_${DATASET}_${SAFE_MODEL_NAME}.out
#SBATCH --error=slurm_logs/langgfm_i_${DATASET}_${SAFE_MODEL_NAME}.err
#SBATCH --gres=gpu:2                     # Request GPUs (A100 or H100)
#SBATCH --constraint="A100|H100"         # Allow both A100 and H100 GPUs
#SBATCH --cpus-per-task=16               # Request CPUs
#SBATCH --mem=64G                        # Request memory
#SBATCH --time=12:00:00                  # Maximum runtime

# Load necessary modules (if required)
module load cuda  # Adjust based on the actual CUDA version

# Activate the Conda environment
source ~/softwares/anaconda3/bin/activate
conda activate GFM

# Change to the working directory
cd ~/projects/LangGFM

# Execute Python script with the provided dataset and model
python scripts/training.py --train_dir experiments/langgfm_i/$DATASET/train_800 --eval_dir experiments/langgfm_i/$DATASET/test_200 --model_name_or_path $MODEL --lora_rank 64 --lora_alpha 256 --lora_dropout 0. --use_rslora True --learning_rate 2.0e-5 --batch_size 64 --num_train_epochs 50 --warmup_ratio 0.2 --eval_steps 25 --save_steps 25

EOF

# 提交 SLURM 任务
sbatch "$SLURM_SCRIPT"
