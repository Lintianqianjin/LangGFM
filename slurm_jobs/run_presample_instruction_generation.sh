#!/usr/bin/env bash

mkdir -p slurm_logs
mkdir -p slurm_scripts

# 第一个参数作为 dataset_name
DATASET_NAME="$1"

# 若未传入参数，给出提示并退出
if [ -z "$DATASET_NAME" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

# 生成临时 SLURM 脚本
cat <<EOF > slurm_scripts/generate_instruction_${DATASET_NAME}.slurm
#!/bin/bash
#SBATCH --job-name=insgen-${DATASET_NAME}
#SBATCH --output=slurm_logs/insgen_${DATASET_NAME}.out
#SBATCH --error=slurm_logs/insgen_${DATASET_NAME}.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=15:00:00

source ~/softwares/anaconda3/bin/activate
conda activate GFM

cd ~/projects/LangGFM

# 运行 Python 脚本
python scripts/generate_instruction_dataset.py --job_path experiments/benchmark/${DATASET_NAME}/pre_selection/
EOF

# 提交作业
sbatch slurm_scripts/generate_instruction_${DATASET_NAME}.slurm
