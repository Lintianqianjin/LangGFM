#!/usr/bin/env bash

mkdir -p slurm_logs
mkdir -p slurm_scripts

# 第一个参数作为 dataset_name
DATASET_NAME="$1"
SAMPLE_SIZE="$2"

# 若未传入参数，给出提示并退出
if [ -z "$DATASET_NAME" ] || [ -z "$SAMPLE_SIZE" ]; then
  echo "Usage: $0 <dataset_name> <sample_size>"
  exit 1
fi

# 生成临时 SLURM 脚本
cat <<EOF > slurm_scripts/presample_${DATASET_NAME}_${SAMPLE_SIZE}.slurm
#!/bin/bash
#SBATCH --job-name=ps-${DATASET_NAME}-${SAMPLE_SIZE}
#SBATCH --output=slurm_logs/presample_${DATASET_NAME}_${SAMPLE_SIZE}.out
#SBATCH --error=slurm_logs/presample_${DATASET_NAME}_${SAMPLE_SIZE}.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --time=8:00:00

source ~/softwares/anaconda3/bin/activate
conda activate GFM

cd ~/projects/LangGFM

python scripts/generate_benchmark_indices.py --dataset_name ${DATASET_NAME} --sample_size ${SAMPLE_SIZE}
python scripts/generate_instruction_dataset.py --job_path experiments/benchmark/${DATASET_NAME}/pre_selection/
python scripts/post_process_benchmark_indices.py ${DATASET_NAME} --max_token_cnt=15000 --train_sizes="[200,400,800,1600,3200]" --test_size=200
EOF


sbatch slurm_scripts/presample_${DATASET_NAME}_${SAMPLE_SIZE}.slurm
