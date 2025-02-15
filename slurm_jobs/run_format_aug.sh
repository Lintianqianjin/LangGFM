# #!/usr/bin/env bash

# mkdir -p slurm_logs
# mkdir -p slurm_scripts


# FORMAT="$1"

# # 
# if [ -z "$FORMAT" ]; then
#   echo "Usage: $0 <format>"
#   exit 1
# fi

# MODEL="Qwen/Qwen2.5-7B-Instruct"
# DATASET="shortest_path"

# # # 处理 MODEL 变量以创建安全的文件名（替换 / 为 _）
# # SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/\//_/g')

# # # 生成 SLURM 脚本的文件名
# # SLURM_SCRIPT="slurm_scripts/format_aug_${FORMAT}_${DATASET}_${SAFE_MODEL_NAME}.slurm"

# # # 生成 SLURM 脚本
# # cat <<EOF > "$SLURM_SCRIPT"
# # #!/bin/bash
# # #SBATCH --job-name=fmt-${FORMAT}-${DATASET}-${SAFE_MODEL_NAME}
# # #SBATCH --output=slurm_logs/format_aug_${FORMAT}_${DATASET}_${SAFE_MODEL_NAME}.out
# # #SBATCH --error=slurm_logs/format_aug_${FORMAT}_${DATASET}_${SAFE_MODEL_NAME}.err
# # #SBATCH --gres=gpu:1                     # Request GPUs (A100 or H100)
# # #SBATCH --constraint="A100|H100"         # Allow both A100 and H100 GPUs
# # #SBATCH --cpus-per-task=4               # Request CPUs
# # #SBATCH --mem=64G                        # Request memory
# # #SBATCH --time=12:00:00                  # Maximum runtime

# # # Load necessary modules (if required)
# # module load cuda  # Adjust based on the actual CUDA version

# # # Activate the Conda environment
# # # source ~/softwares/anaconda3/bin/activate
# # source ~/miniconda3/bin/activate
# # conda activate GFM

# # # Change to the working directory
# # cd ~/projects/LangGFM

# # # Execute Python script with the provided dataset and model
# # python scripts/generate_instruction_dataset.py --job_path experiments/format_aug/$DATASET/$FORMAT/train
# # python scripts/generate_instruction_dataset.py --job_path experiments/format_aug/$DATASET/$FORMAT/test
# # python scripts/training.py --train_dir experiments/format_aug/$DATASET/$FORMAT/train --eval_dir experiments/format_aug/$DATASET/$FORMAT/test --model_name_or_path $MODEL --lora_rank 64 --lora_alpha 256 --lora_dropout 0. --use_rslora True --learning_rate 2.0e-5 --batch_size 64 --num_train_epochs 20 --warmup_ratio 0.5 --eval_steps 25 --save_steps 25

# # EOF

# # # 提交 SLURM 任务
# # sbatch "$SLURM_SCRIPT"


# python scripts/generate_instruction_dataset.py --job_path experiments/format_aug/$DATASET/$FORMAT/train
# python scripts/generate_instruction_dataset.py --job_path experiments/format_aug/$DATASET/$FORMAT/test
# python scripts/training.py --train_dir experiments/format_aug/$DATASET/$FORMAT/train --eval_dir experiments/format_aug/$DATASET/$FORMAT/test --model_name_or_path $MODEL --lora_rank 64 --lora_alpha 256 --lora_dropout 0. --use_rslora True --learning_rate 2.0e-5 --batch_size 64 --num_train_epochs 20 --warmup_ratio 0.5 --eval_steps 25 --save_steps 25

# joint
python scripts/training.py --train_dir experiments/format_aug/shortest_path/joint/train --eval_dir experiments/format_aug/shortest_path/joint/test --model_name_or_path Qwen/Qwen2.5-7B-Instruct --lora_rank 64 --lora_alpha 256 --lora_dropout 0. --use_rslora True --learning_rate 2.0e-5 --batch_size 64 --num_train_epochs 20 --warmup_ratio 0.5 --eval_steps 25 --save_steps 25
