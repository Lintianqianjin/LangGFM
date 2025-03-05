#!/bin/bash
set -euo pipefail

# 定义训练目录数组
TRAIN_DIRS=(
  "experiments/langgfm_j/train_800/ogbn_arxiv"
  "experiments/langgfm_j/train_800/wikics"
  "experiments/langgfm_j/train_800/twitch"
  "experiments/langgfm_j/train_800/re_europe"
  "experiments/langgfm_j/train_800/oag_scholar_interest"
  "experiments/langgfm_j/train_800/fb15k237"
  "experiments/langgfm_j/train_800/movielens1m"
  "experiments/langgfm_j/train_800/ogbl_vessel"
  "experiments/langgfm_j/train_800/stack_elec"
  "experiments/langgfm_j/train_800/yelp_review"
  "experiments/langgfm_j/train_800/fingerprint"
  "experiments/langgfm_j/train_800/explagraphs"
  "experiments/langgfm_j/train_800/bace"
  "experiments/langgfm_j/train_800/esol"
  "experiments/langgfm_j/train_800/chebi20"
  "experiments/langgfm_j/train_800/node_counting"
  "experiments/langgfm_j/train_800/edge_counting"
  "experiments/langgfm_j/train_800/node_attribute_retrieval"
  "experiments/langgfm_j/train_800/edge_attribute_retrieval"
  "experiments/langgfm_j/train_800/edge_existence"
  "experiments/langgfm_j/train_800/degree_counting"
  "experiments/langgfm_j/train_800/connectivity"
  "experiments/langgfm_j/train_800/shortest_path"
  "experiments/langgfm_j/train_800/cycle_checking"
  "experiments/langgfm_j/train_800/hamilton_path"
  "experiments/langgfm_j/train_800/graph_automorphic"
  "experiments/langgfm_j/train_800/graph_structure_detection"
)

# 定义评估目录数组
EVAL_DIRS=(
  "experiments/langgfm_i/ogbn_arxiv/test_200"
  "experiments/langgfm_i/wikics/test_200"
  "experiments/langgfm_i/twitch/test_200"
  "experiments/langgfm_i/re_europe/test_200"
  "experiments/langgfm_i/oag_scholar_interest/test_200"
  "experiments/langgfm_i/fb15k237/test_200"
  "experiments/langgfm_i/movielens1m/test_200"
  "experiments/langgfm_i/ogbl_vessel/test_200"
  "experiments/langgfm_i/stack_elec/test_200"
  "experiments/langgfm_i/yelp_review/test_200"
  "experiments/langgfm_i/fingerprint/test_200"
  "experiments/langgfm_i/explagraphs/test_200"
  "experiments/langgfm_i/bace/test_200"
  "experiments/langgfm_i/esol/test_200"
  "experiments/langgfm_i/chebi20/test_200"
  "experiments/langgfm_i/node_counting/test"
  "experiments/langgfm_i/edge_counting/test"
  "experiments/langgfm_i/node_attribute_retrieval/test"
  "experiments/langgfm_i/edge_attribute_retrieval/test"
  "experiments/langgfm_i/edge_existence/test"
  "experiments/langgfm_i/degree_counting/test"
  "experiments/langgfm_i/connectivity/test"
  "experiments/langgfm_i/shortest_path/test"
  "experiments/langgfm_i/cycle_checking/test"
  "experiments/langgfm_i/hamilton_path/test"
  "experiments/langgfm_i/graph_automorphic/test"
  "experiments/langgfm_i/graph_structure_detection/test"
)

# 使用 IFS 将数组转换为逗号分隔的字符串
TRAIN_DIRS_STR=$(IFS=, ; echo "${TRAIN_DIRS[*]}")
EVAL_DIRS_STR=$(IFS=, ; echo "${EVAL_DIRS[*]}")

python scripts/training.py \
  --train_dir "$TRAIN_DIRS_STR" \
  --eval_dir "$EVAL_DIRS_STR" \
  --exp_dir "experiments/langgfm_j" \
  --overwrite_cache false \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --lora_rank 256 \
  --lora_alpha 1024 \
  --lora_dropout 0.1 \
  --use_rslora true \
  --learning_rate 2.0e-5 \
  --batch_size 128 \
  --num_train_epochs 20 \
  --warmup_ratio 0.4 \
  --eval_steps 100 \
  --save_steps 100 \
  --run_name "langgfm_joint_everything"
