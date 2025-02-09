#!/bin/bash
# Usage: ./run_inference.sh <adapter_name_or_path>
# Example:
# ./run_inference.sh experiments/langgfm_i/ogbn_arxiv/train_800/ckpts/Llama-3.1-8B-Instruct/lora_rank=64/lora_alpha=256/lora_dropout=0.0/learning_rate=2e-05/num_train_epochs=50/warmup_ratio=0.2/batch_size=64/checkpoint-225

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <adapter_name_or_path>"
    exit 1
fi

adapter_path="$1"

# Split the adapter_path by "/" into an array
IFS='/' read -ra parts <<< "$adapter_path"

# Automatically extract model_name_or_path
# Assuming the path contains ".../ckpts/<model_name_or_path>/..."
# In the array, the element immediately following "ckpts" is the model name
model_name=""
for i in "${!parts[@]}"; do
    if [ "${parts[$i]}" == "ckpts" ] && [ $((i+1)) -lt "${#parts[@]}" ]; then
        model_name="${parts[$((i+1))]}"
        break
    fi
done

if [ -z "$model_name" ]; then
    echo "Error: Unable to extract model_name_or_path from the adapter path."
    exit 1
fi

# If the model_name starts with "Llama", prefix it with "meta-llama/"
if [[ "$model_name" == Llama* ]]; then
    model_name="meta-llama/${model_name}"
fi

# Construct the dataset parameter
# Given the input path format:
# experiments/langgfm_i/ogbn_arxiv/train_800/...
# We keep the first three parts and replace the fourth part "train_800" with "test_200",
# then join them with "__"
if [ "${#parts[@]}" -lt 4 ]; then
    echo "Error: The adapter path format is invalid; cannot extract dataset information."
    exit 1
fi
dataset="${parts[0]}__${parts[1]}__${parts[2]}__test_200"

# Construct the output_dir parameter
# Replace "train_800" with "test_200" in adapter_path
output_dir="${adapter_path/train_800/test_200}"

# Print debug information
echo "Parsed parameters:"
echo "  model_name_or_path: ${model_name}"
echo "  adapter_name_or_path: ${adapter_path}"
echo "  dataset: ${dataset}"
echo "  output_dir: ${output_dir}"
echo ""

# Execute the inference command
python scripts/inference.py \
    --model_name_or_path "${model_name}" \
    --adapter_name_or_path "${adapter_path}" \
    --dataset "${dataset}" \
    --output_dir "${output_dir}" \
    --top_k 1 \
    --temperature 0 \
    --template llama3

# python scripts/eval_langgfm_i.py ${output_dir}/predictions.json
# End of file