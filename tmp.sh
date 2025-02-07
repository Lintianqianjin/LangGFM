#!/bin/bash

datasets=(
    node_counting
    edge_counting
    node_attribute_retrieval
    edge_attribute_retrieval
    degree_counting
    shortest_path
    cycle_check
    hamilton_path
    graph_automorphic
    graph_structure_detection
    edge_existence
    connectivity
)

for dataset in "${datasets[@]}"; do
    python scripts/generate_instruction_dataset.py experiments/$dataset/test_200/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
    python scripts/generate_instruction_dataset.py experiments/$dataset/train_100/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
    python scripts/generate_instruction_dataset.py experiments/$dataset/train_200/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
    python scripts/generate_instruction_dataset.py experiments/$dataset/train_400/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
    python scripts/generate_instruction_dataset.py experiments/$dataset/train_800/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
    python scripts/generate_instruction_dataset.py experiments/$dataset/train_1600/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
    python scripts/generate_instruction_dataset.py experiments/$dataset/train_3200/ --tokenizer_name_or_path ./models/Llama-3.1-8B-Instruct/
done
