

import json
import os

from langgfm.utils.io import save_beautiful_json
# from langgfm.data.build_synthetic_graph import *


# for task in StructuralTaskDatasetBuilder.registry:
#     print(f"Building dataset for task: {task}")
#     builder = StructuralTaskDatasetBuilder.create(task)
#     splits = builder._generate_splits()
#     builder._save_splits(splits, split_path='src/langgfm/data/config/splits_structure_tasks.json')
#     print(f"{task} dataset saved successfully.")


structure_splits = json.load(open('src/langgfm/configs/data_splits_structure.json'))
train_structure_splits = {task: structure_splits[task]['train'] for task in structure_splits}
# train_structure_mini_splits = {task: structure_splits[task]['train'][:10] for task in structure_splits}

# with open('experiments/training_v1/indices_structure.json', 'w') as f:
#     json.dump(train_v1_splits, f)

# with open('experiments/training_v1/indices_structure_mini.json', 'w') as f:
    # json.dump(train_v1_mini_splits, f)

save_beautiful_json(train_structure_splits, 'experiments/training_v2/indices_structure.json')
# save_beautiful_json(train_structure_mini_splits, 'experiments/training_v2/indices_structure_mini.json')


# train_v1_semantic_splits = json.load(open('experiments/training_v1/indices_semantic.json'))
# train_v1_semantic_mini_splits = json.load(open('experiments/training_v1/indices_semantic_mini.json'))

# train_v1_splits = {**train_v1_structure_splits, **train_v1_semantic_splits}
# train_v1_mini_splits = {**train_v1_structure_mini_splits, **train_v1_semantic_mini_splits}

# save_beautiful_json(train_v1_splits, 'experiments/training_v1/indices.json')
# save_beautiful_json(train_v1_mini_splits, 'experiments/training_v1/indices_mini.json')
