from langgfm.data.build_synthetic_graph import *
for task in StructuralTaskDatasetBuilder.registry:
    print(f"Building dataset for task: {task}")
    builder = StructuralTaskDatasetBuilder.create(task)
    splits = builder._generate_splits()
    builder._save_splits(splits, split_path='experiments/train/train_v1/splits.json')
    print(f"{task} dataset saved successfully.")