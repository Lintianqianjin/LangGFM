import os
import json
import random
import fire
from langgfm.data.graph_generator import InputGraphGenerator
from langgfm.utils.random_control import set_seed

def presample_graph_data(dataset_name: str = "ogbn_arxiv", sample_size: int = 10000):
    """
    Sample data from a specified graph dataset and save indices to a JSON file.

    Args:
        dataset_name (str): Name of the dataset.
        sample_size (int): Number of samples to draw.
    """
    generator = InputGraphGenerator.create(dataset_name)
    sample_size = min(sample_size, len(generator.all_samples))
    sampled_items = random.sample(generator.all_samples, sample_size)
    indices = {dataset_name: sampled_items}
    
    folder_path = f"experiments/benchmark/{dataset_name}/pre_selection"
    os.makedirs(folder_path, exist_ok=True)
    json.dump(indices, open(f"{folder_path}/indices.json", "w"), indent=2)
    
    print(f"Sampled {sample_size} items from {dataset_name} and saved to {folder_path}/indices.json")

if __name__ == "__main__":
    set_seed(0)
    fire.Fire(presample_graph_data)
