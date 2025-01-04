import os
import torch
from langgfm.utils.io import load_yaml



def transform_labels(task):
    """
    Transform the labels to the required format.
    """
    config_file_path = os.path.join(os.path.dirname(__file__), '../../configs/synthetic_graph_generation.yaml')
    configs = load_yaml(config_file_path)
    task_configs = configs[task]
    if task in ["node_counting"]:
    #     print(f"Building dataset for task: {task}")
        # dataset_builder.build_dataset(task)
        
        data_path = os.path.join(os.path.dirname(__file__), task_configs['file_path'])
        dataset = torch.load(data_path)
        print(dataset['graphs'][0])
        print(dataset['labels'][0])

if __name__ == "__main__":
    tasks = [
        'node_counting', 'edge_counting', 'node_attribute_retrieval','edge_attribute_retrieval', 
        'degree_counting', 'edge_existence',
        'connectivity', 'shortest_path', 'cycle_checking',
        'hamilton_path', 'graph_automorphic'
    ]
    


            