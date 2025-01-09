import os
import numpy as np
import json
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.utils.io import load_yaml


def add_random_splits(train_size, val_size, test_size, task):
    train_mask, val_mask, test_mask = generate_splits_mask(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    split_indices = {
        'train': train_mask.nonzero(as_tuple=True)[0].tolist(),
        'val': val_mask.nonzero(as_tuple=True)[0].tolist(),
        'test': test_mask.nonzero(as_tuple=True)[0].tolist(),
    }
    
    split_path = os.path.join(os.path.dirname(__file__), '../../configs/dataset_splits.json')
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            all_splits = json.load(f)
    else:
        all_splits = {}

    all_splits[task] = split_indices
    with open(split_path, 'w') as f:
        json.dump(all_splits, f, indent=4)


def check_label_distribution(task):
    """
    Check the distribution of the labels.
    """
    config_file_path = os.path.join(os.path.dirname(__file__), '../../configs/structural_task_generation.yaml')
    configs = load_yaml(config_file_path)
    task_configs = configs[task]
    data_path = os.path.join(os.path.dirname(__file__), task_configs['file_path'])
    dataset = torch.load(data_path)
    labels = [label[0] for label in dataset['labels']]
    print(f"Label distribution for task {task}: {dict(zip(*np.unique(labels, return_counts=True)))}")


def transform_labels(task):
    """
    Transform the labels to the required format.
    """
    config_file_path = os.path.join(os.path.dirname(__file__), '../../configs/structural_task_generation.yaml')
    configs = load_yaml(config_file_path)
    task_configs = configs[task]
    data_path = os.path.join(os.path.dirname(__file__), task_configs['file_path'])
    dataset = torch.load(data_path) 
    if task in ["node_counting", "edge_counting",
                "graph_automorphic", "hamilton_path", "cycle_checking"]:
    #     print(f"Building dataset for task: {task}")
        # dataset_builder.build_dataset(task)
        print(f"{dataset['labels'][0]=}")
        new_labels = []
        for label in dataset['labels']:
            new_labels.append((label, ()))
    elif task in ["node_attribute_retrieval"]:
        print(f"{dataset['labels'][0]=}")
        new_labels = []
        for label in dataset['labels']:
            new_labels.append((label[2], (label[0], label[1])))
    elif task in ["edge_attribute_retrieval"]:
        print(f"{dataset['labels'][0]=}")
        new_labels = []
        for label in dataset['labels']:
            new_labels.append((label[2]['weight'], (label[0], label[1])))
    elif task in ["degree_counting"]:
        print(f"{dataset['labels'][0]=}")
        new_labels = []
        for label in dataset['labels']:
            new_labels.append((label[1], (label[0],)))
    elif task in ["edge_existence"]:
        print(f"{dataset['labels'][0]=}")
        # return 
        new_labels = []
        for label in dataset['labels']:
            new_labels.append((label[1], label[0]))
    elif task in ["connectivity"]:
        print(f"{dataset['labels'][0]=}")
    elif task in ["shortest_path"]:
        print(f"{dataset['labels'][0]=}")
        new_labels = []
        for label in dataset['labels']:
            new_labels.append((label[2], (label[0], label[1])))
    # elif task in ["cycle_checking"]: 
    #     print(f"{dataset['labels'][0]=}")
    #     return


    dataset['labels'] = new_labels
    print(f"{new_labels[0]=}")
    torch.save(dataset, data_path)
    # save a demo data in json with dataset[0]
    with open('demo.json', 'w') as f:
        f.write(str(dataset['labels'][0]))  


def check_shortest_path(task='shortest_path'):
    config_file_path = os.path.join(os.path.dirname(__file__), '../../configs/structural_task_generation.yaml')
    configs = load_yaml(config_file_path)
    task_configs = configs[task]
    data_path = os.path.join(os.path.dirname(__file__), task_configs['file_path'])
    dataset = torch.load(data_path) 
    for label in dataset['labels']:
        print(label)


if __name__ == "__main__":
    tasks = [
        'node_counting', 'edge_counting', 'node_attribute_retrieval','edge_attribute_retrieval', 
        'degree_counting', 'edge_existence',
        'connectivity', 'shortest_path', 'cycle_checking',
        'hamilton_path', 'graph_automorphic'
    ]
    # transform_labels('node_counting')
    # transform_labels('edge_counting')
    # transform_labels('graph_automorphic')
    # transform_labels('hamilton_path')
    # transform_labels('node_attribute_retrieval')
    # transform_labels('edge_attribute_retrieval')
    # transform_labels('degree_counting')
    # transform_labels('edge_existence')
    # transform_labels('connectivity')
    # transform_labels('shortest_path')
    # transform_labels('cycle_checking')

    # check_label_distribution('cycle_checking')
    # check_label_distribution('hamilton_path')
    # check_label_distribution('graph_automorphic')

    # check_shortest_path(task='shortest_path')

    # restore the splits 


            