"""
structural tasks includes:
- structure free
    - graph size: node counting, edge counting
    - attribute retrieval: node attribute retrieval, edge attribute retrieval
- 1-hop 
    - edge exsistence
    - degree counting
- local structure
    - connectivity
    - shortest path
    - cycle checking
- global structure
    - hamilton path
    - graph structure detection
    - graph automorphic
"""
import random
import os
import json

import networkx as nx
import torch
from networkx.readwrite import json_graph
from tqdm import tqdm

from utils import (create_random_bipartite_graph, create_random_graph, create_random_graph_node_weights, create_smaller_graph, create_topology_graph)
from utils import load_yaml, build_args, generate_splits_mask


def build_node_counting_dataset(config):
    total_size = config['train_size'] + config['val_size'] + config['test_size']
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(
            config["min_nodes"],
            config["max_nodes"],
            config["min_sparsity"],
            config["max_sparsity"],
            config['weight'],
            config['is_directed']
        )
        if nx.is_connected(random_graph):
            # note that we convert graph with node_link to save it in json
            graphs.append(json_graph.node_link_data(random_graph))
            labels.append(random_graph.number_of_nodes())
            valid += 1 
            pbar.update(1)
    pbar.close()

    splits = generate_splits_mask(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size']
    )
    return graphs, labels, splits


def build_edge_counting_dataset(config):
    total_size = config['train_size'] + config['val_size'] + config['test_size']
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(
            config["min_nodes"],
            config["max_nodes"],
            config["min_sparsity"],
            config["max_sparsity"],
            config['weight'],
            config['is_directed']
        )
        if nx.is_connected(random_graph):
            # note that we convert graph with node_link to save it in json
            graphs.append(json_graph.node_link_data(random_graph))
            # **get the label**
            labels.append(random_graph.number_of_edges())
            valid += 1 
            pbar.update(1)
    pbar.close()

    splits = generate_splits_mask(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size']
    )
    return graphs, labels, splits


def build_node_attribute_retrieval_dataset(config):
    total_size = config['train_size'] + config['val_size'] + config['test_size']
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(
            config["min_nodes"],
            config["max_nodes"],
            config["min_sparsity"],
            config["max_sparsity"],
            config['weight'],
            config['is_directed']
        )
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            label_node = random.choice(list(random_graph))
            label_node_attribute = random_graph.nodes[label_node]['weight']
            labels.append(('weight', label_node, label_node_attribute))
            valid += 1 
            pbar.update(1)
    pbar.close()

    splits = generate_splits_mask(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size']
    )
    return graphs, labels, splits


def build_edge_attribute_retrieval_dataset(config):
    total_size = config['train_size'] + config['val_size'] + config['test_size']
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(
            config["min_nodes"],
            config["max_nodes"],
            config["min_sparsity"],
            config["max_sparsity"],
            config['weight'],
            config['is_directed']
        )
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            # **get the label**
            sample_ = random.choice(list(random_graph.edges(data=True)))
            labels.append(('weight',([sample_[0],sample_[1]]), sample_[2]))
            valid += 1 
            pbar.update(1)
    pbar.close()

    splits = generate_splits_mask(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size']
    )
    return graphs, labels, splits


def build_degree_counting_dataset(config):
    total_size = config['train_size'] + config['val_size'] + config['test_size']
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        # build the graph
        random_graph = create_random_graph(
            config["min_nodes"],
            config["max_nodes"],
            config["min_sparsity"],
            config["max_sparsity"],
            config['weight'],
            config['is_directed']
        )
        # build the query / label
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            sample_ = random.choice(list(random_graph.nodes))
            sample_degree = random_graph.degree(sample_)
            labels.append((sample_, sample_degree))
            valid += 1 
            pbar.update(1)
    pbar.close()

    splits = generate_splits_mask(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size']
    )
    return graphs, labels, splits


def build_edge_existence_dataset(config):
    total_size = config['train_size'] + config['val_size'] + config['test_size']
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        # build the graph
        random_graph = create_random_graph(
            config["min_nodes"],
            config["max_nodes"],
            config["min_sparsity"],
            config["max_sparsity"],
            config['weight'],
            config['is_directed']
        )
        # build the query / label
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            sample_query = random.sample(list(random_graph.nodes),2)
            sample_anwer = random_graph.has_edge(sample_query[0], sample_query[1])
            labels.append((sample_query, sample_anwer))
            valid += 1 
            pbar.update(1)
    pbar.close()

    splits = generate_splits_mask(
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size']
    )
    return graphs, labels, splits

    
def build_connectivity_dataset(config):
    pass


def pipeline(task):
    resolved_path = os.path.join(os.path.dirname(__file__), '../../configs/synthetic_graph_generation.yaml')
    print(f"loading config from {resolved_path}")
    config = load_yaml(resolved_path)

    if task == 'node_counting':
        graphs, labels, splits = build_node_counting_dataset(config['node_counting'])
    elif task == 'edge_counting':
        graphs, labels, splits = build_edge_counting_dataset(config['edge_counting'])
    elif task == 'node_attribute_retrieval':
        graphs, labels, splits = build_node_attribute_retrieval_dataset(config['node_attribute_retrieval'])
    elif task == 'edge_attribute_retrieval':
        graphs, labels, splits = build_edge_attribute_retrieval_dataset(config['edge_attribute_retrieval'])
    elif task == 'degree_counting':
        graphs, labels, splits = build_degree_counting_dataset(config['degree_counting'])
    elif task == 'edge_existence':
        graphs, labels, splits = build_edge_existence_dataset(config['edge_existence'])
    else:
        print("task not supported.")
    
    # save the dataset 
    dataset = {'graphs': graphs, 'labels': labels}
    torch.save(dataset, os.path.join(os.path.dirname(__file__), config[task]['file_path']))
    print(f"{task} dataset saved.")
    
    # save the splits into dataset_splits.json
    train_mask, val_mask, test_mask = splits
    split_indices = {
        'train': train_mask.nonzero(as_tuple=True)[0].tolist(),
        'val': val_mask.nonzero(as_tuple=True)[0].tolist(),
        'test': test_mask.nonzero(as_tuple=True)[0].tolist(),
    }
    split_path = os.path.join(os.path.dirname(__file__), '../../configs/dataset_splits.json')
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            splits = json.load(f)
    else:
        splits = {}
    splits[task] = split_indices
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=4)


def tmp(task):
    resolved_path = os.path.join(os.path.dirname(__file__), '../../configs/synthetic_graph_generation.yaml')
    print(f"loading config from {resolved_path}")
    task_config = load_yaml(resolved_path)[task]
    # load ugm dataset
    file_path = os.path.join(os.path.dirname(__file__), task_config['file_path'])
    dataset = torch.load(file_path)
    print(dataset)
    # exit()

    # train_mask, val_mask, test_mask = dataset['train_mask'],dataset['val_mask'],dataset['test_mask']
    train_mask, val_mask, test_mask = generate_splits_mask(
        train_size=task_config['train_size'],
        val_size=task_config['val_size'],
        test_size=task_config['test_size']
    )
    split_indices = {
        'train': train_mask.nonzero(as_tuple=True)[0].tolist(),
        'val': val_mask.nonzero(as_tuple=True)[0].tolist(),
        'test': test_mask.nonzero(as_tuple=True)[0].tolist(),
    }
    # save the train, val, test split with index list into ../../configs/dataset_splits.json
    split_path = os.path.join(os.path.dirname(__file__), '../../configs/dataset_splits.json')
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            splits = json.load(f)
    else:
        splits = {}
    splits[task] = split_indices
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits for {task} saved at {split_path}.")
    

if __name__ == "__main__":
    # args = build_args()
    for task in [
        # 'node_counting','edge_counting','node_attribute_retrieval','edge_attribute_retrieval',
        # 'degree_counting', 
        # 'shortest_path',
        # 'graph_automorphic', 
        'hamilton_path',
        ]:
        tmp(task)

