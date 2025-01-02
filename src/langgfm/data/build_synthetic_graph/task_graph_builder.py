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
    - hamiltonian path
    - graph structure detection
    - graph automorphism
"""
import random
import os
import json

import networkx as nx
import torch
from networkx.readwrite import json_graph
from tqdm import tqdm

from utils import (create_random_bipartite_graph, create_random_graph, create_random_graph_node_weights, create_smaller_graph, create_topology_graph)
from utils import load_yaml, build_args



def build_node_counting_dataset(config):
    
    min_nodes, max_nodes = config["min_nodes"], config["max_nodes"]
    min_sparsity, max_sparsity = config["min_sparsity"], config["max_sparsity"]
    weight, directed = config["is_weighted"], config["is_directed"]

    train_size, val_size, test_size = config['train_size'], config['val_size'], config['test_size']
    total_size = train_size + val_size + test_size

    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)
    
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weight, directed)
        if nx.is_connected(random_graph):
            # note that we convert graph with node_link to save it in json
            graphs.append(json_graph.node_link_data(random_graph))
            labels.append(random_graph.number_of_nodes())

            if valid < train_size:
                train_mask[valid] = True
            elif train_size <= valid < train_size + val_size:
                val_mask[valid] = True
            else:
                test_mask[valid] = True
            
            valid += 1 
            pbar.update(1)
    pbar.close()

    dataset = {
        'graphs': graphs,
        'labels': labels,
    }
    # save the train, val, test split with index list into ../../configs/dataset_splits.json
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
    splits['node_counting'] = split_indices
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=4)

    path = os.path.join(os.path.dirname(__file__),'../../../../data/node_counting_dataset.pt')
    torch.save(dataset, path)
    print(f"node counting dataset saved at {path}.")


def build_edge_counting_dataset(config):
    min_nodes, max_nodes = config["min_nodes"], config["max_nodes"]
    min_sparsity, max_sparsity = config["min_sparsity"], config["max_sparsity"]
    weight, directed = config["is_weighted"], config["is_directed"]

    train_size, val_size, test_size = config['train_size'], config['val_size'], config['test_size']
    total_size = train_size + val_size + test_size

    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)
    
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weight, directed)
        if nx.is_connected(random_graph):
            # note that we convert graph with node_link to save it in json
            graphs.append(json_graph.node_link_data(random_graph))
            # **get the label**
            labels.append(random_graph.number_of_edges())

            if valid < train_size:
                train_mask[valid] = True
            elif train_size <= valid < train_size + val_size:
                val_mask[valid] = True
            else:
                test_mask[valid] = True
            
            valid += 1 
            pbar.update(1)
    pbar.close()

    dataset = {'graphs': graphs,'labels': labels}
    # save the train, val, test split with index list into ../../configs/dataset_splits.json
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
    splits['edge_counting'] = split_indices
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved at {split_path}.")

    path = os.path.join(os.path.dirname(__file__),'../../../../data/edge_counting_dataset.pt')
    torch.save(dataset, path)
    print(f"edge counting dataset saved at {path}.")


def build_node_attribute_retrieval_dataset(config):
    min_nodes, max_nodes = config["min_nodes"], config["max_nodes"]
    min_sparsity, max_sparsity = config["min_sparsity"], config["max_sparsity"]
    weight, directed = config["is_weighted"], config["is_directed"]

    train_size, val_size, test_size = config['train_size'], config['val_size'], config['test_size']
    total_size = train_size + val_size + test_size

    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)
    
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph_node_weights(min_nodes, max_nodes, min_sparsity, max_sparsity, directed)
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            label_node = random.choice(list(random_graph))
            label_node_attribute = random_graph.nodes[label_node]['weight']
            labels.append(('weight', label_node, label_node_attribute))
            if valid < train_size:
                train_mask[valid] = True
            elif train_size <= valid < train_size + val_size:
                val_mask[valid] = True
            else:
                test_mask[valid] = True
            
            valid += 1 
            pbar.update(1)
    pbar.close()

    dataset = {
        'graphs': graphs,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }
    path = os.path.join(os.path.dirname(__file__),'../../../../data/node_attribute_retrieval_dataset.pt')
    torch.save(dataset, path)
    print(f"node_attribute_retrieval dataset saved at {path}.")


def build_edge_attribute_retrieval_dataset(config):
    min_nodes, max_nodes = config["min_nodes"], config["max_nodes"]
    min_sparsity, max_sparsity = config["min_sparsity"], config["max_sparsity"]
    weighted, directed = config["is_weighted"], config["is_directed"]

    train_size, val_size, test_size = config['train_size'], config['val_size'], config['test_size']
    total_size = train_size + val_size + test_size

    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)
    
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        # **generate suitable graphs**: with attributes on edges
        random_graph = create_random_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weighted, directed)
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            # **get the label**
            sample_ = random.choice(list(random_graph.edges(data=True)))
            labels.append(('weight',([sample_[0],sample_[1]]), sample_[2]))

            if valid < train_size:
                train_mask[valid] = True
            elif train_size <= valid < train_size + val_size:
                val_mask[valid] = True
            else:
                test_mask[valid] = True
            valid += 1 
            pbar.update(1)
    pbar.close()

    dataset = {
        'graphs': graphs,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }
    path = os.path.join(os.path.dirname(__file__),'../../../../data/edge_attribute_retrieval_dataset.pt')
    torch.save(dataset, path)
    print(f"edge_attribute_retrieval dataset saved at {path}.")


def build_degree_counting_dataset(config):
    min_nodes, max_nodes = config["min_nodes"], config["max_nodes"]
    min_sparsity, max_sparsity = config["min_sparsity"], config["max_sparsity"]
    weighted, directed = config["is_weighted"], config["is_directed"]

    train_size, val_size, test_size = config['train_size'], config['val_size'], config['test_size']
    total_size = train_size + val_size + test_size

    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)
    
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        # **generate suitable graphs**: with attributes
        random_graph = create_random_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weighted, directed)
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            sample_ = random.choice(list(random_graph.nodes))
            sample_degree = random_graph.degree(sample_)
            labels.append((sample_, sample_degree))

            if valid < train_size:
                train_mask[valid] = True
            elif train_size <= valid < train_size + val_size:
                val_mask[valid] = True
            else:
                test_mask[valid] = True
            valid += 1 
            pbar.update(1)
    pbar.close()

    dataset = {
        'graphs': graphs,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }
    path = os.path.join(os.path.dirname(__file__),'../../../../data/degree_counting_dataset.pt')
    torch.save(dataset, path)
    print(f"degree_counting_dataset saved at {path}.")



def build_edge_existence_dataset(config):
    min_nodes, max_nodes = config["min_nodes"], config["max_nodes"]
    min_sparsity, max_sparsity = config["min_sparsity"], config["max_sparsity"]
    weighted, directed = config["is_weighted"], config["is_directed"]

    train_size, val_size, test_size = config['train_size'], config['val_size'], config['test_size']
    total_size = train_size + val_size + test_size

    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)
    
    valid = 0
    graphs, labels = [], []
    pbar = tqdm(total=total_size)
    while valid < total_size:
        random_graph = create_random_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weighted, directed)
        if nx.is_connected(random_graph):
            graphs.append(json_graph.node_link_data(random_graph))
            sample_query = random.sample(list(random_graph.nodes),2)
            sample_anwer = random_graph.has_edge(sample_query[0], sample_query[1])
            labels.append((sample_query, sample_anwer))

            if valid < train_size:
                train_mask[valid] = True
            elif train_size <= valid < train_size + val_size:
                val_mask[valid] = True
            else:
                test_mask[valid] = True
            valid += 1 
            pbar.update(1)
    pbar.close()

    dataset = {'graphs': graphs,'labels': labels}
    # save the train, val, test split with index list into ../../configs/dataset_splits.json
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
    splits['edge_existence'] = split_indices
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved at {split_path}.")
    
    path = os.path.join(os.path.dirname(__file__),'../../../../data/edge_existence_dataset.pt')
    torch.save(dataset, path)
    print(f"edge_existence_dataset saved at {path}.")


def build_connectivity_dataset(config):
    pass


def main(task):
    resolved_path = os.path.join(os.path.dirname(__file__), '../../configs/synthetic_graph_generation.yaml')
    print(f"loading config from {resolved_path}")
    config = load_yaml(resolved_path)

    if task == 'node_counting':
        build_node_counting_dataset(config['node_counting'])
    elif task == 'edge_counting':
        build_edge_counting_dataset(config['edge_counting'])
    elif task == 'node_attribute_retrieval':
        build_node_attribute_retrieval_dataset(config['node_attribute_retrieval'])
    elif task == 'edge_attribute_retrieval':
        build_edge_attribute_retrieval_dataset(config['edge_attribute_retrieval'])
    elif task == 'degree_counting':
        build_degree_counting_dataset(config['degree_counting'])
    elif task == 'edge_existence':
        build_edge_existence_dataset(config['edge_existence'])
    else:
        print("task not supported.")


if __name__ == "__main__":
    args = build_args()
    args.task = 'edge_existence'
    main(args.task)

