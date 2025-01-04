import os
import yaml
import json
import argparse
import networkx as nx
import random
import torch

from random import randint, sample, shuffle
from itertools import combinations, permutations, product


def generate_splits_mask(train_ratio=None, val_ratio=None, test_ratio=None, total_size=None, train_size=None, val_size=None, test_size=None):
    if total_size is None:
        total_size = train_size + val_size + test_size

    if train_ratio is not None and val_ratio is not None and test_ratio is not None:
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1.")
        if not (train_ratio + val_ratio + test_ratio == 1):
            raise ValueError("Ratios must sum to 1.")
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

    # Initialize masks
    train_mask = torch.zeros(total_size, dtype=torch.bool)
    val_mask = torch.zeros(total_size, dtype=torch.bool)
    test_mask = torch.zeros(total_size, dtype=torch.bool)

    # Generate random indices for train, val, test splits
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Assign True to the corresponding indices in the masks
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)


def build_args():
    parser = argparse.ArgumentParser(description='GraphReasoner')
    parser.add_argument('--task', type=str, default='cycle_train', help='task to perform')
    return parser.parse_args()


def create_random_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weight=False, directed=False):
    if not 0 <= max_sparsity <= 1 or not 0 <= min_sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_nodes = randint(min_nodes, max_nodes)
    # print(f"{num_nodes=}")
    # print(f"{len(test_samples)=}")

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(num_nodes)) 
    
    max_possible_edges = num_nodes * (num_nodes - 1) // 2

    lower_limit_edges = int(min_sparsity * max_possible_edges)
    upper_limit_edges = int(max_sparsity * max_possible_edges)

    num_edges_to_add = randint(lower_limit_edges, upper_limit_edges)
    # print(f"{num_edges_to_add=}")

    if num_edges_to_add == 0:
        num_edges_to_add = 1
    
    all_possible_edges = list(combinations(range(num_nodes), 2))
    edges_to_add = sample(all_possible_edges, num_edges_to_add) if num_edges_to_add > 0 else []
    
    if weight:
        edges_with_weights = [(u, v, randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        edges_with_weights = [(u, v, 0) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    
    return G


def create_random_bipartite_graph(min_nodes, max_nodes, max_edges, min_sparsity, max_sparsity, weight=False, directed=False):
    if not 0 <= max_sparsity <= 1 or not 0 <= min_sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")

    num_nodes = random.randint(min_nodes, max_nodes)
    
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    split_index = random.randint(1, num_nodes - 1)
    nodes_set_1 = nodes[:split_index]
    nodes_set_2 = nodes[split_index:]
    
    max_possible_edges = len(nodes_set_1) * len(nodes_set_2)

    lower_limit_edges = int(min_sparsity * max_possible_edges)
    upper_limit_edges = int(max_sparsity * max_possible_edges)
    
    num_edges_to_add = random.randint(lower_limit_edges, min(upper_limit_edges, max_edges))
    if num_edges_to_add == 0:
        num_edges_to_add = 1

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)

    all_possible_edges = list(product(nodes_set_1, nodes_set_2))
    random.shuffle(all_possible_edges)  

    edges_to_add = all_possible_edges[:num_edges_to_add]
    if weight:
        edges_with_weights = [(u, v, random.randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        G.add_edges_from(edges_to_add)

    return G


def create_smaller_graph(min_nodes, max_nodes, max_edges, min_sparsity, max_sparsity, weight=False, directed=True):
    if not 0 <= max_sparsity <= 1 or not 0 <= min_sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_nodes = max(randint(min_nodes, max_nodes)/2, 3)
    # num_nodes = min(num_nodes, 10)
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))     
    max_possible_edges = num_nodes * (num_nodes - 1) // 2

    lower_limit_edges = int(min_sparsity * max_possible_edges)
    upper_limit_edges = int(max_sparsity * max_possible_edges)
    
    num_edges_to_add = randint(lower_limit_edges, min(upper_limit_edges, max_edges))/2

    if num_edges_to_add == 0:
        num_edges_to_add = 1
    
    all_possible_edges = list(combinations(range(int(num_nodes)), 2))
    edges_to_add = sample(all_possible_edges, int(num_edges_to_add)) if num_edges_to_add > 0 else []
    
    if weight:
        edges_with_weights = [(u, v, randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        edges_with_weights = [(u, v, 0) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    
    return G


def create_random_graph_node_weights(min_nodes, max_nodes, min_sparsity, max_sparsity, directed=True):
    if not 0 <= max_sparsity <= 1 or not 0 <= min_sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")

    num_nodes = randint(min_nodes, max_nodes)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes with random weights
    for node in range(num_nodes):
        G.add_node(node, weight=randint(1, 10))

    max_possible_edges = num_nodes * (num_nodes - 1) if directed else num_nodes * (num_nodes - 1) // 2

    lower_limit_edges = int(min_sparsity * max_possible_edges)
    upper_limit_edges = int(max_sparsity * max_possible_edges)

    num_edges_to_add = randint(lower_limit_edges, upper_limit_edges)

    if num_edges_to_add == 0:
        num_edges_to_add = 1

    all_possible_edges = list(permutations(range(num_nodes), 2)) if directed else list(combinations(range(num_nodes), 2))
    edges_to_add = sample(all_possible_edges, num_edges_to_add) if num_edges_to_add > 0 else []

    G.add_edges_from(edges_to_add)

    return G


def create_topology_graph(min_nodes, max_nodes, min_sparsity, max_sparsity, weight=False, directed=True):
    if not directed:
        raise ValueError("The graph must be directed to ensure a unique topological path.")
    
    num_nodes = randint(min_nodes, max_nodes)
    G = nx.DiGraph()
    
    nodes = list(range(num_nodes))
    shuffle(nodes)
    G.add_nodes_from(nodes)
    
    for i in range(num_nodes - 1):
        G.add_edge(nodes[i], nodes[i + 1], weight=randint(1, 10) if weight else 0)
    
    max_possible_additional_edges = num_nodes * (num_nodes - 1) - (num_nodes - 1)
    
    target_min_edges = int(min_sparsity * max_possible_additional_edges)
    target_max_edges = int(max_sparsity * max_possible_additional_edges)
    
    target_max_edges = min(target_max_edges, max_possible_additional_edges)
    
    num_additional_edges = randint(target_min_edges, target_max_edges)
    
    possible_additional_edges = [(u, v) for u, v in permutations(nodes, 2) if not G.has_edge(u, v)]
    
    additional_edges = sample(possible_additional_edges, num_additional_edges)
    
    for u, v in additional_edges:
        if not nx.has_path(G, v, u): 
            G.add_edge(u, v, weight=randint(1, 10) if weight else 0)
    
    return G


def resave_structure_graphs(task):
    old_dataset = torch.load(f"./RandomGraph/node_size_counting.pt",)
    train_mask, val_mask, test_mask = dataset['train_mask'],dataset['val_mask'],dataset['test_mask']

    dataset = torch.load(os.path.join(os.path.dirname(__file__),'../../../../data/structure_graphs.pt'))
    graphs = dataset['graphs']


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