import os
import sys
import random
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from networkx.readwrite import json_graph
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils import (
    create_random_graph,
    create_random_bipartite_graph, 
    create_random_graph_node_weights, 
    create_smaller_graph, 
    create_topology_graph,
    generate_splits_mask
)
from langgfm.utils.io import load_yaml, safe_mkdir


class SyntheticGraphDatasetBuilder:
    """
    note that structural tasks includes:
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
        - graph automorphic
        - graph structure detection (not included here, complemented by dataset *MiniGC*)
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.build_graph_methods = {
            "random": create_random_graph,
            "bipartite": create_random_bipartite_graph,
            "node_weights": create_random_graph_node_weights,
            "smaller": create_smaller_graph,
            "topology": create_topology_graph
        }
    
    def _load_config(self):
        print(f"Loading config from {self.config_path}")
        return load_yaml(self.config_path)
    
    def _generate_graphs_and_labels(self, config, label_function, task=None):
        total_size = config['train_size'] + config['val_size'] + config['test_size']
        valid = 0
        try_ = 0
        graphs, labels = [], []
        pbar = tqdm(total=total_size)
        
        while valid < total_size:
            try_ += 1
            random_graph = self.build_graph_methods[config['graph_type']](
                config["min_nodes"],
                config["max_nodes"],
                config["min_sparsity"],
                config["max_sparsity"],
                config['is_weighted'],
                config['is_directed']
            )
            if (task == "connectivity") ^ nx.is_connected(random_graph):
                graphs.append(json_graph.node_link_data(random_graph))
                labels.append(label_function(random_graph))
                valid += 1
                pbar.update(1)
        pbar.close()
        splits = generate_splits_mask(
            train_size=config['train_size'],
            val_size=config['val_size'],
            test_size=config['test_size']
        )
        return graphs, labels, splits

    def build_dataset(self, task):
        if task not in self.config:
            raise ValueError(f"Task {task} is not supported in the configuration file.")

        task_config = self.config[task]
        print(f"Building dataset for task: {task}:\n {task_config}")
        label_function = self._get_label_function(task)

        graphs, labels, splits = self._generate_graphs_and_labels(task_config, label_function, task)

        self._save_dataset(task, graphs, labels, task_config)
        self._save_splits(task, splits)
        print(f"{task} dataset saved successfully.")
    
    def _get_label_function(self, task):
        if task == 'node_counting':
            return lambda graph: (graph.number_of_nodes(), ())
        elif task == 'edge_counting':
            return lambda graph: (graph.number_of_edges(), ())
        elif task == 'node_attribute_retrieval':
            return lambda graph: self._retrieve_random_node_attribute(graph)
        elif task == 'edge_attribute_retrieval':
            return lambda graph: self._retrieve_random_edge_attribute(graph)
        elif task == 'degree_counting':
            return lambda graph: self._retrieve_node_degree(graph)
        elif task == 'edge_existence':
            return lambda graph: self._retrieve_edge_existence(graph)
        elif task == 'connectivity':
            # check whether two nodes are connected
            return lambda graph: self._retrieve_connectivity(graph)
        elif task == 'shortest_path':
            return lambda graph: self._retrieve_shortest_path(graph)
        elif task == 'cycle_checking':
            return lambda graph: self._retrieve_cycle_checking(graph)
        else:
            raise ValueError(f"Task {task} is not implemented.")
    
    def _retrieve_random_node_attribute(self, graph):
        label_node = random.choice(list(graph))
        return (graph.nodes[label_node]['weight'], ('weight', label_node))

    def _retrieve_random_edge_attribute(self, graph):
        sample_edge = random.choice(list(graph.edges(data=True)))
        return (sample_edge[2], ('weight', sample_edge[0], sample_edge[1]))
    
    def _retrieve_node_degree(self, graph):
        sample_node = random.choice(list(graph.nodes))
        return (graph.degree(sample_node), (sample_node))
    
    def _retrieve_edge_existence(self, graph):
        sample_nodes = random.sample(list(graph.nodes), 2)
        return (graph.has_edge(sample_nodes[0], sample_nodes[1]), sample_nodes)
    
    # def _retrieve_connectivity(self, graph):
    #     sample_nodes = random.sample(list(graph.nodes), 2)

    #     return (nx.has_path(graph, sample_nodes[0], sample_nodes[1]), sample_nodes)
    
    def _retrieve_connectivity(self, graph):
        
        def find_disconnected_pair(graph):
            # Precompute connected components
            connected_components = list(nx.connected_components(graph))
            
            # If the graph has only one connected component, no disconnected pair exists
            if len(connected_components) <= 1:
                return None  # No disconnected pairs available
            
            # Randomly select two distinct components
            component1, component2 = random.sample(connected_components, 2)
            
            # Randomly select one node from each component
            node1 = random.choice(list(component1))
            node2 = random.choice(list(component2))
            
            return (False, [node1, node2])
        
        choice_by_god = random.choice([True, False])
        pair = None
        if choice_by_god:
            while pair is None:
                sample_nodes = random.sample(list(graph.nodes), 2)
                if nx.has_path(graph, sample_nodes[0], sample_nodes[1]):
                    pair = (True, sample_nodes)
        else:
            pair = find_disconnected_pair(graph)

        return pair


    def _retrieve_shortest_path(self, graph):
        sample_nodes = random.sample(list(graph.nodes), 2)
        return (nx.shortest_path_length(graph, sample_nodes[0], sample_nodes[1]), sample_nodes)
    
    def _retrieve_cycle_checking(self, graph):
        return (True if nx.cycle_basis(graph) else False, )
    
    def _save_dataset(self, task, graphs, labels, task_config):
        dataset = {'graphs': graphs, 'labels': labels}
        dataset_path = os.path.join(os.path.dirname(__file__), task_config['file_path'])
        directory = os.path.dirname(dataset_path)
        safe_mkdir(directory)
        # check the distribution of the labels
        if task_config['task_type'] == 'cls':
            labels = [label[0] for label in labels]
            print(f"Label distribution for task {task}: {dict(zip(*np.unique(labels, return_counts=True)))}")
        torch.save(dataset, dataset_path)
    
    def _save_splits(self, task, splits):
        train_mask, val_mask, test_mask = splits
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




if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), '../../configs/synthetic_graph_generation.yaml')
    dataset_builder = SyntheticGraphDatasetBuilder(config_file_path)

    tasks = [
        'node_counting', 'edge_counting', 'node_attribute_retrieval','edge_attribute_retrieval', 
        'degree_counting', 'edge_existence',
        'connectivity', 'shortest_path', 'cycle_checking',
        'hamilton_path', 'graph_automorphic'
    ]
    
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    for task in tasks:
        if task == "node_counting":
        #     print(f"Building dataset for task: {task}")
            # dataset_builder.build_dataset(task)
            configs = dataset_builder.config[task]
            
            data_path = os.path.join(os.path.dirname(__file__), configs['file_path'])
            dataset = torch.load(data_path)
            print(dataset['graphs'][0])
            print(dataset['labels'][0])

            
            # change the origin labels to the new labels with the format of (label, ())
        #     labels = [label[0] for label in dataset['labels']]
        #     print(f"Label distribution for task {task}: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # # # add random splits for graph_structure_detection
        # # if task == "graph_structure_detection":
        # #     print(f"Building dataset splits for task: {task}")
        # #     add_random_splits(500,100,200,"graph_structure_detection")
