import os
import sys
import random
import json
import numpy as np
import networkx as nx
import torch
import math

from tqdm import tqdm
from abc import ABC, abstractmethod
from networkx.readwrite import json_graph
# from dgl.data import MiniGCDataset
# from torch_geometric.utils import to_networkx

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
from langgfm.utils.random_control import set_seed


class StructuralTaskDatasetBuilder(ABC):
    """
    note that structural tasks includes:
    synthetic graph generation tasks, and MiniGC.
    """
    def __init__(self,seed=42):
        self._load_config()
        set_seed(seed)

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '../../configs/structural_task_generation.yaml')
        self.config = load_yaml(config_path)

    @abstractmethod
    def _generate_graphs_and_labels(self, config, label_function, task=None) -> tuple:
        "graphs, labels, splits = self._generate_graphs_and_labels(task_config, label_function, task)"
        pass

    @abstractmethod
    def _get_label_function(self, task) -> callable:
        pass

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


class SyntheticGraphDatasetBuilder(StructuralTaskDatasetBuilder):
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
    def __init__(self):
        self.build_graph_methods = {
            "random": create_random_graph,
            "bipartite": create_random_bipartite_graph,
            "node_weights": create_random_graph_node_weights,
            "smaller": create_smaller_graph,
            "topology": create_topology_graph
        }
        super().__init__()

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


class GraphStructureDetectionBuilder(StructuralTaskDatasetBuilder):
    """
    i.e MiniGC
    """
    def __init__(self, seed=42):
        super().__init__()
        self.task = 'graph_structure_detection'
        self.num_graphs = self.config[self.task]['train_size'] \
            + self.config[self.task]['val_size'] \
            + self.config[self.task]['test_size']
        self.min_num_v = self.config[self.task]['min_nodes']
        self.max_num_v = self.config[self.task]['max_nodes']
        self.graphs = []
        self.labels = []


    def _generate_graphs_and_labels(self, config, label_function, task=None) -> tuple:
        "graphs, labels, splits = self._generate_graphs_and_labels(task_config, label_function, task)"
        self._gen_cycle(self.num_graphs // 8)
        self._gen_star(self.num_graphs // 8)
        self._gen_wheel(self.num_graphs // 8)
        self._gen_lollipop(self.num_graphs // 8)
        self._gen_hypercube(self.num_graphs // 8)
        self._gen_grid(self.num_graphs // 8)
        self._gen_clique(self.num_graphs // 8)
        self._gen_circular_ladder(self.num_graphs - len(self.graphs))
        # preprocess
        graphs = [json_graph.node_link_data(graph) for graph in self.graphs]
        labels = [label_function(label) for label in self.labels]
        splits = generate_splits_mask(
            train_size=config['train_size'],
            val_size=config['val_size'],
            test_size=config['test_size']
        )
        return graphs, labels, splits

    def _gen_cycle(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.cycle_graph(num_v)
            self.graphs.append(g)
            self.labels.append(0)

    def _gen_star(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            # nx.star_graph(N) gives a star graph with N+1 nodes
            g = nx.star_graph(num_v - 1)
            self.graphs.append(g)
            self.labels.append(1)

    def _gen_wheel(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.wheel_graph(num_v)
            self.graphs.append(g)
            self.labels.append(2)

    def _gen_lollipop(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            path_len = np.random.randint(2, num_v // 2)
            g = nx.lollipop_graph(m=num_v - path_len, n=path_len)
            self.graphs.append(g)
            self.labels.append(3)

    def _gen_hypercube(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.hypercube_graph(int(math.log(num_v, 2)))
            g = nx.convert_node_labels_to_integers(g)
            self.graphs.append(g)
            self.labels.append(4)

    def _gen_grid(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            assert num_v >= 4, (
                "We require a grid graph to contain at least two "
                "rows and two columns, thus 4 nodes, got {:d} "
                "nodes".format(num_v)
            )
            n_rows = np.random.randint(2, num_v // 2)
            n_cols = num_v // n_rows
            g = nx.grid_graph([n_rows, n_cols])
            g = nx.convert_node_labels_to_integers(g)
            self.graphs.append(g)
            self.labels.append(5)

    def _gen_clique(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.complete_graph(num_v)
            self.graphs.append(g)
            self.labels.append(6)

    def _gen_circular_ladder(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.circular_ladder_graph(num_v // 2)
            self.graphs.append(g)
            self.labels.append(7)

    def _get_label_function(self, task) -> callable:
        if task == 'graph_structure_detection':
            return lambda label: self._map_label(label)
    
    def _map_label(self, label):
        class_mapping = {
            0 : 'cycle graph',
            1 : 'star graph',
            2 : 'wheel graph',
            3 : 'lollipop graph',
            4 : 'hypercube graph',
            5 : 'grid graph',
            6 : 'clique graph',
            7 : 'circular ladder graph'
        }
        return (class_mapping[label], ())


if __name__ == "__main__":
    dataset_builder = GraphStructureDetectionBuilder()
    dataset_builder.build_dataset(task=dataset_builder.task)


    # dataset_builder._load_config()  # Make sure config is loaded

    # tasks = [
    #     'node_counting', 'edge_counting', 'node_attribute_retrieval','edge_attribute_retrieval', 
    #     'degree_counting', 'edge_existence',
    #     'connectivity', 'shortest_path', 'cycle_checking',
    #     'hamilton_path', 'graph_automorphic'
    # ]
    
    # print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    # for task in tasks:
        # if task == "connectivity":
        #     print(f"Building dataset for task: {task}")
            # dataset_builder.build_dataset(task)
    # configs = dataset_builder.config[task]
    
    # data_path = os.path.join(os.path.dirname(__file__), configs['file_path'])
    # dataset = torch.load(data_path)
    # print(dataset['graphs'][0])
    # print(dataset['labels'][0])

    
    # change the origin labels to the new labels with the format of (label, ())
#     labels = [label[0] for label in dataset['labels']]
#     print(f"Label distribution for task {task}: {dict(zip(*np.unique(labels, return_counts=True)))}")

# # # add random splits for graph_structure_detection
# # if task == "graph_structure_detection":
# #     print(f"Building dataset splits for task: {task}")
# #     add_random_splits(500,100,200,"graph_structure_detection")
