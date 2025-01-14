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

from .utils import (
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
    def __init__(self, task=None, seed=42):
        set_seed(seed)
        self.task = task
        self.config = self._load_config()
        self.num_graphs = sum([
            self.config['train_size'], 
            self.config['val_size'], 
            self.config['test_size']
        ])

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '../../configs/structural_task_generation.yaml')
        return load_yaml(config_path)[self.task]

    def build_dataset(self):
        print(f"Building dataset for task: {self.task}:\n {self.config}")
        graphs, labels, splits = self._generate_graphs_labels_splits()
        self._save_dataset(graphs, labels)
        self._save_splits(splits)
        print(f"{self.task} dataset saved successfully.")

    def _save_dataset(self, graphs, labels):
        dataset = {'graphs': graphs, 'labels': labels}
        dataset_path = os.path.join(os.path.dirname(__file__), self.config['file_path'])
        directory = os.path.dirname(dataset_path)
        safe_mkdir(directory)
        # check the distribution of the labels
        if self.config['task_type'] == 'cls':
            self._check_label_distribution(labels)
        torch.save(dataset, dataset_path)
    
    def _save_splits(self, splits):
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

        all_splits[self.task] = split_indices
        with open(split_path, 'w') as f:
            json.dump(all_splits, f, indent=4)

    def _check_label_distribution(self, labels):
        labels = [label[0] for label in labels]
        distribution = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Label distribution for task {self.task}: {distribution}")
        return distribution
    
    @abstractmethod
    def _generate_graphs_labels_splits(self) -> tuple:
        "`_graph_generate_function`, `_get_label` is used."
        pass


class SyntheticDatasetBuilder(StructuralTaskDatasetBuilder):
    """
    note that Synthetic tasks includes:
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
    """
    def __init__(self, task=None, seed=42):
        super().__init__(task, seed)
    
    def _generate_graphs_labels_splits(self) -> tuple:
        
        valid = 0
        try_ = 0
        graphs, labels = [], []

        pbar = tqdm(total=self.num_graphs)
        while valid < self.num_graphs:
            try_ += 1
            random_graph = self._graph_generate_function()(
                self.config["min_nodes"],
                self.config["max_nodes"],
                self.config["min_sparsity"],
                self.config["max_sparsity"],
                self.config['is_weighted'],
                self.config['is_directed']
            )
            if (self.task == "connectivity") ^ nx.is_connected(random_graph):
                graphs.append(json_graph.node_link_data(random_graph))
                labels.append(self._get_label(random_graph))
                valid += 1
                pbar.update(1)
        pbar.close()

        splits = generate_splits_mask(
            train_size=self.config['train_size'],
            val_size=self.config['val_size'],
            test_size=self.config['test_size']
        )
        return graphs, labels, splits

    def _graph_generate_function(self):
        build_graph_methods = {
            "random": create_random_graph,
            "bipartite": create_random_bipartite_graph,
            "node_weights": create_random_graph_node_weights,
            "smaller": create_smaller_graph,
            "topology": create_topology_graph
        }
        return build_graph_methods[self.config['graph_type']]
    
    @abstractmethod
    def _get_label(self) -> tuple:
        pass


class NodeCountingDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self,seed=42):
        task = 'node_counting'
        super().__init__(task, seed)
    
    @classmethod
    def _get_label(cls, graph):
            return (graph.number_of_nodes(), ())
    

class EdgeCountingDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'edge_counting'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        return (graph.number_of_edges(), ())
    

class NodeAttributeRetrievalDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'node_attribute_retrieval'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        label_node = random.choice(list(graph))
        return (graph.nodes[label_node]['weight'], ('weight', label_node))
    

class EdgeAttributeRetrievalDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'edge_attribute_retrieval'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        sample_edge = random.choice(list(graph.edges(data=True)))
        return (sample_edge[2], ('weight', sample_edge[0], sample_edge[1]))
    

class DegreeCountingDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'degree_counting'
        super().__init__(task, seed)

    def _get_label(self, graph):
        sample_node = random.choice(list(graph.nodes))
        return (graph.degree(sample_node), (sample_node))
    

class EdgeExistenceDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'edge_existence'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        sample_nodes = random.sample(list(graph.nodes), 2)
        return (graph.has_edge(sample_nodes[0], sample_nodes[1]), sample_nodes)
    

class ConnectivityDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'connectivity'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
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
    

class ShortestPathDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'shortest_path'
        super().__init__(task, seed)
    
    # TODO
    def _get_label(self, graph):
        sample_nodes = random.sample(list(graph.nodes), 2)
        return (nx.shortest_path_length(graph, sample_nodes[0], sample_nodes[1]), sample_nodes)
    

class CycleCheckingDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'cycle_checking'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        return (True if nx.cycle_basis(graph) else False, ())


class HamiltonPathDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'hamilton_path'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        return (nx.is_hamiltonian(graph), ())
    

class GraphAutomorphicDatasetBuilder(SyntheticDatasetBuilder):

    def __init__(self, seed=42):
        task = 'graph_automorphic'
        super().__init__(task, seed)
    
    def _get_label(self, graph):
        return (nx.is_isomorphic(graph, graph), ())


class GraphStructureDetectionBuilder(StructuralTaskDatasetBuilder):
    """
    i.e MiniGC
    """
    def __init__(self, seed=42):
        task = 'graph_structure_detection'
        super().__init__(task, seed)
        
        self.min_num_v = self.config[self.task]['min_nodes']
        self.max_num_v = self.config[self.task]['max_nodes']
        self.graphs = []
        self.labels = []

    def _generate_graphs_and_labels_splits(self, config, label_function, task=None) -> tuple:
        "graphs, labels, splits = self._generate_graphs_and_labels(task_config, label_function, task)"
        # generate graphs, with labels
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
        labels = [self._map_label(label) for label in self.labels]
        # generate splits
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
    dataset_builder = ShortestPathDatasetBuilder()
    dataset_builder.build_dataset()


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
