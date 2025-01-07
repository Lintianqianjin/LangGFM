from abc import ABC, abstractmethod
import os
import torch
import networkx as nx
from networkx.readwrite import json_graph

from .utils.sampling import generate_node_centric_k_hop_subgraph, generate_edge_centric_k_hop_subgraph
from .utils.shuffle_graph import shuffle_nodes_randomly
from langgfm.utils.io import load_yaml

class InputGraphGenerator(ABC):
    """
    Abstract base class for generating NetworkX graph samples 
    from different datasets. Each dataset should implement its
    specific logic by subclassing this class.
    """
    
    registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a subclass with a specific name.
        """
        def decorator(subclass):
            cls.registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Factory method to create an instance of a registered subclass.
        """
        if name not in cls.registry:
            raise ValueError(f"Unknown generator type: {name}. Available types: {list(cls.registry.keys())}")
        return cls.registry[name](*args, **kwargs)
    
    @abstractmethod
    def load_data(self):
        """
        Load the dataset. This method should initialize all necessary
        data structures for the specific dataset.
        
        Example:
            - Reading raw data files (e.g., CSV, JSON).
            - Parsing the dataset into memory.
        """
        pass
    
    @abstractmethod
    def generate_graph(self, sample_id: int) -> nx.Graph:
        """
        Generate a single NetworkX graph for a given sample ID.
        
        Args:
            sample_id (int): The ID of the sample to generate a graph for.
        
        Returns:
            nx.Graph: A NetworkX graph representing the specific sample.
        
        Example:
            - Nodes and edges could be derived from the dataset.
            - Features and labels can be attached to the nodes.
        """
        pass

    def describe(self) -> str:
        """
        Provide a description of the dataset and its generation logic.
        This is an optional utility method for debugging or logging.
        
        Returns:
            str: A textual description of the dataset.
        """
        return "InputGraphGenerator: Abstract class for generating graph samples."
    
    
    
class NodeTaskGraphGenerator(InputGraphGenerator):
    """
    A concrete implementation of InputGraphGenerator to generate graphs
    based on node-centric sampling logic.
    """
    def __init__(self, num_hops=2, sampling=False, neighbor_size: int = None, random_seed: int = None, **kwargs):
        self.num_hops = num_hops
        self.sampling = sampling
        self.neighbor_size = neighbor_size
        self.random_seed = random_seed
        self.graph = None
        
        if self.sampling:
            assert neighbor_size is not None, "neighbor_size should be specified"
            assert random_seed is not None, "random_seed should be specified"

        self.load_data()

    def egograph_sampling(self, sample_id: int) -> nx.Graph:
        '''
        Generate a k-hop subgraph for a given node ID.
        
        Parameters:
            sample_id: The node ID for which the subgraph is generated.
            
        Returns:
            sub_graph_edge_index: The combined edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
            sub_graph_edge_mask: The edge mask of the overall graph for sub_graph_edge_index.
        '''
        sub_graph_edge_index, sub_graph_nodes, sub_graph_edge_mask = generate_node_centric_k_hop_subgraph(
            self.graph, sample_id, self.num_hops, self.neighbor_size, 
            self.random_seed, self.sampling
        )
        
        # raw node index to new node index mapping
        node_mapping = {
            raw_node_idx: new_node_idx 
            for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)
        }
        
        return sub_graph_edge_index, node_mapping, sub_graph_edge_mask
    
    @abstractmethod
    def get_query(self, target_node_idx:int) -> str:
        """
        Get the query for the main task based on the target_node_idx 
        in the networkx graph object."""
        
        pass
    
    @abstractmethod
    def get_answer(self, sample_id: int, target_node_idx:int) -> str:
        """
        Get the label of a node based on the sample ID."""
        
        pass
    
    @abstractmethod
    def create_networkx_graph(self, sub_graph_edge_index, node_mapping:dict, sub_graph_edge_mask=None) -> nx.Graph:
        """
        Create a NetworkX graph from the sampled subgraph.
        
        Args:
            sub_graph_edge_index: The edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
            sub_graph_edge_mask: The edge mask of the overall graph for sub_graph_edge_index.
        
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        pass
    
    
    def generate_graph(self, sample: int) -> nx.Graph:
        """
        Generate a single graph centered around a node using num_hops.
        If sampling is enabled, sample neighbors up to neighbor_size.

        Args:
            sample (int): The ID of the node to center the graph around.

        Returns:
            nx.Graph: A NetworkX graph for the sample.
        """
        sub_graph_edge_index, node_mapping, sub_graph_edge_mask = self.egograph_sampling(sample)
        G = self.create_networkx_graph(sub_graph_edge_index, node_mapping, sub_graph_edge_mask)
        new_G, node_idx_mapping_old_to_new = shuffle_nodes_randomly(G)
        # G = new_G
        # target sample_id in the shuffled graph
        target_node_idx = node_idx_mapping_old_to_new[node_mapping[sample]]
        
        query = self.get_query(target_node_idx)
        answer = self.get_answer(sample, target_node_idx)
        
        metadata = {
            "raw_sample_id": sample,
            "num_hop": self.num_hops,
            "sampling": {
                "enable": self.sampling,
                "neighbor_size": self.neighbor_size,
                "random_seed": self.random_seed
            },
            "main_task": {
                "query": query,
                "answer": answer,
                "target_node": target_node_idx
            }
        }
        
        return new_G, metadata
        
        
class EdgeTaskGraphGenerator(InputGraphGenerator):
    """
    A concrete implementation of InputGraphGenerator to generate graphs
    based on edge-centric sampling logic.
    """
    def __init__(self, num_hops=2, sampling=False, neighbor_size: int = None, random_seed: int = None, **kwargs):
        self.num_hops = num_hops
        self.sampling = sampling
        self.neighbor_size = neighbor_size
        self.random_seed = random_seed
        self.graph = None
        
        if self.sampling:
            assert neighbor_size is not None, "neighbor_size should be specified"
            assert random_seed is not None, "random_seed should be specified"

        self.load_data()
    
    def edge_egograph_sampling(self, edge: tuple, edge_index=None):
        '''
        Generate a k-hop subgraph for a given edge.
        
        Parameters:
            edge: The src and dst node indices of the edge for which the subgraph is generated.
            
        Returns:
            sub_graph_edge_index: The combined edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
            sub_graph_edge_mask: The edge mask of the overall graph for sub_graph_edge_index.
        '''
        if edge_index is None:
            edge_index = self.graph.edge_index
            
        sub_graph_nodes, sub_graph_edge_mask = generate_edge_centric_k_hop_subgraph(
            edge_index, edge, self.num_hops, self.neighbor_size, 
            self.random_seed, self.sampling
        )
        
        # raw node index to new node index mapping
        node_mapping = {
            raw_node_idx: new_node_idx 
            for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)
        }
        
        return node_mapping, sub_graph_edge_mask
    
    
    @abstractmethod
    def get_query(self, target_src_node_idx:int, target_dst_node_idx:int) -> str:
        """
        Get the query for the main task based on the target_node_idx 
        in the networkx graph object."""
        pass
    
    @abstractmethod
    def get_answer(self, edge:tuple, target_src_node_idx:int, target_dst_node_idx:int) -> str:
        """
        edge (tuple): The raw node indices of the src and dst node.
        target_src_node_idx (int): The new node index of the src node in the networkx graph.
        target_dst_node_idx (int): The new node index of the dst node in the networkx graph.
        Get the label of an edge based on the pair of nodes."""
        
        pass
    
    @abstractmethod
    def create_networkx_graph(self, node_mapping:dict, sub_graph_edge_mask=None, edge=None) -> nx.Graph:
        """
        Create a NetworkX graph from the sampled subgraph.
        
        Args:
            sub_graph_edge_index: The edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
            sub_graph_edge_mask: The edge mask of the overall graph for sub_graph_edge_index.
        
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        pass
    
    def generate_graph(self, sample: tuple, edge_index = None) -> nx.Graph:
        """
        Generate a single graph centered around an edge using num_hops.
        If sampling is enabled, sample neighbors up to neighbor_size.

        Args:
            sample tuple: The node ids of the edge src and dst.

        Returns:
            nx.Graph: A NetworkX graph for the sample.
        """
        if len(sample) == 2:
            src, dst = sample
        elif len(sample) == 3:
            # multiplex graph, multiple edges between the same node pair
            # id_ is the id_-th edge between the same node pair
            src, dst, id_ = sample
        
        node_mapping, sub_graph_edge_mask = self.edge_egograph_sampling((src, dst), edge_index=edge_index)
        
        G = self.create_networkx_graph(
            node_mapping=node_mapping, 
            sub_graph_edge_mask=sub_graph_edge_mask,
            edge=sample
        )
        
        new_G, node_idx_mapping_old_to_new = shuffle_nodes_randomly(G)
        
        # src, dst = sample
        target_src_node_idx = node_idx_mapping_old_to_new[node_mapping[src]]
        target_dst_node_idx = node_idx_mapping_old_to_new[node_mapping[dst]]
        
        query = self.get_query(target_src_node_idx, target_dst_node_idx)
        answer = self.get_answer(sample, target_src_node_idx, target_dst_node_idx)
        
        metadata = {
            "raw_sample_id": sample,
            "num_hop": self.num_hops,
            "sampling": {
                "enable": self.sampling,
                "neighbor_size": self.neighbor_size,
                "random_seed": self.random_seed
            },
            "main_task": {
                "query": query,
                "answer": answer,
                "target_edge": (target_src_node_idx, target_dst_node_idx)
            }
        }

        return new_G, metadata


class GraphTaskGraphGenerator(InputGraphGenerator):
    """
    A concrete implementation of InputGraphGenerator to generate graphs
    based on the entire graph.
    """
    def __init__(self, **kwargs):
        """
        Initialize the dataset and load the data. No sampling is required."""
        self.load_data()
    
    @abstractmethod
    def get_query(self, **kwargs) -> str:
        """
        Get the query for the main task based on the graph object."""
        
        pass
    
    @abstractmethod
    def get_answer(self, sample) -> str:
        """
        Get the label of the graph."""
        
        pass
    
    @abstractmethod
    def create_networkx_graph(self, sample) -> nx.Graph:
        """
        Create a NetworkX graph from the graph.
        
        Args:
        
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        pass
    
    def generate_graph(self, sample: int) -> nx.Graph:
        """
        Generate the entire graph

        Args:
            sample (int): The ID of the sample to generate a graph for.

        Returns:
            nx.Graph: A NetworkX graph for the sample.
        """
        G = self.create_networkx_graph(sample)
        
        query = self.get_query() # no need to specify target node or edge
        answer = self.get_answer(sample)
        
        new_G, node_idx_mapping_old_to_new = shuffle_nodes_randomly(G)
        
        metadata = {
            "raw_sample_id": sample,
            "main_task": {
                "query": query,
                "answer": answer,
            }
        }

        return new_G, metadata
    

class StructuralTaskGraphGenerator(InputGraphGenerator):
    """
    A concrete implementation of InputGraphGenerator to generate graphs
    of synthetic graph related tasks.
    """
    def __init__(self, task):
        self.config = load_yaml(
            os.path.join(
                os.path.dirname(__file__),
                "./../../configs/structural_task_generation.yaml"
            )
        )[task]
        self.root = os.path.join(os.path.dirname(__file__), self.config['file_path'])
        self.load_data()

    def load_data(self):
        """
        Load the dataset and preprocess required mappings.
        """
        dataset = torch.load(self.root)
        self.graphs, self.labels = dataset['graphs'], dataset['labels']

    def generate_graph(self, sample_id: int) -> nx.Graph:
        """
        Generate a single NetworkX graph for a given sample ID.
        Args:
            sample_id (int): The ID of the sample to generate a graph for.
        Returns:
            nx.Graph: A NetworkX graph representing the specific sample.
        """
        G = json_graph.node_link_graph(self.graphs[sample_id], directed=False)
        # print(f"load: {G=}")
        G = nx.MultiDiGraph(G)
        # print(f"multidi: {G=}")
    
        label, query_entity = self.labels[sample_id]
        query = self._generate_query(query_entity)
        # print(f"labels: {label=}, {query_entity=}")
        # exit()
        answer = self._generate_answer(label, query_entity)
        
        meta_data = {
            "raw_sample_id": sample_id,
            "main_task": {
                "query": query,
                "answer": answer,
                "query_entity": query_entity,
            }
        }

        return G, meta_data

    def _generate_query(self, query_entity):
        return self.config['query_format'].format(*query_entity)

    @abstractmethod
    def _generate_answer(self, label, query_entity=None):
        """
        Create a natural language answer based on the label and the task answer format.
        
        Args:
            label: The label of the sample.
        Returns:
            str: A text answer, combined with the anwser format.
        """
        pass
    
