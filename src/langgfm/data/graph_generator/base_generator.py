from abc import ABC, abstractmethod
import networkx as nx
from .utils.sampling import generate_node_centric_k_hop_subgraph
from .utils.shuffle_graph import shuffle_nodes_randomly

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
    
    
    
@InputGraphGenerator.register("NodeGraphGenerator")
class NodeGraphGenerator(InputGraphGenerator):
    """
    A concrete implementation of InputGraphGenerator to generate graphs
    based on node-centric sampling logic.
    """
    def __init__(self, num_hops=2, sampling=False, neighbor_size: int = None, random_seed: int = None):
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
        
        sub_graph_edge_index, sub_graph_nodes = generate_node_centric_k_hop_subgraph(
            self.graph, sample_id, self.num_hops, self.neighbor_size, 
            self.random_seed, self.sampling
        )
        
        # raw node index to new node index mapping
        node_mapping = {
            raw_node_idx: new_node_idx 
            for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)
        }
        
        return sub_graph_edge_index, node_mapping
    
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
    def create_networkx_graph(self, sub_graph_edge_index, node_mapping:dict) -> nx.Graph:
        """
        Create a NetworkX graph from the sampled subgraph.
        
        Args:
            sub_graph_edge_index: The edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
        
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        pass
    
    
    def generate_graph(self, sample_id: int) -> nx.Graph:
        """
        Generate a single graph centered around a node using num_hops.
        If sampling is enabled, sample neighbors up to neighbor_size.

        Args:
            sample_id (int): The ID of the node to center the graph around.

        Returns:
            nx.Graph: A NetworkX graph for the sample.
        """
        sub_graph_edge_index, node_mapping = self.egograph_sampling(sample_id)
        G = self.create_networkx_graph(sub_graph_edge_index, node_mapping)
        new_G, node_idx_mapping_old_to_new = shuffle_nodes_randomly(G)
        # G = new_G
        # target sample_id in the shuffled graph
        target_node_idx = node_idx_mapping_old_to_new[node_mapping[sample_id]]
        
        query = self.get_query(target_node_idx)
        answer = self.get_answer(sample_id, target_node_idx)
        
        metadata = {
            "raw_sample_id": sample_id,
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
        
        