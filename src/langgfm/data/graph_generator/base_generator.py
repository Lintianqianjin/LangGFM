from abc import ABC, abstractmethod
import networkx as nx

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