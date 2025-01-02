from abc import ABC, abstractmethod
import networkx as nx
import random

class SelfSupervisedGraphTask(ABC):
    """
    Abstract base class for generating self-supervised task samples
    from a given NetworkX graph.
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
            raise ValueError(f"Unknown task type: {name}. Available types: {list(cls.registry.keys())}")
        return cls.registry[name](*args, **kwargs)

    @abstractmethod
    def modify_graph(self, graph: nx.Graph) -> nx.Graph:
        """
        Modify the input graph to generate a new graph for self-supervised tasks.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            nx.Graph: The modified NetworkX graph.
        """
        pass

    @abstractmethod
    def generate_query(self, graph: nx.Graph) -> str:
        """
        Generate a query for the self-supervised task.

        Args:
            graph (nx.Graph): The modified NetworkX graph.

        Returns:
            any: A query specific to the self-supervised task.
        """
        pass

    @abstractmethod
    def generate_answer(self, graph: nx.Graph, query: any) -> str:
        """
        Generate the answer for the self-supervised task based on the query.

        Args:
            graph (nx.Graph): The modified NetworkX graph.
            query (any): The query generated for the self-supervised task.

        Returns:
            any: The answer corresponding to the query.
        """
        pass

    def generate_sample(self, graph: nx.Graph) -> dict:
        """
        Generate a self-supervised task sample.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            dict: A dictionary containing:
                - 'modified_graph': The modified NetworkX graph.
                - 'query': The query for the self-supervised task.
                - 'answer': The answer to the query.
        """
        modified_graph = self.modify_graph(graph)
        query = self.generate_query(modified_graph)
        answer = self.generate_answer(modified_graph, query)
        return {
            'modified_graph': modified_graph,
            'query': query['query_text'],
            'answer': answer
        }