from abc import ABC, abstractmethod
import networkx as nx
import random
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.utils.logger import logger

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
    def modify_graph(self, graph: nx.Graph) -> dict:
        """
        Modify the input graph to generate a new graph for self-supervised tasks.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            nx.Graph: The modified NetworkX graph.
        """
        pass

    @abstractmethod
    def generate_query(self, modify_outputs: dict) -> dict:
        """
        Generate a query for the self-supervised task.

        Args:
            graph (nx.Graph): The modified NetworkX graph.

        Returns:
            any: A query specific to the self-supervised task.
        """
        pass

    @abstractmethod
    def generate_answer(self, modify_outputs: dict, query_outputs: dict) -> dict:
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
        modify_outputs = self.modify_graph(graph)
        logger.debug(f"{modify_outputs=}")
        query_outputs = self.generate_query(modify_outputs)
        answer = self.generate_answer(modify_outputs, query_outputs)
        return {
            'modified_graph': modify_outputs['modified_graph'],
            'query': query_outputs['query_text'],
            'answer': answer
        }