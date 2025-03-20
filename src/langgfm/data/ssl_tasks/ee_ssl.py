from .base_ssl import SelfSupervisedGraphTask
import networkx as nx
import random


@SelfSupervisedGraphTask.register("edge_existence_prediction")
class EdgeExistencePrediction(SelfSupervisedGraphTask):
    def __init__(self, remove_edge_ratio=0.1, min_edges_to_remove=1):
        """
        Initialize the EdgeExistencePrediction task.

        Args:
            remove_edge_ratio (float): Ratio of edges to remove (0 < ratio < 1).
            min_edges_to_remove (int): Minimum number of edges to remove.
        """
        self.remove_edge_ratio = remove_edge_ratio
        self.min_edges_to_remove = min_edges_to_remove
        
        # Validate parameters
        if not (0 < self.remove_edge_ratio < 1):
            raise ValueError("remove_edge_ratio must be between 0 and 1.")
        if self.min_edges_to_remove < 1:
            raise ValueError("min_edges_to_remove must be at least 1.")

    def modify_graph(self, graph: nx.Graph) -> dict:
        """
        Remove a subset of edges from the graph.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            dict: A dictionary containing:
                - "modified_graph": The graph with some edges removed.
                - "removed_edges": List of edges that were removed.
                - "original_graph": The original unmodified graph.
        """
        # Make a deep copy of the graph to avoid modifying the original
        original_graph = graph.copy()
        modified_graph = graph.copy()
        
        # Get all edges from the graph
        if modified_graph.is_multigraph():
            edges = list(modified_graph.edges(keys=True))
        else:
            edges = list(modified_graph.edges())
        
        # Calculate number of edges to remove
        num_edges_to_remove = max(
            self.min_edges_to_remove,
            int(len(edges) * self.remove_edge_ratio)
        )
        num_edges_to_remove = min(num_edges_to_remove, len(edges))
        
        # Randomly select edges to remove
        edges_to_remove = random.sample(edges, num_edges_to_remove)
        
        # Remove the selected edges
        for edge in edges_to_remove:
            if modified_graph.is_multigraph():
                u, v, k = edge
                modified_graph.remove_edge(u, v, k)
            else:
                u, v = edge
                modified_graph.remove_edge(u, v)
        
        return {
            "modified_graph": modified_graph,
            "removed_edges": edges_to_remove,
            "original_graph": original_graph
        }

    def generate_query(self, modify_outputs: dict) -> dict:
        """
        Generate a query about the existence of an edge.

        Args:
            modify_outputs: Output from modify_graph.

        Returns:
            dict: A dictionary containing:
                - "query_text": Natural language query about edge existence.
                - "query_edge": The edge being queried.
                - "query_type": Type of query (exists or not_exists).
        """
        modified_graph = modify_outputs["modified_graph"]
        removed_edges = modify_outputs["removed_edges"]
        original_graph = modify_outputs["original_graph"]
        
        # Decide whether to query about an existing edge or a removed edge
        query_type = random.choice(["exists", "not_exists"])
        
        if query_type == "exists" and len(list(modified_graph.edges())) > 0:
            # Query about an existing edge
            if modified_graph.is_multigraph():
                edges = list(modified_graph.edges(keys=True))
                u, v, k = random.choice(edges)
                query_edge = (u, v, k)
                query_text = f"Is there an edge with key {k} between node {u} and node {v} in the graph?"
            else:
                edges = list(modified_graph.edges())
                u, v = random.choice(edges)
                query_edge = (u, v)
                query_text = f"Is there an edge between node {u} and node {v} in the graph?"
        elif len(removed_edges) > 0:
            # Query about a removed edge
            query_edge = random.choice(removed_edges)
            if modified_graph.is_multigraph():
                u, v, k = query_edge
                query_text = f"Is there an edge with key {k} between node {u} and node {v} in the graph?"
            else:
                u, v = query_edge
                query_text = f"Is there an edge between node {u} and node {v} in the graph?"
            query_type = "not_exists"
        else:
            # Fallback if no edges were removed
            if modified_graph.is_multigraph():
                edges = list(modified_graph.edges(keys=True))
                u, v, k = random.choice(edges)
                query_edge = (u, v, k)
                query_text = f"Is there an edge with key {k} between node {u} and node {v} in the graph?"
            else:
                edges = list(modified_graph.edges())
                u, v = random.choice(edges)
                query_edge = (u, v)
                query_text = f"Is there an edge between node {u} and node {v} in the graph?"
            query_type = "exists"
        
        return {
            "query_text": query_text,
            "query_edge": query_edge,
            "query_type": query_type
        }

    def generate_answer(self, modify_outputs: dict, query_outputs: dict) -> str:
        """
        Generate a natural language answer about edge existence.

        Args:
            modify_outputs: Output from modify_graph.
            query_outputs: Output from generate_query.

        Returns:
            str: A natural language answer.
        """
        query_edge = query_outputs["query_edge"]
        query_type = query_outputs["query_type"]
        
        if query_type == "exists":
            if len(query_edge) == 3:
                u, v, k = query_edge
                return f"Yes, there is an edge with key {k} between node {u} and node {v} in the graph."
            else:
                u, v = query_edge
                return f"Yes, there is an edge between node {u} and node {v} in the graph."
        else:
            if len(query_edge) == 3:
                u, v, k = query_edge
                return f"No, there is no edge with key {k} between node {u} and node {v} in the graph."
            else:
                u, v = query_edge
                return f"No, there is no edge between node {u} and node {v} in the graph."

