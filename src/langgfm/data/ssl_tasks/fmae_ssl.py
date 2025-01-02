from .base_ssl import SelfSupervisedGraphTask
import networkx as nx
import copy
import random

@SelfSupervisedGraphTask.register("feature_masked_autoencoder")
class FeatureMaskedAutoencoder(SelfSupervisedGraphTask):
    def __init__(self, mask_node_ratio=0.2, mask_edge_ratio=0.2):
        """
        Initialize the FeatureMaskedAutoencoder.

        Args:
            mask_node_ratio (float): Ratio of nodes to mask.
            mask_edge_ratio (float): Ratio of edges to mask.
        """
        self.mask_node_ratio = mask_node_ratio
        self.mask_edge_ratio = mask_edge_ratio

    def modify_graph(self, graph: nx.Graph) -> nx.Graph:
        """
        Apply feature masking to nodes and edges in the graph.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            nx.Graph: The modified graph with masked features.
        """
        G = copy.deepcopy(graph)
        self.node_features = self.get_node_features(G)
        self.edge_features = self.get_edge_features(G)

        # Mask node features
        if self.node_features:
            nodes = list(self.node_features.keys())
            num_nodes_to_mask = int(len(nodes) * self.mask_node_ratio)
            self.masked_nodes = random.sample(nodes, num_nodes_to_mask)
            self.masked_nodes_features = {node: copy.deepcopy(self.node_features[node]) for node in self.masked_nodes}
            for node in self.masked_nodes:
                for key in G.nodes[node]:
                    G.nodes[node][key] = "Unknown"  # Mask the feature value

        # Mask edge features
        if self.edge_features:
            edges = list(self.edge_features.keys())
            num_edges_to_mask = int(len(edges) * self.mask_edge_ratio)
            self.masked_edges = random.sample(edges, num_edges_to_mask)
            self.masked_edges_features = {edge: copy.deepcopy(self.edge_features[edge]) for edge in self.masked_edges}
            for edge in self.masked_edges:
                for key in G.edges[edge]:
                    G.edges[edge][key] = "Unknown"  # Mask the feature value

        return G

    def generate_query(self, graph: nx.Graph) -> str:
        """
        Generate a natural language query for the masked features.

        Args:
            graph (nx.Graph): The modified graph.

        Returns:
            str: A natural language query.
        """
        query = []
        if hasattr(self, "masked_nodes") and self.masked_nodes:
            query.append(
                f"Please recover the features of the following nodes: {', '.join(map(str, self.masked_nodes))}."
            )
        if hasattr(self, "masked_edges") and self.masked_edges:
            query.append(
                f"Please recover the features of the following edges: {', '.join(map(str, self.masked_edges))}."
            )
        return " ".join(query)

    def generate_answer(self, graph: nx.Graph, query: str) -> str:
        """
        Generate a natural language answer for the masked features.

        Args:
            graph (nx.Graph): The modified graph.
            query (str): The natural language query generated.

        Returns:
            str: A natural language answer.
        """
        answer = []
        if hasattr(self, "masked_nodes_features") and self.masked_nodes_features:
            for node, features in self.masked_nodes_features.items():
                features_str = ", ".join(f"{key}: {value}" for key, value in features.items())
                answer.append(f"The features of node {node} are: {features_str}.")
        if hasattr(self, "masked_edges_features") and self.masked_edges_features:
            for edge, features in self.masked_edges_features.items():
                features_str = ", ".join(f"{key}: {value}" for key, value in features.items())
                answer.append(f"The features of edge {edge} are: {features_str}.")
        return " ".join(answer)

    def get_node_features(self, graph: nx.Graph) -> dict:
        """
        Retrieve the node features from the graph.

        Args:
            graph (nx.Graph): The input graph.

        Returns:
            dict: A dictionary of node features.
        """
        return {node: data for node, data in graph.nodes(data=True)}

    def get_edge_features(self, graph: nx.Graph) -> dict:
        """
        Retrieve the edge features from the graph.

        Args:
            graph (nx.Graph): The input graph.

        Returns:
            dict: A dictionary of edge features.
        """
        return {edge: data for edge, data in graph.edges(data=True)}