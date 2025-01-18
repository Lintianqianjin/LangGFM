from .base_ssl import SelfSupervisedGraphTask
import networkx as nx
import copy
import random
import copy
import random


import logging
logger = logging.getLogger("main_logger")


class FeatureMaskedAutoencoder(SelfSupervisedGraphTask):
    def __init__(self, mask_node_ratio=0.2, mask_edge_ratio=0.2, mask_reverse_edges=True, mask_value="Unknown", random_seed=None):
        """
        Initialize the FeatureMaskedAutoencoder.

        Args:
            mask_node_ratio (float): Ratio of nodes to mask (0 <= ratio <= 1).
            mask_edge_ratio (float): Ratio of edges to mask (0 <= ratio <= 1).
            mask_reverse_edges (bool): Whether to also mask the reverse edge if it exists.
            mask_value (str): The value to use for masking node or edge features.
            random_seed (int, optional): Seed for random sampling to ensure reproducibility.
        """
        self.mask_node_ratio = mask_node_ratio
        self.mask_edge_ratio = mask_edge_ratio
        self.mask_reverse_edges = mask_reverse_edges
        self.mask_value = mask_value
        self.random_seed = random_seed

        # Validate the mask ratios to ensure they are in the valid range.
        self.__validate_ratios()
        if self.random_seed is not None:
            random.seed(self.random_seed)  # Set the random seed for reproducibility.

    def __validate_ratios(self):
        """
        Validate the mask ratios to ensure they are within the range [0, 1].
        """
        if not (0 <= self.mask_node_ratio <= 1):
            raise ValueError("mask_node_ratio must be between 0 and 1.")
        if not (0 <= self.mask_edge_ratio <= 1):
            raise ValueError("mask_edge_ratio must be between 0 and 1.")

    def modify_graph(self, graph: nx.Graph) -> dict:
        """
        Mask features of nodes and edges in the graph.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            dict: A dictionary containing:
                - "modified_graph": The modified graph with masked features.
                - "masked_nodes_features": Original features of masked nodes.
                - "masked_edges_features": Original features of masked edges.
        """
        # Ensure the input graph is a supported type.
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError("Unsupported graph type. Please provide a NetworkX Graph or MultiDiGraph.")

        # Deep copy the graph to avoid modifying the original graph.
        G = graph.copy()
        
        # save node and edge features from the original graph.
        # changes following will be made on the copied graph.
        node_features = self.__get_node_features(graph)

        edge_features = self.__get_edge_features(graph)

        # Mask node features as self.mask_value on G. 
        masked_nodes, masked_nodes_features = self.__mask_node_features(G, node_features)

        # Mask node features as self.mask_value on G. 
        masked_edges, masked_edges_features = self.__mask_edge_features(G, edge_features)

        # Return the modified graph and the masked feature mappings.
        return {
            "modified_graph": G,
            "masked_nodes_features": masked_nodes_features,
            "masked_edges_features": masked_edges_features
        }

    def __get_node_features(self, G):
        """
        Extract node features from the graph.

        Args:
            G (nx.Graph): The input graph.

        Returns:
            dict: A dictionary where keys are nodes and values are their features.
        """
        node_features = {}
        for node, attrs in G.nodes(data=True):
            if attrs:  # Only include nodes that have attributes.
                node_features[node] = attrs
        return node_features

    def __get_edge_features(self, G):
        """
        Extract edge features from the graph.

        Args:
            G (nx.Graph): The input graph.

        Returns:
            dict: A dictionary where keys are edges (tuples) and values are their features.
        """
        edge_features = {}
        if G.is_multigraph():
            # For MultiGraph or MultiDiGraph, include edge keys.
            for u, v, k, attrs in G.edges(data=True, keys=True):
                if attrs:
                    edge_features[(u, v, k)] = attrs
        else:
            # For standard Graph or DiGraph, no edge keys are needed.
            for u, v, attrs in G.edges(data=True):
                if attrs:
                    edge_features[(u, v)] = attrs
        return edge_features

    def __mask_node_features(self, G, node_features):
        """
        Mask a subset of node features in the graph.

        Args:
            G (nx.Graph): The input graph.
            node_features (dict): Original node features.

        Returns:
            tuple: (masked_nodes, masked_nodes_features)
                - masked_nodes: List of nodes that were masked.
                - masked_nodes_features: Original features of the masked nodes.
        """
        masked_nodes = []
        masked_nodes_features = {}
        if node_features:
            nodes = list(node_features.keys())
            num_nodes_to_mask = int(len(nodes) * self.mask_node_ratio)
            num_nodes_to_mask = max(1, num_nodes_to_mask)  # Ensure at least one node is masked.
            masked_nodes = random.sample(nodes, num_nodes_to_mask)
            masked_nodes_features = {node: node_features[node] for node in masked_nodes}
            
            # Apply masking to the selected nodes.
            for node in masked_nodes:
                for key in G.nodes[node]:
                    G.nodes[node][key] = self.mask_value
        return masked_nodes, masked_nodes_features

    def __mask_edge_features(self, G, edge_features):
        """
        Mask a subset of edge features in the graph, including reverse edges if specified.

        Args:
            G (nx.Graph): The input graph.
            edge_features (dict): Original edge features.

        Returns:
            tuple: (masked_edges, masked_edges_features)
                - masked_edges: List of edges that were masked.
                - masked_edges_features: Original features of the masked edges.
        """
        masked_edges = []
        masked_edges_features = {}
        if edge_features:
            edges = list(edge_features.keys())
            num_edges_to_mask = int(len(edges) * self.mask_edge_ratio)
            num_edges_to_mask = max(1, num_edges_to_mask)  # Ensure at least one edge is masked.
            selected_edges = random.sample(edges, num_edges_to_mask)
            
            # Mask the selected edges and optionally their reverse edges.
            for edge in selected_edges:
                src, dst, *key = edge  # Handle optional key for MultiGraph/MultiDiGraph.
                masked_edges.append(edge)
                masked_edges_features[edge] = edge_features[edge]
                
                # Mask the edge in the graph.
                for k in G.edges[edge]:
                    G.edges[edge][k] = self.mask_value

                # Optionally mask the reverse edge if it exists.
                if self.mask_reverse_edges:
                    reverse_edge = (dst, src, *key) if key else (dst, src)
                    if reverse_edge in edge_features:
                        masked_edges.append(reverse_edge)
                        masked_edges_features[reverse_edge] = edge_features[reverse_edge]
                        for k in G.edges[reverse_edge]:
                            G.edges[reverse_edge][k] = self.mask_value
        return masked_edges, masked_edges_features
    

@SelfSupervisedGraphTask.register("node_feature_masked_autoencoder")
class NodeFeatureMaskedAutoencoder(FeatureMaskedAutoencoder):
    def generate_query(self, modify_outputs: dict) -> str:
        """
        Generate a natural language query for a single masked node.

        Args:
            modify_outputs: outputs of function modify_graph.

        Returns:
            str: A natural language query.
        """
        masked_nodes = list(modify_outputs["masked_nodes_features"].keys())
        selected_node = random.choice(masked_nodes)

        query_text = f"The attribute(s) of node {selected_node} seems to be missing. Please infer the attribute value(s) and return them in dictionary form (with attribute names as keys and inferred values as values)." 
        return {
            "query_node": selected_node,
            "query_text": query_text
        }


    def generate_answer(self, modify_outputs: dict, query_outputs: dict) -> str:
        """
        Generate a natural language answer for the masked node feature.

        Args:
            graph (nx.Graph): The modified graph.
            query (str): The natural language query generated.

        Returns:
            str: A natural language answer.
        """
        selected_node = query_outputs['query_node']
        features = modify_outputs['masked_nodes_features'][selected_node]
        # features_str = ", ".join(f"{key}: {value}" for key, value in features.items())
        return f"The attribute(s) of node {selected_node} should be {features}."


@SelfSupervisedGraphTask.register("edge_feature_masked_autoencoder")
class EdgeFeatureMaskedAutoencoder(FeatureMaskedAutoencoder):
    def generate_query(self, modify_outputs: nx.Graph) -> str:
        """
        Generate a natural language query for a single masked edge.

        Args:
            graph (nx.Graph): The modified graph.

        Returns:
            str: A natural language query.
        """
        
        masked_edges = list(modify_outputs["masked_edges_features"].keys())
        # print(f"{masked_edges=}")
        logger.debug(f"{masked_edges=}")
        selected_edge = random.choice(masked_edges)
        
        if modify_outputs['modified_graph'].is_multigraph(): 
            (src, dst, key) = selected_edge
            query_text = f"The attribute(s) of the edge between node {src} and node {dst} with key {key} seems to be missing. Please infer the attribute value(s) and return them in dictionary form (with attribute names as keys and inferred values as values)."
        else:
            (src, dst) = selected_edge
            query_text = f"The attribute(s) of the edge between node {src} and node {dst} seems to be missing. Please infer the attribute value(s) and return them in dictionary form (with attribute names as keys and inferred values as values)." 
        
        return {
            "query_edge": selected_edge,
            "query_text": query_text
        }


    def generate_answer(self, modify_outputs: nx.Graph, query_outputs: str) -> str:
        """
        Generate a natural language answer for the masked edge feature.

        Args:
            graph (nx.Graph): The modified graph.
            query (str): The natural language query generated.

        Returns:
            str: A natural language answer.
        """
        selected_edge = query_outputs['query_edge']
        features = modify_outputs['masked_edges_features'][selected_edge]
        
        if len(selected_edge) == 3:
            (src, dst, key) = selected_edge
            return f"The attribute(s) of the edge between node {src} and node {dst} with key {key} should be {features}."
        else:
            (src, dst) = selected_edge
            return f"The attribute(s) of the edge between node {src} and node {dst} should be {features}."
        