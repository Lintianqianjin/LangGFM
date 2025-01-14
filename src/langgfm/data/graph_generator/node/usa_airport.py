import json
import networkx as nx

import torch

from .._base_generator import NodeTaskGraphGenerator


@NodeTaskGraphGenerator.register("usa_airport")
class USAirportGraphGenerator(NodeTaskGraphGenerator):
    """
    USAirportGraphGenerator: A generator for creating k-hop subgraphs 
    from the USAirport dataset using NetworkX format.
    """

    def load_data(self):
        """
        Load the USAirport dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        
        self.graph = torch.load(f'{self.root}/USAAirport/usa_airport.pt')
        self.node_idx_label_mapping = self.graph.y.numpy() # array
        self.all_samples = set(range(self.graph.num_nodes))
    
    def get_query(self, target_node_idx: int) -> str:
        """
        Get the query for the main task based on the target_node_idx 
        in the networkx graph object.
        """
        query = (f"Airport activity is measured by the total number of people passing through "
            f"(arrivals plus departures) during a given period. We use the quartiles from the "
            f"activity distribution to divide the airports into four groups. Group 0 represents "
            f"the 25% least active airports, and so forth. Please determine which group the "
            f"airport with the node id {target_node_idx} belongs to."
        )
        return query

    def get_answer(self, sample_id, target_node_idx: int) -> str:
        """
        Get the answer for a specific sample node.
        Args:
            sample_id (int): The index of the sample node.
        """
        label = self.node_idx_label_mapping[sample_id]
        answer = f"The airport with node id of {target_node_idx} likely belongs to group {label}."
        return answer
        
    def create_networkx_graph(self, node_mapping:dict, sub_graph_edge_mask=None) -> nx.Graph:
        """
        Create a NetworkX graph from the sampled subgraph.
        
        Args:
            sub_graph_edge_index: The edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
            sub_graph_edge_mask: The edge mask of the overall graph for sub_graph_edge_index.
        
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        G = nx.MultiDiGraph()
        for raw_node_idx, new_node_idx in node_mapping.items():
            G.add_node(new_node_idx)
            
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            G.add_edge(src, dst)

        return G
    