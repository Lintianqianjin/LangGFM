import json
import networkx as nx

import torch
import pandas as pd

from .utils.ogb_dataset import CustomPygLinkPropPredDataset
from ._base_generator import EdgeTaskGraphGenerator

@EdgeTaskGraphGenerator.register("ogbl_vessel")
class OgblVesselGraphGenerator(EdgeTaskGraphGenerator):
    """
    OgblVesselGraphGenerator: A generator for creating k-hop subgraphs 
    from the OGBL-Vessel dataset using NetworkX format.
    """

    def __convert_edges_to_dict(self, edges:torch.Tensor) -> dict:
        """
        Convert edge tensors to a dictionary where keys are (src, dst) tuples 
        and values are 1 for edges from 'edge' and 0 for edges from 'edge_neg'. 
        Also ensures (dst, src) pairs exist with the same value.
        If a conflicting value is encountered, an assertion error is raised.

        Args:
            edges (dict): A dictionary containing 'edge' and 'edge_neg' keys,
                        each mapping to an Nx2 torch.Tensor.

        Returns:
            dict: A dictionary where keys are (src, dst) tuples and values are 0 or 1.
        """
        edge_dict = {}

        def add_edges(tensor, value):
            """Add edges to the dictionary ensuring (src, dst) and (dst, src) symmetry."""
            for i in range(tensor.shape[0]):
                src, dst = int(tensor[i, 0]), int(tensor[i, 1])

                # Check if the edge already exists with a different value
                if (src, dst) in edge_dict:
                    assert edge_dict[(src, dst)] == value, \
                        f"Conflict detected for edge ({src}, {dst}): existing value {edge_dict[(src, dst)]}, new value {value}"
                
                if (dst, src) in edge_dict:
                    assert edge_dict[(dst, src)] == value, \
                        f"Conflict detected for edge ({dst}, {src}): existing value {edge_dict[(dst, src)]}, new value {value}"

                # Assign the value ensuring symmetry
                edge_dict[(src, dst)] = value
                edge_dict[(dst, src)] = value  

        # Process positive edges
        if 'edge' in edges and isinstance(edges['edge'], torch.Tensor):
            add_edges(edges['edge'], 1)

        # Process negative edges
        if 'edge_neg' in edges and isinstance(edges['edge_neg'], torch.Tensor):
            add_edges(edges['edge_neg'], 0)

        return edge_dict
        
    
    def load_data(self):
        """
        Load the OGBL-Vessel dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        dataset = CustomPygLinkPropPredDataset(name = 'ogbl-vessel', root=f'{self.root}') 
        self.graph = dataset[0] # pyg graph object containing only training edges
        
        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
        
        self.edge_label_mapping = self.__convert_edges_to_dict(train_edge)
        self.all_samples = set(self.edge_label_mapping.keys())
    
    @property
    def graph_description(self):
        """
        Get the description of the graph.
        """
        return "This graph is an undirected, unweighted spatial graph of a partial mouse brain. "\
            "Nodes represent bifurcation points, edges represent the vessels. The node features are 3-dimensional, "\
            "representing the spatial (x, y, z) coordinates of the nodes in Allen Brain atlas reference space."

    def get_query(self, target_src_node_idx, target_dst_node_idx):
        """
        Get the query for the main task based on the target_src_node_idx and target_dst_node_idx
        in the networkx graph object.
        """
        query = (f"Plese infer whether a vessel exists between "
            f"the bifurcation point with node id {target_src_node_idx} "
            f"and the bifurcation point with node id {target_dst_node_idx}.")
        
        return query
            
    def get_answer(self, edge, target_src_node_idx, target_dst_node_idx):
        """
        Get the answer for a specific edge.
        Args:
            edge (tuple): The node indices of the src and dst node.
        """
        src, dst = edge
        label = self.edge_label_mapping[(src, dst)]
        if label == 1:
            answer = (f"Yes, a vessel likely exists between the "
            f"bifurcation point with node id {target_src_node_idx} and "
            f"the bifurcation point with node id {target_dst_node_idx}.")
        else: 
            answer = (f"No, a vessel likely does not exist between the "
            f"bifurcation point with node id {target_src_node_idx} and "
            f"the bifurcation point with node id {target_dst_node_idx}.")
        
        return answer
    
    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None, edge=None, **kwargs):
        """
        Create a NetworkX graph from the sampled subgraph.
        """
        G = nx.MultiDiGraph()
        
        for raw_node_idx, new_node_idx in node_mapping.items():
            x, y, z = self.graph.x[raw_node_idx].numpy()
            G.add_node(new_node_idx, type='bifurcation', x=x, y=y, z=z)

        target_src = node_mapping[edge[0]]
        target_dst = node_mapping[edge[1]]
        
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            # Skip the target edge
            if not (src == target_src and dst == target_dst) or (src == target_dst and dst == target_src): 
                G.add_edge(src, dst, type='vessel')
                G.add_edge(dst, src, type='vessel')
        
        return G

    