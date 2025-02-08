import os
import sys
import json
import logging
import warnings
import datetime
import networkx as nx
import torch
import pandas as pd
from tqdm import tqdm

from ..utils.graph_utils import get_edge_idx_in_graph, represent_edges_with_multiplex_id

from .._base_generator import EdgeTaskGraphGenerator


@EdgeTaskGraphGenerator.register("fb15k237")
class FB15K237GraphGenerator(EdgeTaskGraphGenerator):
    
    directed = True
    has_node_attr = True
    has_edge_attr = True
    
    def load_data(self):
        self.root = "./data/FB15K237"
        
        # Data(x=[14541, 1], edge_index=[2, 310116], train_mask=[310116], valid_mask=[310116], test_mask=[310116], edge_types=[310116])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            data = torch.load(f"{self.root}/full_data.pt")

        self.graph = data
        
        # logger.info(f"{data=}")
        
        with open(f"{self.root}/etypeid2rel.json") as etypeid2rel, \
            open(f"{self.root}/node_features_list.json") as node_features_list, \
            open(f"{self.root}/etype_split_distribution.json") as etype_distribution:
            
            self.etypeid2rel = json.load(etypeid2rel)
            self.node_features_list = json.load(node_features_list)
            self.etype_distribution = json.load(etype_distribution)
       
        # logger.info(f"{self.etypeid2rel=}")
       
        self.labels = list(map(lambda x: self.etypeid2rel[x], self.etype_distribution.keys()))
        
        edge_indices_candidates = torch.logical_or(torch.logical_or(self.graph.train_mask, self.graph.valid_mask), self.graph.test_mask).nonzero(as_tuple=True)[0].numpy()
        # logger.debug(f"{edge_indices_candidates=}")
        edges = self.graph.edge_index.T[edge_indices_candidates] # torch.tensor [num_edges, 2]
        # convert edges into set of tuples
        resutls = represent_edges_with_multiplex_id(self.graph.edge_index,edge_indices_candidates)
        # logger.debug(f"{resutls=}")
        self.all_samples = list(resutls)
    
    def graph_description(self):
        desc = "This graph is a subgraph from the knowledge graph FB15K237 where nodes represent entities and edges represent relationships."

        return desc
    
    def get_query(self, target_src_node_idx, target_dst_node_idx):
        query = (f"Plese infer the relation between the entity with node id {target_src_node_idx} and entity with node id {target_dst_node_idx}. "
        f"The available relations are: {self.labels}. ")
        return query

    def get_answer(self, edge, target_src_node_idx, target_dst_node_idx):
        
        src, dst, multiplex_id = edge
        edge_idx = get_edge_idx_in_graph(src, dst, self.graph.edge_index, multiplex_id=multiplex_id)
        
        # logger.debug(f"{edge_idx=}")
        # logger.debug(f"{self.graph.edge_types[edge_idx]=}")
        ground_truth = self.etypeid2rel[str(self.graph.edge_types[edge_idx].item())]
        
        answer = f"The most likely relation between the entity with node id {target_src_node_idx} and entity with node id {target_dst_node_idx} is {ground_truth}."
        return answer
    
    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None, edge=None):
        G = nx.MultiDiGraph()
        
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_feats = self.node_features_list[raw_node_idx]
            G.add_node(new_node_idx, type='entity', **node_feats)
        
        target_src = node_mapping[edge[0]]
        target_dst = node_mapping[edge[1]]
        
        # adding edges with attributes
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            # Skip the target edge
            if not (src == target_src and dst == target_dst): 
                edge_text = self.etypeid2rel[str(self.graph.edge_types[edge_idx].item())]
                G.add_edge(src, dst, type='relation', description = edge_text)
        
        return G
    