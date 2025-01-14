import re
import os
import sys
import json
import networkx as nx
import torch
import pandas as pd

from torch_geometric.data import Data
from collections import Counter

from ..utils.graph_utils import get_edge_idx_in_graph
from .._base_generator import EdgeTaskGraphGenerator
from ....utils.logger import logger
logger.set_level("DEBUG")

@EdgeTaskGraphGenerator.register("stack_elec")
class StackElecGraphGenerator(EdgeTaskGraphGenerator):
    """
    StackElecGraphGenerator: A generator for creating k-hop subgraphs from the StackExchange dataset using NetworkX format.
    """
    
    directed = True
    
    def load_data(self):
        self.root = "data/stack_elec"
        
        self.entity_text = pd.read_csv(f"{self.root}/entity_text.csv", index_col=0)
        self.edge_list = pd.read_csv(f"{self.root}/edge_list.csv", index_col=0)
        logger.debug(f"{Counter([(row['u'], row['i']) for row_idx, row in self.edge_list.iterrows()]).most_common(10)=}")
        # logger.debug(self.edge_list.loc[self.edge_list['i'] < 67155])
        self.relation_text = pd.read_csv(f"{self.root}/relation_text.csv", index_col=0)
        
        edge_index = torch.tensor(self.edge_list[['u','i']].values, dtype=torch.long).t().contiguous()
        relation_idx = torch.tensor(self.edge_list['r'].values, dtype=torch.long)
        edge_label = torch.tensor(self.edge_list['label'].values, dtype=torch.long)
        
        # create pyg data object
        if not os.path.exists(f"{self.root}/data.pt"):
            data = Data(edge_index=edge_index, edge_relation_idx=relation_idx, edge_label=edge_label)
            torch.save(data, f"{self.root}/data.pt")
        else:
            data = torch.load(f"{self.root}/data.pt", weights_only=False)
            
        self.graph = data
        
        self.all_samples = set([(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))])
        
    def graph_description(self):
        desc = ("This is a graph constructed from question-and-answer data related to electronic techniques of the Stack Exchange platform, "
        "forming a dynamic bipartite structure that captures multi-round interactions between users and questions. "
        "Nodes are users and questions. User nodes represent individuals who ask, answer, or comment on questions, "
        "while question nodes correspond to technical inquiries on the platform. "
        "Edges represent interactions between users and questions, including answers and comments.")

        return desc
    
    def get_query(self, target_src_node_idx, target_dst_node_idx):
        query = (f"Each edge, i.e., answer or comment, is classified as 'Useful' if its voting count is greater than 1, and 'Useless' otherwise."
        f"Please infer the category of the edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx}.")
        return query
    
    def get_answer(self, edge, target_src_node_idx, target_dst_node_idx):
        """
        Get the answer for a specific edge.
        Args:
            edge (tuple): The node indices of the src and dst node.
        """
        src, dst = edge
        edge_idx = get_edge_idx_in_graph(src, dst, self.graph.edge_index)
        label = self.graph.edge_label[edge_idx]
        
        if label == 1:
            answer = (f"The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} "
            "is likely to be classified as 'Useful'.")
            
        else: 
            answer = (f"The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} "
            "is likely to be classified as 'Useless'.")
        
        return answer

    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None, edge=None, **kwargs):
        """
        Create a NetworkX graph from the sampled subgraph.
        """
        
        G = nx.MultiDiGraph()
        
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_text = self.entity_text.at[raw_node_idx,'text']
            node_type, features = self.__parse_node_text(node_text)
            G.add_node(new_node_idx, type=node_type, **features)
            

        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            # get relation text
            relation_idx = self.graph.edge_relation_idx[edge_idx].item()
            relation_text = self.relation_text.at[relation_idx, 'text']
            
            G.add_edge(src, dst, content=relation_text)
            
        
        return G
        
    def __parse_node_text(self, node_text):
    
        """
        Parse the input text to extract node type (node_type) and corresponding feature information (features).
        :param text: str, input text content
        :return: tuple, (node_type, features)
        """
        user_pattern = re.compile(r"Name:\s*(.*?)\s*Location:\s*(.*?)\s*Introduction:\s*(.*)", re.DOTALL)
        question_pattern = re.compile(r"Title:\s*(.*?)\s*Post:\s*(.*)", re.DOTALL)
        
        user_match = user_pattern.match(node_text)
        
        if user_match:
            # remove "Name": user_match.group(1), 
            return "user", {"Location": user_match.group(2), "Introduction": user_match.group(3)}
        
        question_match = question_pattern.match(node_text)
        if question_match:
            return "question", {"Title": question_match.group(1), "Post": question_match.group(2)}
        
        return None, {}  # Return None and an empty dictionary if no pattern is matched
