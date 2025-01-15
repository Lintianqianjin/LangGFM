import ast
import networkx as nx
import pandas as pd
import warnings

import torch

from ..utils.graph_utils import get_node_slices

from .._base_generator import NodeTaskGraphGenerator


@NodeTaskGraphGenerator.register("oag_scholar_interest")
class OAGScholarInterestGraphGenerator(NodeTaskGraphGenerator):
    """
    OAGScholarInterestGraphGenerator: A generator for creating k-hop subgraphs 
    from the OAG Scholar Interest dataset using NetworkX format.
    """
    directed = True
    has_node_attr = True
    has_edge_attr = False

    def load_data(self):
        """
        Load the OAG Scholar Interest dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data/oag_scholar_interest'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            self.graph = torch.load(f'{self.root}/raw/dblp_hetero_data.pt')
        
        self.author_nodes = pd.read_csv(f'{self.root}/raw/author_nodes_with_interests.csv')
        self.author_nodes["Research_Interests"] = self.author_nodes["Research_Interests"].fillna("[]")
        self.author_nodes["Research_Interests"] = self.author_nodes["Research_Interests"].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x))
        self.author_labelled_index = self.author_nodes.loc[self.author_nodes['labelled']]['node_idx'].to_list()
        
        self.all_samples = set(self.author_labelled_index)
        
        self.paper_nodes = pd.read_csv(f'{self.root}/raw/paper_nodes.csv')
        self.venue_nodes = pd.read_csv(f'{self.root}/raw/venue_nodes.csv')
        
        self.node_slices = get_node_slices(self.graph.num_nodes_dict)
        self.node_type_mapping = {0: 'author', 1: 'paper', 2: 'venue'}
        self.edge_type_mapping = {0:'writes', 1:'cites', 2:'publishes'}
        self.graph = self.graph.to_homogeneous()
    
    @property
    def graph_description(self):
        desc = (
            "This is an academic graph from DBLP. Nodes are authors, papers, and venues. "
            "Edges are authors write papers, papers cite other papers, and venues publish papers."
        )
        return desc
    
    def get_query(self, target_node_idx):
        query = (f"Scholars usually describe their research interests with a few words or phrases in their profiles. "
                f"Can you predict how the author with id {target_node_idx} would describe research interests in 2024?")
        
        return query

    def __format_research_interests(self, interests):
        """Format a list of research interests into a properly structured string."""
        if not interests:
            return ""
        if len(interests) == 1:
            return interests[0]
        return ", ".join(interests[:-1]) + ", and " + interests[-1]


    def get_answer(self, sample_id, target_node_idx):
        answer = self.author_nodes.loc[self.author_nodes['node_idx'] == sample_id, 'Research_Interests'].values[0]
        
        return f"The author with id {target_node_idx} would likely describe research interests as {self.__format_research_interests(answer)}."
    
    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None):
        """
        Create a NetworkX graph from the subgraph edge index and node mapping.
        
        Args:
            sub_graph_edge_index (torch.Tensor): Edge index of the subgraph.
            node_mapping (dict): Mapping of node indices to node features.
            sub_graph_edge_mask (torch.Tensor): Mask for the subgraph edge index.
        
        Returns:
            nx.Graph: NetworkX graph object.
        """
        # Create a NetworkX graph
        G = nx.MultiDiGraph()
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_type = self.node_type_mapping[self.graph.node_type[raw_node_idx].item()]
            if node_type == 'paper':
                paper_idx = raw_node_idx - self.node_slices['paper'][0]
                
                G.add_node(
                    new_node_idx, type = 'paper', 
                    title=self.paper_nodes.at[paper_idx,'title'],
                    year=int(self.paper_nodes.at[paper_idx,'year']),
                    n_citation=int(self.paper_nodes.at[paper_idx,'n_citation'])
                )
                
            elif node_type == 'author':
                G.add_node(
                    new_node_idx, type = 'author'
                )
                
            elif node_type == 'venue':
                venue_idx = raw_node_idx - self.node_slices['venue'][0]
                G.add_node(
                    new_node_idx, type = 'venue', 
                    name = self.venues.at[venue_idx,'name_d']
                )
        
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            etype_id = self.graph.edge_type[edge_idx].item()
            edge_type = self.edge_type_mapping[etype_id]
            
            G.add_edge(src, dst, type = edge_type)
            
        return G
    