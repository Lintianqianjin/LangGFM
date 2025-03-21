import json
import networkx as nx
import warnings
import torch
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from torch_geometric.utils import to_undirected

from .._base_generator import NodeTaskGraphGenerator


@NodeTaskGraphGenerator.register("twitch")
class TwitchGraphGenerator(NodeTaskGraphGenerator):
    """
    TwitchGraphGenerator: A generator for creating k-hop subgraphs 
    from the Twitch dataset using NetworkX format.
    """
    directed = False
    has_node_attr = True
    has_edge_attr = False
    
    def load_data(self):
        """
        Load the Twitch dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.graph = torch.load(f"{self.root}/Twitch/twitch.pt")
        self.graph.edge_index = to_undirected(self.graph.edge_index)
        
        self.nodes = pd.read_csv(f"{self.root}/Twitch/large_twitch_features.csv")
        self.nodes['dead_account'] = self.nodes['dead_account'].replace({1: True, 0: False}).infer_objects(copy=False)
        self.nodes['affiliate'] = self.nodes['affiliate'].replace({1: "Affiliate", 0: "Basic"})

        self.nodes = self.nodes.rename(columns={
            'views': "View_Count",
            "life_time": "Account_Lifetime",
            "created_at": "Creation_Date",
            "updated_at": "Last_Update_Date",
            "dead_account": "Inactive",
            "language": "Broadcaster_Language",
            "affiliate": "Affiliate_Status"
        })

        self.node_idx_feature_mapping = self.nodes[['View_Count','Account_Lifetime','Creation_Date','Last_Update_Date','Inactive','Broadcaster_Language','Affiliate_Status']].to_dict(orient='index') # key is node id (int)

        self.node_idx_label_mapping = self.nodes[['mature']].to_dict()['mature'] # key is node id (int)

        self.all_samples = list(range(self.graph.num_nodes))
    
    @property
    def graph_description(self):
        return "This graph is an ego-net of a Twitch user. "\
            "Nodes are users and links are friendships. "
    
    def get_query(self, target_node_idx: int) -> str:
        """
        Get the query for the main task based on the target_node_idx 
        in the networkx graph object.
        """
        query = f"""Please infer whether the user with node id of {target_node_idx} is a mature content streamer or gaming content streamer."""
        return query
        
    def get_answer(self, sample_id, target_node_idx: int) -> str:
        """
        Get the answer for a specific sample node.
        Args:
            sample_id (int): The index of the sample node.
        Returns:
            str: The answer for the sample node.
        """
        label = self.node_idx_label_mapping[sample_id]
        content_type = "mature" if label==1 else "gaming"
        return f"The user with node id {target_node_idx} is likely a {content_type} content streamer."

    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None):
        '''
        Create a NetworkX graph from the sampled subgraph.'''
        
        G = nx.MultiDiGraph()
        
        for raw_node_idx, new_node_idx in node_mapping.items():
            G.add_node(new_node_idx, type='user', **self.node_idx_feature_mapping[raw_node_idx])
            
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            G.add_edge(src, dst, type='friendship')
        
        return G
    
    