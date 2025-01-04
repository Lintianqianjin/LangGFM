import json
import networkx as nx

import torch
import pandas as pd
from torch_geometric.utils import to_undirected

from .base_generator import NodeGraphGenerator


@NodeGraphGenerator.register("twitch")
class TwitchGraphGenerator(NodeGraphGenerator):
    """
    TwitchGraphGenerator: A generator for creating k-hop subgraphs 
    from the Twitch dataset using NetworkX format.
    """

    def load_data(self):
        """
        Load the Twitch dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        
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

        self.all_samples = set(range(self.graph.num_nodes))
        
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

    def create_networkx_graph(self, sub_graph_edge_index, node_mapping, sub_graph_edge_mask=None):
        '''
        Create a NetworkX graph from the sampled subgraph.'''
        
        G = nx.MultiDiGraph()
        
        for raw_node_idx, new_node_idx in node_mapping.items():
            G.add_node(new_node_idx, **self.node_idx_feature_mapping[raw_node_idx])
            
        for edge_idx in range(sub_graph_edge_index.size(1)):
            src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
            dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
            G.add_edge(src, dst)
        
        return G