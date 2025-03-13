import os
import json
import networkx as nx
import pandas as pd


from .._base_generator import GraphTaskGraphGenerator
from ..utils.molecule_utils import smiles2graph

@GraphTaskGraphGenerator.register("hiv")
class BaceGraphGenerator(GraphTaskGraphGenerator):
    """
    BaceGraphGenerator: A generator for creating graphs 
    from the Bace dataset using NetworkX format.
    """
    
    directed = False
    has_node_attr = True
    has_edge_attr = True

    def load_data(self):
        """
        Load the Bace dataset and preprocess required mappings.
        """
        self.root = os.path.join(self.raw_data_dir, "hiv")
        self.df = pd.read_csv(f"{self.root}/hiv.csv",index_col=0)
        self.all_samples = list(self.df['molecule_index'].tolist())
    
    @property
    def graph_description(self):
        return "This graph is a molecule graph, where explicit hydrogen "\
            "atoms have been removed. Nodes represent atoms and edges represent chemical bonds."
            
    def get_query(self, **kwargs):
        query = ("In drug discovery, certain molecules can inhibit the replication of the HIV virus by targeting viral enzymes such as reverse transcriptase, integrase, or protease. Does the given molecule inhibit HIV virus replication? ")
        return query

    def get_answer(self, sample_id):
        filtered_df = self.df.loc[self.df['molecule_index'] == sample_id]['label']
        if len(filtered_df) == 1:
            label = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for molecule_index={sample_id}")
        
        if label == "No":
            answer = "<answer> No </answer>, the given molecule does not inhibit HIV virus replication."
        elif label == "Yes":
            answer = "<answer> Yes </answer>, the given molecule inhibits HIV virus replication."
        
        return answer
    
    def create_networkx_graph(self, sample_id):
        filtered_df = self.df.loc[self.df['molecule_index'] == sample_id]['graph']
        if len(filtered_df) == 1:
            graph = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for molecule_index={sample_id}")
    
        G = smiles2graph(graph)
        
        return G
            
    