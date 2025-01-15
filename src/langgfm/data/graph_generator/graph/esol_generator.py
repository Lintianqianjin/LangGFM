import os
import json
import networkx as nx
import pandas as pd


from .._base_generator import GraphTaskGraphGenerator
from ..utils.molecule_utils import smiles2graph

@GraphTaskGraphGenerator.register("esol")
class ESOLGraphGenerator(GraphTaskGraphGenerator):
    """
    BaceGraphGenerator: A generator for creating graphs 
    from the Bace dataset using NetworkX format.
    """
    
    directed = False
    has_node_attr = True
    has_edge_attr = True
    
    def load_data(self):
        """
        Load the ESOL dataset and preprocess required mappings.
        """
        self.root = './data/esol'
        self.df = pd.read_csv(f"{self.root}/esol.csv",index_col=0)
        self.all_samples = set(self.df['molecule_index'].tolist())
    
    def get_query(self, **kwargs):
        query = ("Aqueous solubility is one of the most critical properties of a therapeutic compound. "
        "The calculation formula of the 10-based logarithm of the aqueous solubility ($\log S$) is as follows: "
        "$\log S = 0.16 - 0.63 \log P - 0.0062 MW + 0.066 RB - 0.74 Apolar$, where $S$ is Aqueous solubility (mol/L), "
        "$\log P$ is the octanol-water partition coefficient (lipophilicity indicator), $MW$ is Molecular weight (g/mol), "
        "$RB$ is the number of rotatable bonds (flexibility measure) and $Apolar$ is aromatic proportion, i.e., the "
        "proportion of heavy atoms in the molecule that are in anaromatic ring. "
        "Please estimate the $\log S$ of the given compound. " )
            
        return query

    def get_answer(self, sample_id):
        
        filtered_df = self.df.loc[self.df['molecule_index'] == sample_id]['label']
        
        if len(filtered_df) == 1:
            label = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for molecule_index={sample_id}")

        answer = f"The $\log S$ of this compound is likely {label}."
        
        return answer
    
    def create_networkx_graph(self, sample_id):
        filtered_df = self.df.loc[self.df['molecule_index'] == sample_id]['graph']
        if len(filtered_df) == 1:
            graph = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for molecule_index={sample_id}")
    
        G = smiles2graph(graph)
        
        return G
            
    @property
    def graph_description(self):
        return "This graph is a molecule graph, where explicit hydrogen atoms " \
            "have been removed. Nodes represent atoms and edges represent chemical bonds. "