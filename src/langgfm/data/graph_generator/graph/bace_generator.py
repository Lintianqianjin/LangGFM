import os
import json
import networkx as nx
import pandas as pd


from .._base_generator import GraphTaskGraphGenerator
from ..utils.molecule_utils import smiles2graph

@GraphTaskGraphGenerator.register("bace")
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
        self.root = os.path.join(self.raw_data_dir, "bace")
        self.df = pd.read_csv(f"{self.root}/bace.csv",index_col=0)
        self.all_samples = list(self.df['molecule_index'].tolist())
    
    @property
    def graph_description(self):
        return "This graph is a molecule graph, where explicit hydrogen "\
            "atoms have been removed. Nodes represent atoms and edges represent chemical bonds."
            
    def get_query(self, **kwargs):
        query = ("β-Secretase 1 (BACE-1) encodes a member of the peptidase A1 family of aspartic proteases. "
        # "Alternative splicing results in multiple transcript variants, at least one of which encodes a "
        # "preproprotein that is proteolytically processed to generate the mature protease. This transmembrane "
        # "protease catalyzes the first step in the formation of amyloid beta peptide from amyloid precursor protein. "
        # "Amyloid beta peptides are the main constituent of amyloid beta plaques, which accumulate in the brains of "
        # "human Alzheimer's disease patients. "
        "The active site of BACE1 is located in its extracellular domain and contains a typical aspartic protease catalytic "
        "site, composed of two conserved aspartic acid residues (Asp32 and Asp228), forming a catalytic dyad. This active site "
        "is situated in a highly hydrophilic cleft, enabling it to bind to the β-site of APP for specific cleavage. "
        "Key features of BACE1 inhibitors include: high affinity (stable binding to Asp32 and Asp228, mimicking APP binding), "
        "selectivity (avoiding inhibition of homologous proteins), good brain penetration (optimizing lipophilicity and reducing "
        "P-gp efflux), metabolic stability (prolonging half-life), and low side effects (minimizing non-Aβ-related impacts). "
        "Does the given molecule inhibit BACE-1?")
        return query

    def get_answer(self, sample_id):
        filtered_df = self.df.loc[self.df['molecule_index'] == sample_id]['label']
        if len(filtered_df) == 1:
            label = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for molecule_index={sample_id}")
        
        if label == "No":
            answer = "<answer> No </answer>, the given molecule is unlikely to inhibit BACE-1."
        elif label == "Yes":
            answer = "<answer> Yes </answer>, the given molecule is likely to inhibit BACE-1."
        
        return answer
    
    def create_networkx_graph(self, sample_id):
        filtered_df = self.df.loc[self.df['molecule_index'] == sample_id]['graph']
        if len(filtered_df) == 1:
            graph = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for molecule_index={sample_id}")
    
        G = smiles2graph(graph)
        
        return G
            
    