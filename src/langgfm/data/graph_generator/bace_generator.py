import os
import json
import networkx as nx
import pandas as pd


from .base_generator import GraphTaskGraphGenerator


@GraphTaskGraphGenerator.register("bace")
class BaceGraphGenerator(GraphTaskGraphGenerator):
    """
    BaceGraphGenerator: A generator for creating graphs 
    from the Bace dataset using NetworkX format.
    """
    
    def load_data(self):
        """
        Load the Bace dataset and preprocess required mappings.
        """
        self.root = './data/Bace'
        graphs = self.__parse_graphs(self.root, dataset_name = 'Bace')
        # print(f"{graphs=}")
        
        self.graph = graphs


def BACE(self, number_samples=None):
    df = pd.read_csv("./GraphData/MoleculeNetDataset/bace.csv",index_col=0)
    for row_idx, row_content in tqdm(df.iterrows()):
        graph = row_content['graph']
        G = smiles2graph(graph)
        nxgs = [G]

        graph_idx = row_content['molecule_index']
        answer = row_content['label']

        query = "BACE1 is an aspartic-acid protease important in the pathogenesis of Alzheimer's disease, and in the formation of myelin sheaths. It cleaves amyloid precursor protein (APP) to reveal the N-terminus of the beta-amyloid peptides. The beta-amyloid peptides are the major components of the amyloid plaques formed in the brain of patients with Alzheimer's disease (AD). Since BACE mediates one of the cleavages responsible for generation of AD, it is regarded as a potential target for pharmacological intervention in AD. BACE1 is a member of family of aspartic proteases. Same as other aspartic proteases, BACE1 is a bilobal enzyme, each lobe contributing a catalytic Asp residue, with an extended active site cleft localized between the two lobes of the molecule. Is this molecule effective to the assay?" 

        if answer == "No":
            answer = "No, this molecule is not effective to this assay."
        elif answer == "Yes":
            answer = "Yes, this molecule is effective to this assay."
        
        yield nxgs, query, answer, f"GraphIndex({graph_idx})"