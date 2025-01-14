import os
import json
import networkx as nx
import pandas as pd


from ._base_generator import GraphTaskGraphGenerator
from .utils.molecule_utils import smiles2graph

@GraphTaskGraphGenerator.register("chebi20")
class ChEBI20GraphGenerator(GraphTaskGraphGenerator):
    """
    BaceGraphGenerator: A generator for creating graphs 
    from the ChEBI20 dataset using NetworkX format.
    """
    def load_data(self):
        """
        Load the ESOL dataset and preprocess required mappings.
        """
        self.root = './data/ChEBI-20-MM'
        self.dataset_train = pd.read_csv(f'{self.root}/train.csv', header=0)
        # print(f"{len(set(self.dataset_train['CID']))=}")
        # print(f"{self.dataset_train.shape=}")
        self.dataset_valid = pd.read_csv(f'{self.root}/validation.csv', header=0)
        # print(f"{len(set(self.dataset_valid['CID']))=}")
        # print(f"{self.dataset_valid.shape=}")
        self.dataset_test = pd.read_csv(f'{self.root}/test.csv', header=0)
        # print(f"{len(set(self.dataset_test['CID']))=}")
        # print(f"{self.dataset_test.shape=}")
        
        self.dataset = pd.concat([self.dataset_train, self.dataset_valid, self.dataset_test],axis=0)
        # print(f"{len(set(all_samples['CID']))=}")
        # print(f"{all_samples.shape=}")
        
        self.all_samples = set(self.dataset['CID'])
        # self.all_samples = set(self.df['molecule_index'].tolist())
    
    def get_query(self,**kwargs):
        query = ("Molecule Captioning task aims to generate a concise yet informative textual "
        "description of at given molecule based on its structural representation or molecular "
        "descriptors. This caption should encapsulate key aspects of the molecule, including "
        "its chemical structure, physicochemical and biochemical properties, functional "
        "attributes, biological activity, and potential applications. "
        "Please caption the given molecule."
        )
        return query

    def get_answer(self, sample_id):
        
        filtered_df = self.dataset.loc[self.dataset['CID'] == sample_id]['description']
        
        if len(filtered_df) == 1:
            molecule_desc = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for CID={sample}")

        answer = f"The given molecule can be described as: \"{molecule_desc}\"" # no need for period
        
        return answer
    
    def create_networkx_graph(self, sample_id):
        
        filtered_df = self.dataset.loc[self.dataset['CID'] == sample_id]['SMILES']
        
        if len(filtered_df) == 1:
            molecule_smile = filtered_df.iloc[0]  # get graph string
        else:
            raise ValueError(f"Expected one row, but found {len(filtered_df)} rows for CID={sample}")

        G = smiles2graph(molecule_smile)
        
        return G
            
    @property
    def graph_description(self):
        return "This is a chemical compound from ChEBI20 where node " \
            "represent atom and edge represent chemical bond."