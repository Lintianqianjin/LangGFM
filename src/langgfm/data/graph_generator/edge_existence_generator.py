import os
import sys
import torch
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator.base_generator import InputGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@InputGraphGenerator.register("edge_existence")
class EdgeExistenceGraphGenerator(InputGraphGenerator):
    """
    EdgeExistenceGraphGenerator: A generator for creating graphs with random edge existence.
    """

    def __init__(self,):
        self.task = 'edge_existence'
        task_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "./../../configs/synthetic_graph_generation.yaml"))
        self.config = task_config_lookup[self.task]
        self.root = os.path.join(os.path.dirname(__file__), self.config['file_path'])
        self.load_data()


    def load_data(self):
        """
        Load the dataset and preprocess required mappings.
        """
        dataset = torch.load(self.root)
        self.graphs, self.labels = dataset['graphs'], dataset['labels']


    def generate_graph(self, sample_id: int) -> nx.Graph:
        """
        Generate a single NetworkX graph for a given sample ID.

        Args:
            sample_id (int): The ID of the sample to generate a graph for.

        Returns:
            nx.Graph: A NetworkX graph representing the specific sample.
        """
        G = json_graph.node_link_graph(self.graphs[sample_id], directed=True, multigraph=True)
        G = nx.MultiDiGraph(G)
        target_node = self.labels[sample_id][0]
        label = self.labels[sample_id][1]
        query = self.config['query_format'].format(*target_node)
        answer = self.config['answer_format'].format("Yes" if label else "No")
        meta_data = {
            "raw_sample_id": sample_id,
            "main_task":{
                "query": query,
                "anwser": answer,
                "target_node": target_node,
            }

        }

        return G, meta_data
        


if __name__ == '__main__':
    generator = EdgeExistenceGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
    print(generator.describe())