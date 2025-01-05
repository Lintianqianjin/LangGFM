import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import unittest
import random
import networkx as nx

import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.data.graph_generator.base_generator import NodeGraphGenerator, EdgeGraphGenerator
# for registering the real graph generators
from langgfm.data.graph_generator import *
from langgfm.utils.random_control import set_seed


class TestRealGraphGenerator(unittest.TestCase):

    def setUp(self):
        """
        Initialize the generator for testing.
        """
        self.datasets = ['yelp_review',] # 'movielens1m', 'usa_airport', 'twitch', 'ogbn_arxiv','wikics','aminer', 'ogbn_arxiv','ogbl_vessel'
        # load the generator configuration file
        config_path = os.path.join(os.path.dirname(__file__), '../../src/langgfm/configs/graph_generator.json')
        with open(config_path, 'r') as f:
            self.graph_generator_configs = json.load(f)
            
        self.generators = {}
        for dataset in self.datasets:
            print(f"Creating generator for {dataset}...")
            # get the configuration for the dataset
            config = self.graph_generator_configs.get(dataset, {})
            # create the generator instance
            if config['task_level'] == 'node':
                self.generators[dataset] = NodeGraphGenerator.create(dataset, **config)
            elif config['task_level'] == 'edge':
                self.generators[dataset] = EdgeGraphGenerator.create(dataset, **config)
            else:
                raise ValueError("Invalid task type.")
            
    def test_load_data(self):
        """
        Test if the data is loaded correctly.
        """
        for dataset in self.datasets:
            generator = self.generators[dataset]
            self.assertIsNotNone(generator.graph, "Graph should be loaded.")
            # check all_samples
            self.assertGreater(len(generator.all_samples), 0, "Samples should be populated.")
            
            print("Data loaded successfully.")

    def test_generate_graph(self):
        """
        Test if the k-hop subgraph is generated correctly.
        """
        for dataset in self.datasets:
            generator = self.generators[dataset]
            
            set_seed(42)
            samples = random.sample(list(generator.all_samples),k=3)  # Select a sample node ID
            print(f"{samples=}")
            for sample in samples:
                G, metadata = generator.generate_graph(sample=sample)
                
                print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
                
                print(f"{metadata=}")

                # Validate graph structure
                self.assertIsInstance(G, nx.MultiDiGraph, "Generated graph should be a NetworkX MultiDiGraph.")
                self.assertGreater(len(G.nodes), 0, "Generated graph should have nodes.")
                self.assertGreater(len(G.edges), 0, "Generated graph should have edges.")
                
                
            # # validate answer in metadata by comparing with the label of the target node in the original graph
            # raw_sample_id = metadata['raw_sample_id']
            # label = generator.
            # self.assertEqual(answer, metadata['answer'], "Answer should be correct.")
            # print(f"Answer: {answer}")
            # print(f"Answer in metadata: {metadata['answer']}")
            # print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            # print(f"Answer: {answer}")
            # print(f"Answer in metadata: {metadata['answer']}")
            # print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            # print(f"Answer: {answer}")
            # print(f"Answer in metadata: {metadata['answer']}")
            # print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            
            
            


if __name__ == '__main__':
    import unittest
    unittest.main()
