import unittest
import random
import networkx as nx

import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.data.graph_generator.base_generator import NodeGraphGenerator
# for registering the real graph generators
from langgfm.data.graph_generator import *
from langgfm.utils.random_control import set_seed

class TestRealGraphGenerator(unittest.TestCase):

    def setUp(self):
        """
        Initialize the generator for testing.
        """
        self.datasets = ['usa_airport'] # 'twitch', 'ogbn_arxiv','wikics','aminer'
        # load the generator configuration file
        config_path = os.path.join(os.path.dirname(__file__), '../../src/langgfm/configs/graph_generator.json')
        with open(config_path, 'r') as f:
            graph_generator_configs = json.load(f)
            
        self.generators = {}
        for dataset in self.datasets:
            print(f"Creating generator for {dataset}...")
            # get the configuration for the dataset
            config = graph_generator_configs.get(dataset, {})
            # create the generator instance
            self.generators[dataset] = NodeGraphGenerator.create(dataset, **config)
        
    def test_load_data(self):
        """
        Test if the data is loaded correctly.
        """
        for dataset in self.datasets:
            generator = self.generators[dataset]
            self.assertIsNotNone(generator.graph, "Graph should be loaded.")
            # check all_sample_ids
            self.assertGreater(len(generator.all_sample_ids), 0, "Sample IDs should be populated.")
            print("Data loaded successfully.")

    def test_generate_graph(self):
        """
        Test if the k-hop subgraph is generated correctly.
        """
        for dataset in self.datasets:
            generator = self.generators[dataset]
            set_seed(42)
            sample_id = random.choice(list(generator.all_sample_ids))  # Select a sample node ID
            print(f"{sample_id=}")
            G, metadata = generator.generate_graph(sample_id=sample_id)
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
