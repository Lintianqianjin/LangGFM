import unittest
import random
import networkx as nx

import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.data.graph_generator.base_generator import NodeGraphGenerator
# for registering the real graph generators
from langgfm.data.graph_generator import OgbnArxivGraphGenerator, WikicsGraphGenerator
from langgfm.utils.random_control import set_seed

class TestRealGraphGenerator(unittest.TestCase):

    def setUp(self):
        """
        Initialize the generator for testing.
        """
        self.datasets = ['ogbn_arxiv','wikics']
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
            print("Data loaded successfully.")

    def test_generate_graph(self):
        """
        Test if the k-hop subgraph is generated correctly.
        """
        set_seed(0)
        for dataset in self.datasets:
            generator = self.generators[dataset]
            sample_id = random.choice(list(range(generator.graph.num_nodes)))
            print(f"{sample_id=}")
            G, metadata = generator.generate_graph(sample_id=sample_id)
            print(f"{metadata=}")

            # Validate graph structure
            self.assertIsInstance(G, nx.MultiDiGraph, "Generated graph should be a NetworkX MultiDiGraph.")
            self.assertGreater(len(G.nodes), 0, "Generated graph should have nodes.")
            self.assertGreater(len(G.edges), 0, "Generated graph should have edges.")
            
            print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")


if __name__ == '__main__':
    import unittest
    unittest.main()
