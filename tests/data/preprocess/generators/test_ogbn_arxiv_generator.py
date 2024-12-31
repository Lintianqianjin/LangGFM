import unittest

import networkx as nx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))
from langgfm.data.preprocess.generators.ogbn_arxiv_generator import OgbnArxivGraphGenerator

class TestOgbnArxivGraphGenerator(unittest.TestCase):

    def setUp(self):
        """
        Initialize the generator for testing.
        """
        self.generator = OgbnArxivGraphGenerator(num_hops=2)

    def test_load_data(self):
        """
        Test if the data is loaded correctly.
        """
        self.assertIsNotNone(self.generator.graph, "Graph should be loaded.")
        self.assertIsNotNone(self.generator.all_idx, "Index list should be initialized.")
        self.assertTrue(len(self.generator.all_idx) > 0, "All index list should not be empty.")
        self.assertGreater(len(self.generator.node_id_to_title_mapping), 0, "Paper title mapping should be populated.")
        print("Data loaded successfully.")

    def test_generate_graph(self):
        """
        Test if the k-hop subgraph is generated correctly.
        """
        sample_id = self.generator.all_idx[0].item()  # Select a sample node ID
        G, metadata = self.generator.generate_graph(sample_id=sample_id)

        # Validate graph structure
        self.assertIsInstance(G, nx.MultiDiGraph, "Generated graph should be a NetworkX MultiDiGraph.")
        self.assertGreater(len(G.nodes), 0, "Generated graph should have nodes.")
        self.assertGreater(len(G.edges), 0, "Generated graph should have edges.")
        
        # Validate metadata
        self.assertIn("query", metadata, "Metadata should include 'query'.")
        self.assertIn("label", metadata, "Metadata should include 'label'.")
        self.assertIn("target_node", metadata, "Metadata should include 'target_node'.")
        self.assertEqual(metadata["target_node"], 0, "Target node should be correctly mapped.")

        print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")


if __name__ == '__main__':
    import unittest
    unittest.main()
