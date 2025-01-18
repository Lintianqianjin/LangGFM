import unittest
import os
import random
import networkx as nx

from langgfm.data.graph_generator._base_generator import StructuralTaskGraphGenerator
from langgfm.utils.random_control import set_seed


class TestStructureGraphGenerator(unittest.TestCase):

    def setUp(self):
        self.task = "hamilton_path" 
        self.generator = StructuralTaskGraphGenerator.create(self.task)

    def test_load_data(self):
        """
        Test if the data is loaded correctly.
        """
        # self.assertIsNotNone(self.generator.graph, "Graph should be loaded.")
        self.assertGreater(len(self.generator.all_samples), 0, "Samples should be populated.")
        self.assertEqual(len(self.generator.graphs), len(self.generator.labels), "Graphs and labels should have the same length.")
        # print(len(self.generator.graphs), len(self.generator.labels), "Graphs and labels should have the same length.")
        print("Data loaded successfully.")
    
    def test_generate_graph(self):
        """
        Test if the k-hop subgraph is generated correctly.
        """
        set_seed(42)
        samples = random.sample(list(self.generator.all_samples),k=3)
        for sample in samples:
            G, metadata = self.generator.generate_graph(sample_id=sample)
            
            print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            
            print(f"{metadata=}")

            # Validate graph structure
            self.assertIsInstance(G, nx.MultiDiGraph, "Generated graph should be a NetworkX MultiDiGraph.")
            self.assertGreater(len(G.nodes), 0, "Generated graph should have nodes.")
            self.assertGreater(len(G.edges), 0, "Generated graph should have edges.")
            
            # validate answer in metadata by comparing with the label of the target node in the original graph
            print(f"Query: {metadata['main_task']['query']}")
            print(f"Answer: {metadata['main_task']['answer']}")
            print(f"Query Entity: {metadata['main_task']['query_entity']}")

    def test_graph_properties(self):
        print(f"Graphs Description: {self.generator.graph_description}")


if __name__ == '__main__':
    unittest.main()