# write a test file to test the folllowing:
# 1. generate a graph using OgbnArxivGraphGenerator
# 2. convert the graph to different formats using GraphTextualizer
# 3. save the graph texts to files
# 4. load the graph texts from files to networkx using TextualizedGraphLoader
# 5. compare the graph loaded from files with the original graph

import unittest
import tempfile
import networkx as nx
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.data.graph_generator.ogbn_arxiv_generator import OgbnArxivGraphGenerator
from langgfm.data.graph_text_transformation.nxg_to_text import GraphTextualizer
from langgfm.data.graph_text_transformation.text_to_nxg import TextualizedGraphLoader

class TestGraphTextTransformation(unittest.TestCase):
    
    def setUp(self):
        """
        Initialize the generator and textualizer for testing.
        """
        self.generator = OgbnArxivGraphGenerator(num_hops=2, sampling=True, neighbor_size=15, random_seed=0)
        dataset_type_path = os.path.join(os.path.dirname(__file__), '../../src/langgfm/configs/dataset_type.json')
        with open(dataset_type_path, "r") as f:
            self.dataset_type = json.load(f)
        self.textualizer = GraphTextualizer()
        self.loader = TextualizedGraphLoader()

    def test_graph_text_transformation(self):
        """
        Test the graph text transformation pipeline.
        """
        # generate a graph
        sample_id = self.generator.all_idx[0].item()  # Select a sample node ID
        G, metadata = self.generator.generate_graph(sample_id=sample_id)

        # convert the graph to different formats
        for format in ['json', 'graphml', 'gml', 'table']:
            graph_text = self.textualizer.export(G, format=format, **self.dataset_type['ogbn_arxiv'])
            self.assertIsInstance(graph_text, str, "Graph text should be a string.")
            self.assertGreater(len(graph_text), 0, "Graph text should not be empty.")
            print(f"Graph text in {format} format:")
            print(graph_text)

            # save the graph text to a temporaty file for testing, using tempfile.NamedTemporaryFile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(graph_text)
                filename = f.name
            print(f"Graph text saved to {filename}.")
            
            # load the graph text from the file
            with open(filename, 'r') as f:
                graph_text_saved = f.read()
                G_loaded = self.loader.load_graph_from_text(graph_text_saved, format_=format)
            self.assertIsInstance(G_loaded, nx.MultiDiGraph, f"Loaded graph from {format} should be a NetworkX MultiDiGraph.")
            self.assertGreater(len(G_loaded.nodes), 0, f"Loaded graph from {format} should have nodes.")
            self.assertGreater(len(G_loaded.edges), 0, f"Loaded graph from {format} should have edges.")
            print(f"Graph loaded from {format} with {len(G_loaded.nodes)} nodes and {len(G_loaded.edges)} edges.")

            # compare the loaded graph with the original graph
            self.assertEqual(set(G.nodes), set(G_loaded.nodes), f"Graph loaded from {format}. Node sets should match.")
            self.assertEqual(set(G.edges), set(G_loaded.edges), f"Graph loaded from {format}. Edge sets should match.")
            print("Graph loaded successfully.")
            
            # convert the loaded graph to graph_text again
            graph_text_loaded = self.textualizer.export(G_loaded, format=format, **self.dataset_type['ogbn_arxiv'])
            self.assertEqual(graph_text, graph_text_loaded, f"Graph loaded from {format}. Graph text should match.")
            print(f"Graph text loaded in {format} format:")
            print(graph_text_loaded)
            
            # remove the temporary file
            os.remove(filename)
            print(f"Temporary file {filename} removed.")



if __name__ == '__main__':
    import unittest
    unittest.main()
