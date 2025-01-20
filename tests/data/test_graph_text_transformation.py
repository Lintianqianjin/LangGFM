# write a test file to test the folllowing:
# 1. generate a graph using OgbnArxivGraphGenerator
# 2. convert the graph to different formats using GraphTextualizer
# 3. save the graph texts to files
# 4. load the graph texts from files to networkx using TextualizedGraphLoader
# 5. compare the graph loaded from files with the original graph

import unittest
import tempfile
import random
import networkx as nx
import json
import sys
import os

from langgfm.data.graph_generator import InputGraphGenerator
from langgfm.data.graph_text_transformation.nxg_to_text import GraphTextualizer
from langgfm.data.graph_text_transformation.text_to_nxg import TextualizedGraphLoader
import logging
logger = logging.getLogger("root")


class TestGraphTextTransformation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Initialize class-level configuration. 
        Parse dataset name from command-line arguments.
        """
        if len(sys.argv) > 1:
            cls.dataset_name = sys.argv.pop()  # Remove and get the last argument
        else:
            # 'ogbn_arxiv', 'wikics', 'aminer', 're_europe','oag_scholar_interest',  'usa_airport', 'twitch'
            # 'stack_elec','fb15k237', 'ogbl_vessel','movielens1m','yelp_review','ogbl_vessel'
            # 'explagraphs' ,'chebi20', 'esol', 'fingerprint', 'bace'
            cls.dataset_name = "re_europe"  # Default dataset name
    
    def setUp(self):
        """
        Initialize the generator and textualizer for testing.
        """
        self.generator = InputGraphGenerator.create(self.dataset_name, sampling=True, num_hops=2, neighbor_size=[5,3],random_seed = 0)
        # dataset_type_path = os.path.join(os.path.dirname(__file__), '../../src/langgfm/configs/dataset_type.json')
        # with open(dataset_type_path, "r") as f:
        #     self.dataset_type = json.load(f)
        self.textualizer = GraphTextualizer()
        self.loader = TextualizedGraphLoader()

    def test_graph_text_transformation(self):
        """
        Test the graph text transformation pipeline.
        """
        # generate a graph
        sample_id = random.choice(self.generator.all_samples)  # Select a sample node ID
        G, metadata = self.generator.generate_graph(sample_id=sample_id)
        logger.info(f"Sampled Graph {G.nodes=}")
        logger.info(f"Sampled Graph {G.edges=}")

        # convert the graph to different formats
        for format in ['json', 'graphml', 'gml', 'table']: # 'json', 'graphml', 'gml', 
            graph_text = self.textualizer.export(G, format=format, directed = self.generator.directed)
            self.assertIsInstance(graph_text, str, "Graph text should be a string.")
            self.assertGreater(len(graph_text), 0, "Graph text should not be empty.")
            logger.info(f"Graph text in {format} format:")
            logger.info(graph_text)

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
            logger.info(f"{set(G.edges)=}")
            logger.info(f"{set(G_loaded.edges)=}")
            self.assertEqual(set(G.edges), set(G_loaded.edges), f"Graph loaded from {format}. Edge sets should match.")
            print("Graph loaded successfully.")
            
            # convert the loaded graph to graph_text again
            graph_text_loaded = self.textualizer.export(G_loaded, format=format, directed = self.generator.directed)
            logger.info(graph_text)
            logger.info(graph_text_loaded)
            self.assertEqual(graph_text, graph_text_loaded, f"Graph loaded from {format}. Graph text should match.")
            # print(f"Graph text loaded in {format} format:")
            # print(graph_text_loaded)
            
            # remove the temporary file
            os.remove(filename)
            # print(f"Temporary file {filename} removed.")



if __name__ == '__main__':
    unittest.main()
