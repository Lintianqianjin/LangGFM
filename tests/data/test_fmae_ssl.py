# write a test file to test NodeFeatureMaskedAutoencoder and EdgeFeatureMaskedAutoencoder as the folllowing:
# 1. generate a graph using OgbnArxivGraphGenerator
# 2. using NodeFeatureMaskedAutoencoder and EdgeFeatureMaskedAutoencoder to generate ssl samples based on the graph
# 3. using GraphTextualizer to convert the ssl samples to text for better visualization

import unittest
import networkx as nx
import re
import sys
import os
import json

from langgfm.data.graph_generator import OgbnArxivGraphGenerator
from langgfm.data.ssl_tasks.fmae_ssl import NodeFeatureMaskedAutoencoder
from langgfm.data.ssl_tasks.fmae_ssl import EdgeFeatureMaskedAutoencoder
from langgfm.data.graph_text_transformation.nxg_to_text import GraphTextualizer

class TestFMAEAutoencoder(unittest.TestCase):
    
    def setUp(self):
        """
        Initialize the generator and NodeFeatureMaskedAutoencoder and EdgeFeatureMaskedAutoencoder for testing.
        """
        self.generator = OgbnArxivGraphGenerator(num_hops=2, sampling=True, neighbor_size=15, random_seed=0)
        self.node_autoencoder = NodeFeatureMaskedAutoencoder()
        self.edge_autoencoder = EdgeFeatureMaskedAutoencoder()
        dataset_type_path = os.path.join(os.path.dirname(__file__), '../../src/langgfm/configs/dataset_type.json')
        with open(dataset_type_path, "r") as f:
            self.dataset_type = json.load(f)
        self.textualizer = GraphTextualizer()
        
    def test_generate_ssl_sample(self):
        """
        Test the SSL sample generation pipeline.
        """
        # generate a graph
        sample_id = self.generator.all_idx[0].item()  # Select a sample node ID
        G, metadata = self.generator.generate_graph(sample_id=sample_id)
        original_graph_text = self.textualizer.export(G, format='json', **self.dataset_type['ogbn_arxiv'])
        print(f"{original_graph_text=}")

        # generate SSL samples with the NodeFeatureMaskedAutoencoder
        G_ssl_node = self.node_autoencoder.generate_sample(G)
        # the structure of G_ssl_node is: 
        # {
        #     'modified_graph': modified_graph,
        #     'query': query['query_text'],
        #     'answer': answer
        # }
        
        # validate the generated SSL samples
        # G_ssl_node
        self.assertIsInstance(G_ssl_node, dict, "SSL sample should be a dictionary.")
        self.assertIsInstance(G_ssl_node['modified_graph'], nx.MultiDiGraph, "Modified graph should be a NetworkX MultiDiGraph.")
        self.assertGreater(len(G_ssl_node['modified_graph'].nodes), 0, "Modified graph should have nodes.")
        self.assertGreater(len(G_ssl_node['modified_graph'].edges), 0, "Modified graph should have edges.")
        self.assertIsInstance(G_ssl_node['query'], str, "Query should be a string.")
        self.assertIsInstance(G_ssl_node['answer'], str, "Answer should be a string.")
        self.assertGreater(len(G_ssl_node['query']), 0, "Query should not be empty.")
        self.assertGreater(len(G_ssl_node['answer']), 0, "Answer should not be empty.")
        
        # print the generated SSL sample
        print(f"Node SSL sample:")
        print(f"Query: {G_ssl_node['query']}")
        print(f"Answer: {G_ssl_node['answer']}\n\n")
        
        # GraphTextualizer here
        # convert the SSL sample to different formats
        for format in ['json', 'graphml', 'gml', 'table']:
            graph_text = self.textualizer.export(G_ssl_node['modified_graph'], format=format, **self.dataset_type['ogbn_arxiv'])
            self.assertIsInstance(graph_text, str, "Graph text should be a string.")
            self.assertGreater(len(graph_text), 0, "Graph text should not be empty.")
            print(f"\n\nGraph text in {format} format:")
            print(graph_text)
            
        
        # validate whether the answer is correct
        
        # # generate SSL samples with the EdgeFeatureMaskedAutoencoder
        # G_ssl_edge = self.edge_autoencoder.generate_sample(G)
        # # the structure of G_ssl_edge is
        # # {
        # #     'modified_graph': modified_graph,
        # #     'query': query['query_text'],
        # #     'answer': answer
        # # }
        
        # # validate the generated SSL samples
        # # G_ssl_edge
        # self.assertIsInstance(G_ssl_edge, dict, "SSL sample should be a dictionary.")
        # self.assertIsInstance(G_ssl_edge['modified_graph'], nx.MultiDiGraph, "Modified graph should be a NetworkX MultiDiGraph.")
        # self.assertGreater(len(G_ssl_edge['modified_graph'].nodes), 0, "Modified graph should have nodes.")
        # self.assertGreater(len(G_ssl_edge['modified_graph'].edges), 0, "Modified graph should have edges.")
        # self.assertIsInstance(G_ssl_edge['query'], str, "Query should be a string.")
        # self.assertIsInstance(G_ssl_edge['answer'], str, "Answer should be a string.")
        # self.assertGreater(len(G_ssl_edge['query']), 0, "Query should not be empty.")
        # self.assertGreater(len(G_ssl_edge['answer']), 0, "Answer should not be empty.")
        
        # # print the generated SSL sample
        # print(f"Edge SSL sample:")
        # print(f"Query: {G_ssl_edge['query']}")
        # print(f"Answer: {G_ssl_edge['answer']}")
        
        # # validate whether the answer is correct
        
        

if __name__ == '__main__':
    import unittest
    unittest.main()



