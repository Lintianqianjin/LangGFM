# write a test file to test TopologyAutoencoder as the folllowing:
# 1. generate a graph using OgbnArxivGraphGenerator
# 2. using TopologyAutoencoder to generate a ssl sample based on the graph

import unittest
import networkx as nx
import random
import re
import sys
import os

from langgfm.data.graph_generator import OgbnArxivGraphGenerator
from langgfm.data.ssl_tasks.tae_ssl import TopologyAutoencoder
from langgfm.utils.random_control import set_seed


import logging
logger = logging.getLogger("root")


class TestTopologyAutoencoder(unittest.TestCase):
    
    def setUp(self):
        """
        Initialize the generator and TopologyAutoencoder for testing.
        """
        self.generator = OgbnArxivGraphGenerator(num_hops=2, sampling=True, neighbor_size=15, random_seed=0)
        # initialize two TopologyAutoencoders, one is created with distingushing_direction=True, the other is created with distingushing_direction=False
        self.autoencoder = TopologyAutoencoder(distinguish_directions=True)
        self.autoencoder_no_direction = TopologyAutoencoder(distinguish_directions=False)
        
    def test_generate_ssl_sample(self):
        """
        Test the SSL sample generation pipeline.
        """
        set_seed(42)
        samples = random.sample(list(self.generator.all_samples),k=1)
        # generate a graph
        sample_id = samples[0]  # Select a sample node ID
        G, metadata = self.generator.generate_graph(sample_id=sample_id)

        # generate SSL samples with the two TopologyAutoencoders
        G_ssl = self.autoencoder.generate_sample(G)
        G_ssl_no_direction = self.autoencoder_no_direction.generate_sample(G)
        # the structure of G_ssl/G_ssl_no_direction is: 
        # {
        #     'modified_graph': modified_graph,
        #     'query': query['query_text'],
        #     'answer': answer
        # }
        
        # validate the generated SSL samples
        # G_ssl
        self.assertIsInstance(G_ssl, dict, "SSL sample should be a dictionary.")
        self.assertIsInstance(G_ssl['modified_graph'], nx.MultiDiGraph, "Modified graph should be a NetworkX MultiDiGraph.")
        self.assertGreater(len(G_ssl['modified_graph'].nodes), 0, "Modified graph should have nodes.")
        self.assertGreater(len(G_ssl['modified_graph'].edges), 0, "Modified graph should have edges.")
        self.assertIsInstance(G_ssl['query'], str, "Query should be a string.")
        self.assertIsInstance(G_ssl['answer'], str, "Answer should be a string.")
        self.assertGreater(len(G_ssl['query']), 0, "Query should not be empty.")
        self.assertGreater(len(G_ssl['answer']), 0, "Answer should not be empty.")
        
        # logger.debug the generated SSL sample
        logger.debug(f"SSL sample:")
        logger.debug(f"Query: {G_ssl['query']}")
        logger.debug(f"Answer: {G_ssl['answer']}")
        
        # validate whether the answer is correct
        
        # parse the predecessors or successors or neighbors of the query node in answer string using regex
        # the predecessors or successors or neighbors should be in a list of node IDs"
        # this regex should be refined since the [] in the answer string is not escaped
        answer_nodes = re.findall(r'\[.*?\]', G_ssl['answer'])
        # convert the string to a list of node IDs
        answer_nodes = [list(map(int, re.findall(r'\d+', node))) for node in answer_nodes]
        # if two list in answer_nodes, the first is predecessors, the second is successors
        # if one list in answer_nodes, the list is neighbors
        logger.debug(f"Answer nodes: {answer_nodes}")
        
        # parse the query node in query string using regex
        query_node = int(re.findall(r'\d+', G_ssl['query'])[0])
        logger.debug(f"Query node: {query_node}")
        
        # now using networkx to get the predecessors or successors or neighbors of the query node
        predecessors = list(G_ssl['modified_graph'].predecessors(query_node))
        successors = list(G_ssl['modified_graph'].successors(query_node))
        neighbors = list(nx.all_neighbors(G_ssl['modified_graph'], query_node))
        logger.debug(f"Predecessors: {predecessors}")
        logger.debug(f"Successors: {successors}")
        logger.debug(f"Neighbors: {neighbors}")
        
        # validate whether the answer is correct
        if len(answer_nodes) == 2:
            self.assertCountEqual(predecessors, answer_nodes[0], "Predecessors should be correct.")
            self.assertCountEqual(successors, answer_nodes[1], "Successors should be correct.")
        else:
            self.assertCountEqual(neighbors, answer_nodes[0], "Neighbors should be correct.")
       
        # G_ssl_no_direction
        self.assertIsInstance(G_ssl_no_direction, dict, "SSL sample should be a dictionary.")
        self.assertIsInstance(G_ssl_no_direction['modified_graph'], nx.MultiDiGraph, "Modified graph should be a NetworkX MultiDiGraph.")
        self.assertGreater(len(G_ssl_no_direction['modified_graph'].nodes), 0, "Modified graph should have nodes.")
        self.assertGreater(len(G_ssl_no_direction['modified_graph'].edges), 0, "Modified graph should have edges.")
        self.assertIsInstance(G_ssl_no_direction['query'], str, "Query should be a string.")
        self.assertIsInstance(G_ssl_no_direction['answer'], str, "Answer should be a string.")
        self.assertGreater(len(G_ssl_no_direction['query']), 0, "Query should not be empty.")
        self.assertGreater(len(G_ssl_no_direction['answer']), 0, "Answer should not be empty.")
        
        # logger.debug the generated SSL sample
        logger.debug(f"SSL no_direction sample:")
        logger.debug(f"Query: {G_ssl_no_direction['query']}")
        logger.debug(f"Answer: {G_ssl_no_direction['answer']}")
        
        # validate whether the answer is correct
        # parse the predecessors or successors or neighbors of the query node in answer string using regex
        # the predecessors or successors or neighbors should be in a list of node IDs"
        # this regex should be refined since the [] in the answer string is not escaped
        answer_nodes = re.findall(r'\[.*?\]', G_ssl_no_direction['answer'])
        # convert the string to a list of node IDs
        answer_nodes = [list(map(int, re.findall(r'\d+', node))) for node in answer_nodes]
        # if two list in answer_nodes, the first is predecessors, the second is successors
        # if one list in answer_nodes, the list is neighbors
        logger.debug(f"Answer nodes: {answer_nodes}")
        
        # parse the query node in query string using regex
        query_node = int(re.findall(r'\d+', G_ssl_no_direction['query'])[0])
        logger.debug(f"Query node: {query_node}")
        
        # now using networkx to get the predecessors or successors or neighbors of the query node
        predecessors = list(G_ssl_no_direction['modified_graph'].predecessors(query_node))
        successors = list(G_ssl_no_direction['modified_graph'].successors(query_node))
        neighbors = list(nx.all_neighbors(G_ssl_no_direction['modified_graph'], query_node))
        logger.debug(f"Predecessors: {predecessors}")
        logger.debug(f"Successors: {successors}")
        logger.debug(f"Neighbors: {neighbors}")
        
        # validate whether the answer is correct
        if len(answer_nodes) == 2:
            self.assertCountEqual(predecessors, answer_nodes[0], "Predecessors should be correct.")
            self.assertCountEqual(successors, answer_nodes[1], "Successors should be correct.")
        else:
            self.assertCountEqual(neighbors, answer_nodes[0], "Neighbors should be correct.")
        
        


if __name__ == '__main__':
    import unittest
    unittest.main()



