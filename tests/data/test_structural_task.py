from abc import ABC

import unittest
import random
import networkx as nx
import json

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.data.graph_generator.edge_existence_generator import EdgeExistenceGraphGenerator

class TestStructuralTaskGenerator(unittest.TestCase, ABC):

    def setUp(self):
        """
        Initialize the generator for testing.
        """
        self.generator = EdgeExistenceGraphGenerator()

    def test_load_data(self):
        """
        Test if the data is loaded correctly.
        """
        self.assertIsNotNone(self.generator.graphs, "Graph should be loaded.")
        self.assertIsNotNone(self.generator.labels, "Index list should be initialized.")
        self.assertEqual(len(self.generator.graphs), len(self.generator.labels), "Graph and label length should match.")

        # self.assertTrue(len(self.generator.all_idx) > 0, "All index list should not be empty.")
        # self.assertGreater(len(self.generator.node_id_to_title_mapping), 0, "Paper title mapping should be populated.")
        print("Data loaded successfully.")

    def test_generate_graph(self):
        """
        Test if the graph is generated correctly.
        """
        # randomly select a sample ID, which from range(0, len(self.generator.graphs))
        sample_id = random.choice(range(0, len(self.generator.graphs)))

        # sample_id = self.generator.graphs  # Select a sample node ID
        G, metadata = self.generator.generate_graph(sample_id=sample_id)

        # Validate graph structure
        self.assertIsInstance(G, nx.MultiDiGraph, "Generated graph should be a NetworkX MultiDiGraph.")
        self.assertGreater(len(G.nodes), 0, "Generated graph should have nodes.")
        self.assertGreater(len(G.edges), 0, "Generated graph should have edges.")
        
        # Validate metadata
        self.assertIsInstance(metadata, dict, "Metadata should be a dictionary.")
        print(metadata['main_task'])
        # self.assertIn("query", metadata, "Metadata should include 'query'.")
        # self.assertIn("label", metadata, "Metadata should include 'label'.")
        # self.assertIn("target_node", metadata, "Metadata should include 'target_node'.")
        # self.assertEqual(metadata["target_node"], 0, "Target node should be correctly mapped.")

        print(f"Generated graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")


# --------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import unittest
import networkx as nx
import json
from langgfm.data.graph_text_transformation.text_to_nxg import TextualizedGraphLoader


"""
Here are test cases for all the graph generators. The tests aim to check 
whether the generated graphs are valid and the answers are correctly 
generated. Then the test for each generator should include the following:
1. check whether the generated graph is a valid NetworkX graph.
2. check whether the generated graph follows the defined attributes.
3. check whether the metadata contains the required keys.
4. check whether the answer is correctly generated.
5. check whether the answer is in the expected format.
6. check the splits of dataset.

"""

class BaseTest(unittest.TestCase, ABC):
    """
    Abstract base test class for graph generators.
    Provides common setup, utility methods, and enforces abstract methods for task-specific tests.
    """
    def setUp(self):
        """
        Set up common testing resources.
        """
        self.root_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../src/langgfm/data/outputs'
        ))
        self.graph_loader = TextualizedGraphLoader(directed=True, multigraph=True)
        self.graph_format = None


    def load_data(self):
        """
        Load the data for testing.
        """
        pass

    def validate_graph_structure(self, graph_text):
        """
        Utility to validate the structure of a generated graph.
        Args:
            graph (nx.Graph): The generated graph.
        """
        graph = self.graph_loader.load_graph_from_text(graph_text, self.graph_format)
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertGreater(len(graph.nodes), 0, "Graph should have at least one node.")
        self.assertGreater(len(graph.edges), 0, "Graph should have at least one edge.")

    def is_multidigraph_undirected_structure(self, graph):
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("The input graph must be a NetworkX MultiDiGraph.")

        for u, v in graph.edges():
            if not graph.has_edge(v, u):
                return False
        return True

    def validate_metadata(self, metadata, required_keys):
        """
        Utility to validate metadata generated along with the graph.
        Args:
            metadata (dict): The metadata dictionary.
            required_keys (list): List of required keys in the metadata.
        """
        self.assertIsInstance(metadata, dict)
        for key in required_keys:
            self.assertIn(key, metadata, f"Metadata should contain the key: {key}")

    # @abstractmethod
    # def test_generate_graph(self):
    #     """
    #     Abstract method for testing the graph generation.
    #     Must be implemented by subclasses.
    #     """
    #     pass

    @abstractmethod
    def test_answer_generation(self):
        """
        Abstract method for testing answer generation.
        Must be implemented by subclasses.
        """
        pass


class TestDegreeCountingGraphGenerator(BaseTest):
    """
    Test class for the DegreeCountingGraphGenerator.
    Implements the abstract methods from BaseTest.
    """
    def setUp(self, task="degree_counting", format_="gml"):
        super().setUp()
        # self.task = task
        # self.graph_format = format_
        self.load_data()

    def load_data(self):
        self.train_set = json.load(open(os.path.join(self.root_dir, self.task, self.graph_format, 'train.json')))
        self.val_set = json.load(open(os.path.join(self.root_dir, self.task, self.graph_format, 'val.json')))
        self.test_set = json.load(open(os.path.join(self.root_dir, self.task, self.graph_format, 'test.json')))

        self.samples = random.sample(self.train_set, 1) + random.sample(self.val_set, 1) + random.sample(self.test_set, 1)
        return self.samples

    def validate_graph_structure(self, graph_text):
        """
        Utility to validate the structure of a generated graph.
        Args:
            graph (nx.Graph): The generated graph.
        """
        graph = self.graph_loader.load_graph_from_text(graph_text, self.graph_format)
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertGreater(len(graph.nodes), 0, "Graph should have at least one node.")
        self.assertGreater(len(graph.edges), 0, "Graph should have at least one edge.")

    def test_answer_generation(self):
        """
        Test the _generate_answer method.
        """
        label = 42
        query_entity = [1, 2]
        answer = self.generator._generate_answer(label, query_entity=query_entity)
        expected_answer = self.generator.config['answer_format'].format(*[*query_entity, label])
        self.assertEqual(answer, expected_answer, "Generated answer does not match the expected format.")






if __name__ == '__main__':
    # import unittest
    # unittest.main()
    print("test".format(*[]))



    
