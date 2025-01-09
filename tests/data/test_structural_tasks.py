from abc import ABC, abstractmethod
import unittest
import networkx as nx
import json
import sys
import os
import re
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.data.graph_text_transformation.text_to_nxg import TextualizedGraphLoader


class _BaseTest(unittest.TestCase):
    """
    Abstract base test class for graph generators.
    Provides common setup, utility methods, and enforces abstract methods for task-specific tests.
    """
    def setUp(self):
        """
        Set up common testing resources.
        """
        task = "node_counting"
        graph_format = "gml"
        self.root_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../src/langgfm/data/outputs',
            task, 
            graph_format
        ))
        self.graph_loader = TextualizedGraphLoader(directed=True, multigraph=True)
        self.graph_format = graph_format

    def select_samples(self):
        """
        Select one sample from each split.
        """
        train_set = json.load(open(os.path.join(self.root_dir, 'train.json')))
        val_set = json.load(open(os.path.join(self.root_dir, 'val.json')))
        test_set = json.load(open(os.path.join(self.root_dir, 'test.json')))

        self.samples = random.sample(train_set, 1) + random.sample(val_set, 1) + random.sample(test_set, 1)
        return self.samples

    def load_as_graph(self, graph_text):
        """
        Convert textual graph data to a NetworkX graph.
        """
        return self.graph_loader.load_graph_from_text(graph_text, self.graph_format)
    
    def test_dataset_splits(self):
        """
        Test whether the dataset splits have the correct sizes.
        """
        len_train = len(json.load(open(os.path.join(self.root_dir, 'train.json'))))
        len_val = len(json.load(open(os.path.join(self.root_dir, 'val.json'))))
        len_test = len(json.load(open(os.path.join(self.root_dir, 'test.json'))))

        self.assertEqual(len_train, 500, "Training set should be 500.")
        self.assertEqual(len_val, 100, "Validation set should be 100.")
        self.assertEqual(len_test, 200, "Test set should be 200.")

    def test_graph_structure(self, graph):
        """
        Test the structure of the generated graph.
        """
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertTrue(self.is_multidigraph_undirected_structure(graph), "Graph should be undirected.")
        self.assertGreater(len(graph.nodes), 0, "Graph should have at least one node.")
        self.assertGreater(len(graph.edges), 0, "Graph should have at least one edge.")

    # def test_metadata(self, metadata, required_keys):
    #     """
    #     Abstract method to validate metadata generated along with the graph.
    #     """
    #     pass

    # def test_graph_task(self, graph, anwser, query_entity=None):
    #     """
    #     Abstract method to test the label for this specific task.
    #     """
    #     pass

    # def test_pipeline(self):
    #     """
    #     General pipeline for testing each task's dataset files.
    #     1. Set up.
    #     2. Check the dataset splits.
    #     3. Select samples.
    #     4. Load the graph.
    #     5. Test the graph structure.
    #     6. Test the task-specific functionality.
    #     """
    #     self.test_dataset_splits()
    #     samples = self.select_samples()
        
    #     for sample in samples:
    #         graph_text = sample['graph_text']
    #         metadata = sample['metadata']
    #         answer = metadata['main_task']['label']
    #         query_entity = metadata['main_task']['query_entity']
            
    #         # Convert textual graph to NetworkX graph
    #         graph = self.load_as_graph(graph_text)
            
    #         # Test graph structure
    #         self.test_graph_structure(graph)
            
    #         # Test metadata
    #         self.test_metadata(metadata, ["main_task"])
            
    #         # Test task-specific functionality
    #         self.test_graph_task(graph, answer, query_entity)
        
    def is_multidigraph_undirected_structure(self, graph):
        """
        Check if a MultiDiGraph is structurally undirected.
        """
        for u, v in graph.edges():
            if not graph.has_edge(v, u):
                return False
        return True


# class TestDegreeCounting(_BaseTest):
#     """
#     Test case for the DegreeCounting task.
#     """
#     def setUp(self, task="degree_counting", format_="gml"):
#         super().setUp(task, format_)

#     def test_metadata(self, metadata, required_keys=["main_task"]):
#         """
#         Validate metadata for the degree counting task.
#         """
#         for key in required_keys:
#             self.assertIn(key, metadata, f"Metadata should contain key {key}.")
        
#         self.assertIn("degree", metadata['main_task']['query'], "Main task should be degree counting.")

#     def test_graph_task(self, graph, anwser, query_entity=None):
#         """
#         Test the label for the degree counting task.
#         """
#         target_node = query_entity[0]
#         extracted_anwser = int(re.findall(r'\d+', anwser)[0])  # Extract the degree from the answer
#         self.assertEqual(graph.degree[target_node], extracted_anwser, "Degree should be correctly counted.")


if __name__ == '__main__':
    # import unittest
    # unittest.main()
    print("test".format(*[]))



    
