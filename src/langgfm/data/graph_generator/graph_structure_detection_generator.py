import os
import sys
import torch
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from ._base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml


@StructuralTaskGraphGenerator.register("graph_structure_detection")
class GraphStructureDetectionGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with graph_structure_detection.
    """
    def __init__(self, task='graph_structure_detection'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)


if __name__ == '__main__':
    # print("done")
    # exit()
    generator = GraphStructureDetectionGraphGenerator()

    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
    print(generator.describe())


