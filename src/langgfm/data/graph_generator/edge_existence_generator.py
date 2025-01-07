import os
import sys
import torch
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator._base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml


@StructuralTaskGraphGenerator.register("edge_existence")
class EdgeExistenceGraphGenerator(StructuralTaskGraphGenerator):
    """
    EdgeExistenceGraphGenerator: A generator for graphs task with edge existence.
    """
    def __init__(self, task='edge_existence'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format("Yes" if label else "No")


if __name__ == '__main__':
    generator = EdgeExistenceGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
    print(generator.describe())


