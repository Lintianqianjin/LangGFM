import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator._base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("degree_counting")
class DegreeCountingGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with degree_counting.
    """
    def __init__(self, task='degree_counting'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(*[*query_entity,label])


if __name__ == '__main__':
    generator = DegreeCountingGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
    # print(generator.describe())

    # import networkx as nx
    # from networkx.readwrite import json_graph
    # G = nx.Graph(directed=True)
    # # G.add_edge(1, 2)
    # G.add_edges_from([[1, 2],[3,1]])
    # G_text = json_graph.node_link_data(G)
    # print(f"{G_text=}")
    # print(f"{G.edges=}")

    # G_new = json_graph.node_link_graph(G_text)
    # print(f"{G.edges=}")

    # G = nx.MultiDiGraph(G)
    # print(f"{G.edges=}")