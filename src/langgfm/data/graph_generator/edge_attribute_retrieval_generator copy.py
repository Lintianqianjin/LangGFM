import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator.base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("edge_attribute_retrieval")
class EdgeAttributeRetrievalGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with edge_attribute_retrieval.
    """
    def __init__(self, task='edge_attribute_retrieval'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity):
        return self.config['answer_format'].format(*[*query_entity,label])


if __name__ == '__main__':
    generator = EdgeAttributeRetrievalGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
        # tmp
    # print(G.nodes[24])
    # check the weight of a specific edge
    print(G.get_edge_data(27, 36))
    # print(G.get_edge_data(27, 36)['weight'])
    import networkx as nx
    print('\n'.join(nx.generate_gml(G)))
    print(generator.describe())