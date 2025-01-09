import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator._base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("node_attribute_retrieval")
class NodeAttributeRetrievalGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with node_attribute_retrieval.
    """
    def __init__(self, task='node_attribute_retrieval'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity):
        return self.config['answer_format'].format(*[*query_entity,label])


if __name__ == '__main__':
    generator = NodeAttributeRetrievalGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    # tmp
    # print(G.nodes[24])
    print(G.edges)
    print(metadata)
    print(generator.describe())