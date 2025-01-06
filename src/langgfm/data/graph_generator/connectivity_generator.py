import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator.base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("connectivity")
class ConnectivityGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with connectivity.
    """
    def __init__(self, task='connectivity'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)


if __name__ == '__main__':
    generator = ConnectivityGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    # tmp
    # print(G.nodes[24])
    print(G.edges)
    print(metadata)
    print(generator.describe())