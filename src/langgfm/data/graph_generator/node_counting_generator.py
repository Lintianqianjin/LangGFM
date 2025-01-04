import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator.base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("node_counting")
class NodeCountingGraphGenerator(StructuralTaskGraphGenerator):
    """
    NodeCountingGraphGenerator: A generator for graphs task with node counting.
    """
    def __init__(self, task='node_counting'):
        super().__init__(task)

    def _generate_answer(self, label):
        return self.config['answer_format'].format("Yes" if label else "No")


if __name__ == '__main__':
    generator = NodeCountingGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
    print(generator.describe())