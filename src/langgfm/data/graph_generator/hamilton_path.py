import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator.base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("hamilton_path")
class HamiltonPathGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with hamilton_path.
    """
    def __init__(self, task='hamilton_path'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)


if __name__ == '__main__':
    generator = HamiltonPathGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    print(G.edges)
    print(metadata)
    print(generator.describe())