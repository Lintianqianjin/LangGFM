import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from langgfm.data.graph_generator.base_generator import StructuralTaskGraphGenerator
from langgfm.data.build_synthetic_graph.utils import load_yaml

@StructuralTaskGraphGenerator.register("shortest_path")
class ShortestPathGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with shortest_path.
    """
    def __init__(self, task='shortest_path'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=[]):
        return self.config['answer_format'].format(*[*query_entity,label])

if __name__ == '__main__':
    generator = ShortestPathGraphGenerator()
    G, metadata = generator.generate_graph(0)
    print(G.nodes)
    # tmp
    # print(G.nodes[24])
    print(G.edges)
    print(metadata)
    print(generator.describe())