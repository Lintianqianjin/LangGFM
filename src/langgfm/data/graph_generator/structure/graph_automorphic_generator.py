from .._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("graph_automorphic")
class GraphAutomorphicGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with graph_automorphic.
    """
    def __init__(self, task='graph_automorphic'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format("Yes" if label else "No")



