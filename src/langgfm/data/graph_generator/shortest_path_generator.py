from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("shortest_path")
class ShortestPathGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with shortest_path.
    """
    def __init__(self, task='shortest_path'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=[]):
        return self.config['answer_format'].format(*[*query_entity,label])
