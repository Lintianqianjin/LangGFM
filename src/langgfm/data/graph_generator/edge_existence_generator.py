from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("edge_existence")
class EdgeExistenceGraphGenerator(StructuralTaskGraphGenerator):
    """
    EdgeExistenceGraphGenerator: A generator for graphs task with edge existence.
    """
    def __init__(self, task='edge_existence'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format("Yes" if label else "No")




