from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("edge_counting")
class EdgeCountingGraphGenerator(StructuralTaskGraphGenerator):
    """
    NodeCountingGraphGenerator: A generator for graphs task with edge_counting.
    """
    def __init__(self, task='edge_counting'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)
