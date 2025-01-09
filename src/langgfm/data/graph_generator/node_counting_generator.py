from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("node_counting")
class NodeCountingGraphGenerator(StructuralTaskGraphGenerator):
    """
    NodeCountingGraphGenerator: A generator for graphs task with node counting.
    """
    def __init__(self, task='node_counting'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)


