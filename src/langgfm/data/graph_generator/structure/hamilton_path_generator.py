from .._base_generator import StructuralTaskGraphGenerator

@StructuralTaskGraphGenerator.register("hamilton_path")
class HamiltonPathGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with hamilton_path.
    """
    def __init__(self, task='hamilton_path'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)


