from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("cycle_checking")
class CycleCheckingGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with cycle_checking.
    """
    def __init__(self, task='cycle_checking'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format("Yes" if label else "No")




