from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("connectivity")
class ConnectivityGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with connectivity.
    """
    def __init__(self, task='connectivity'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format("Yes" if label else "No")

