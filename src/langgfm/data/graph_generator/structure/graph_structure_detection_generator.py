from .._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("graph_structure_detection")
class GraphStructureDetectionGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with graph_structure_detection.
    """
    def __init__(self, task='graph_structure_detection'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity=None):
        return self.config['answer_format'].format(label)





