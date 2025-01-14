from .._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("edge_attribute_retrieval")
class EdgeAttributeRetrievalGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with edge_attribute_retrieval.
    """
    def __init__(self, task='edge_attribute_retrieval'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity):
        return self.config['answer_format'].format(*[*query_entity,label])

