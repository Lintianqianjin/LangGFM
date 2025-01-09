from ._base_generator import StructuralTaskGraphGenerator


@StructuralTaskGraphGenerator.register("node_attribute_retrieval")
class NodeAttributeRetrievalGraphGenerator(StructuralTaskGraphGenerator):
    """
    A generator for graphs task with node_attribute_retrieval.
    """
    def __init__(self, task='node_attribute_retrieval'):
        super().__init__(task)

    def _generate_answer(self, label, query_entity):
        return self.config['answer_format'].format(*[*query_entity,label])

