import re
import os
import json
import networkx as nx
import pandas as pd


from .._base_generator import GraphTaskGraphGenerator


@GraphTaskGraphGenerator.register("explagraphs")
class ExplagraphsGraphGenerator(GraphTaskGraphGenerator):
    """
    ExplagraphsGraphGenerator: A generator for creating graphs 
    from the Explagraphs dataset using NetworkX format.
    """
    
    directed = True
    has_node_attr = True
    has_edge_attr = True
    
    def load_data(self):
        """
        Load the Explagraphs dataset and preprocess required mappings.
        """
        self.root = './data/explagraph'
        self.samples = pd.read_csv(f'{self.root}/explagraph_train_dev.tsv', sep='\t')
        
        self.all_samples = set(self.samples.index.tolist())
    
    @property
    def graph_description(self):
        desc = "This is a graph constructed from commonsense logic. Nodes represent commonsense objects and edges represent the relation between two objects."
        return desc
    
    def get_query(self, sample_id):
        arg1 = self.samples.at[sample_id, 'arg1']
        arg2 = self.samples.at[sample_id, 'arg2']
        query = (f"Here are two arguments: 1. {arg1} 2. {arg2} "
                "Based on the given commonsense logic graph, do these two arguments support each other or counter each other?")
        
        return query
    
    def get_answer(self, sample_id):
        label = self.samples.at[sample_id, 'label']
        answer = f"These two arguments {label} each other."
        
        return answer

    def create_networkx_graph(self, sample_id):
        """
        Create a NetworkX graph from the sampled subgraph.
        """
        graph_text = self.samples.at[sample_id, 'graph']
        
        
        G = self.__text_to_multidigraph(graph_text)
    
        return G

        
        
    def __text_to_multidigraph(self, text: str) -> nx.MultiDiGraph:
        """
        convert graph text like sequence of (node1; relation; node2) to MultiDiGraph。
        
        params:
            text (str): sequence of (node1; relation; node2) triplets
            
        return:
            nx.MultiDiGraph
        """
        G = nx.MultiDiGraph()
        
        # find all matches like ( ... )
        triplets = re.findall(r"\(([^)]+)\)", text)
        
        name_to_index = {}
        node_counter = 0
        # parse each triplet
        for triplet in triplets:
            # get [source, relation, target]
            parts = [p.strip() for p in triplet.split(";")]
            if len(parts) == 3:
                source, relation, target = parts
                # add nodes 
                if source not in G.nodes():
                    name_to_index[source] = node_counter
                    G.add_node(node_counter, type="common_sense_concept", name=source)
                    node_counter += 1
                    
                if target not in G.nodes():
                    name_to_index[target] = node_counter
                    G.add_node(node_counter, type="common_sense_concept", name=target)
                    node_counter += 1
                    
                # add edge
                G.add_edge(name_to_index[source], name_to_index[target], type = "relation", description=relation)
        
        return G