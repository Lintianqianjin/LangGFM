import json
import networkx as nx

from torch_geometric.datasets import WikiCS
from ._base_generator import NodeTaskGraphGenerator


@NodeTaskGraphGenerator.register("wikics")
class WikicsGraphGenerator(NodeTaskGraphGenerator):
    """
    WikicsGraphGenerator: A generator for creating k-hop subgraphs 
    from the WikiCS dataset using NetworkX format.
    """

    def load_data(self):
        """
        Load the WikiCS dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        wiki_dataset = WikiCS(root=f"{self.root}/WikiCS",is_undirected=False)
        self.graph = wiki_dataset[0]
        
        self.all_samples = set(range(self.graph.num_nodes))
        
        with open(f"{self.root}/WikiCS/raw/metadata.json") as json_file:
            self.wiki_meta = json.load(json_file)
        
        self.labels = [
            "Computational linguistics",
            "Databases",
            "Operating systems",
            "Computer architecture",
            "Computer security", 
            "Internet protocols", 
            "Computer file systems", 
            "Distributed computing architecture", 
            "Web technology", 
            "Programming language topics"
        ]
        
    def get_query(self, sample_id: int) -> str:
        """
        Get the query for a specific sample node.
        Args:
            sample_id (int): The index of the sample node.
        Returns:
            str: The query for the sample node.
        """
        return f"The webpages are classified into 10 categories representing branches of computer science. Please determine the category of the webpage with node id {sample_id}. The available categories are: {self.labels}. "
    
    def get_answer(self, sample_id, target_node_idx: int) -> str:
        """
        Get the answer for a specific sample node.
        Args:
            sample_id (int): The index of the sample node.
        Returns:
            str: The answer for the sample node.
        """
        label = self.wiki_meta['nodes'][sample_id]['label']
        return f"The webpage with node id {target_node_idx} likely belongs to the category '{label}'."
    
    def create_networkx_graph(self, node_mapping: dict, sub_graph_edge_mask) -> nx.Graph:
        """
        Create a NetworkX graph from the sampled subgraph.
        Args:
            sub_graph_edge_index: The edge index of the subgraph.
            node_mapping: The mapping of raw node indices to new node indices.
        Returns:
            nx.Graph: The NetworkX graph.
        """
        G = nx.MultiDiGraph()
        for raw_node_idx, new_node_idx in node_mapping.items():
            G.add_node(new_node_idx, title = self.wiki_meta['nodes'][raw_node_idx]['title'])
            
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            G.add_edge(src, dst)
        return G
    