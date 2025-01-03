import json
import networkx as nx

from torch_geometric.datasets import WikiCS
from .base_generator import InputGraphGenerator, NodeGraphGenerator


@NodeGraphGenerator.register("wikics")
class WikicsGraphGenerator(NodeGraphGenerator):
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
        
        self.wiki_meta = json.load(open(f"{self.root}/WikiCS/raw/metadata.json"))
        
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
    
    def create_networkx_graph(self, sub_graph_edge_index, node_mapping: dict) -> nx.Graph:
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
        for edge_idx in range(sub_graph_edge_index.size(1)):
            src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
            dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
            G.add_edge(src, dst)
        return G
    
    # def generate_graph(self, sample_id: int, ) -> tuple[nx.Graph, dict]:
    #     """
    #     Generate a graph from the WikiCS dataset.
    #     Args:
    #         sample_id (int): The index of the sample node.
    #     Returns:
    #         tuple[nx.Graph, dict]: A tuple containing:
    #                                - nx.Graph: The generated NetworkX graph.
    #                                - dict: The metadata of the graph.
    #     """
    #     sub_graph_edge_index, sub_graph_nodes = generate_node_centric_k_hop_subgraph(self.wiki_data, sample_id, self.num_hops, self.neighbor_size, self.random_seed, self.sampling)
        

    #     G = nx.MultiDiGraph()
    #     # raw_node_idx node idx in pyg graph
    #     # new_node_idx: node idx in nxg graph
    #     node_mapping = {raw_node_idx:new_node_idx for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)}
    #     for raw_node_idx, new_node_idx in node_mapping.items():
    #         G.add_node(new_node_idx, title = self.wiki_meta['nodes'][raw_node_idx]['title'])
    #     for edge_idx in range(sub_graph_edge_index.size(1)):
    #         src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
    #         dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
    #         G.add_edge(src, dst)
        
        
    #     label = self.wiki_meta['nodes'][sample_id]['label']

    #     new_G, node_idx_mapping_old_to_new = shuffle_nodes_randomly(G)
    #     G = new_G
    #     target_node_idx = node_idx_mapping_old_to_new[node_mapping[sample_id]]

    #     # Metadata
    #     metadata = {
    #         "raw_sample_id": sample_id,
    #         "num_hop": self.num_hops,
    #         "sampling": {
    #             "enable": self.sampling,
    #             "neighbor_size": self.neighbor_size,
    #             "random_seed": self.random_seed
    #         },
    #         "main_task": {
    #             "query": f"The webpages are classified into 10 categories representing branches of computer science. Please determine the category of the webpage with node id {target_node_idx}. The available categories are: {self.labels}. ",
    #             "label": f"The webpage with node id {target_node_idx} belongs to the category {label}.",
    #             "answer": label,
    #             "target_node": target_node_idx
    #         }
    #     }

    #     return G, metadata

