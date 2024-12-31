import networkx as nx
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import k_hop_subgraph
import pandas as pd
import random


class OgbnArxivGraphGenerator(InputGraphGenerator):
    """
    OgbnArxivGraphGenerator: A generator for creating k-hop subgraphs 
    from the OGBN-Arxiv dataset using NetworkX format.
    """

    def __init__(self, num_hops=2):
        self.num_hops = num_hops
        self.dataset = None
        self.graph = None
        self.all_idx = None
        self.paper_title_mapping = {}
        self.node_id_mapping = None
        self.labels = []

    def load_data(self):
        """
        Load the OGBN-Arxiv dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = '../../../../../data'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
        self.graph = dataset[0]
        
        # Get split indices
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.all_idx = torch.cat([train_idx, valid_idx, test_idx], dim=0)
        
        # Load title/abstract mappings
        self.paper_mag_id_title_mapping = {}
        with open(f'{self.root}/ogbn_arxiv/raw/titleabs.tsv') as fw:
        
            paper_id = 200971
            _, title, _ = fw.readline().split('\t')
            self.paper_mag_id_title_mapping[paper_id] = title
            
            for line_idx, line in enumerate(fw):
                try:
                    paper_id, title, _ = line.split('\t')
                    self.paper_mag_id_title_mapping[int(paper_id)] = title
                except:
                    continue

        # Load node-to-paper ID mapping
        self.paper_node_id_mag_id_mapping = pd.read_csv(f'{self.root}/ogbn_arxiv/mapping/nodeidx2paperid.csv',index_col=0)
        

        # Define category labels
        self.labelidx2arxivcategeory = [
            "Numerical Analysis",
            "Multimedia",
            "Logic in Computer Science",
            "Computers and Society",
            "Cryptography and Security",
            "Distributed, Parallel, and Cluster Computing",
            "Human-Computer Interaction",
            "Computational Engineering, Finance, and Science",
            "Networking and Internet Architecture",
            "Computational Complexity",
            "Artificial Intelligence",
            "Multiagent Systems",
            "General Literature",
            "Neural and Evolutionary Computing",
            "Symbolic Computation",
            "Hardware Architecture",
            "Computer Vision and Pattern Recognition",
            "Graphics",
            "Emerging Technologies",
            "Systems and Control",
            "Computational Geometry",
            "Other Computer Science",
            "Programming Languages",
            "Software Engineering",
            "Machine Learning",
            "Sound",
            "Social and Information Networks",
            "Robotics",
            "Information Theory",
            "Performance",
            "Computation and Language",
            "Information Retrieval",
            "Mathematical Software",
            "Formal Languages and Automata Theory",
            "Data Structures and Algorithms",
            "Operating Systems",
            "Computer Science and Game Theory",
            "Databases",
            "Digital Libraries",
            "Discrete Mathematics",
        ]

    def generate_graph(self, sample_id: int, num_hops: int = 2) -> tuple[nx.Graph, dict]:
        """
        Generate a k-hop subgraph for the given sample ID in NetworkX format.
        """
        if self.graph is None or self.all_idx is None:
            raise ValueError("Dataset not loaded. Call `load_data` first.")

        # Generate k-hop subgraph
        # src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
        #     node_idx=sample_id, num_hops=self.num_hops, edge_index=self.graph.edge_index, 
        #     relabel_nodes=False, flow='source_to_target', directed=False
        # )
        # tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
        #     node_idx=sample_id, num_hops=self.num_hops, edge_index=self.graph.edge_index, 
        #     relabel_nodes=False, flow='target_to_source', directed=False
        # )
        src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
            node_idx = sample_id, num_hops=num_hops, edge_index=self.graph.edge_index, 
            relabel_nodes=False, flow='source_to_target',directed=False
        )
        tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
            node_idx = sample_id, num_hops=num_hops, edge_index=self.graph.edge_index, 
            relabel_nodes=False, flow='target_to_source',directed=False
        )
        
        # Combine edges and nodes
        sub_graph_edge_index = self.graph.edge_index.T[torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)].T
        sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())
        
        # Create NetworkX graph
        G = nx.MultiDiGraph()
        node_mapping = {raw_node_idx: new_node_idx for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)}
        label = self.labels[self.graph.y[sample_id][0].item()]
        
        # Add nodes with attributes
        for raw_node_idx, new_node_idx in node_mapping.items():
            paper_year = self.graph.node_year[raw_node_idx][0].item()
            paper_title = self.paper_title_mapping[
                self.node_id_mapping.at[raw_node_idx, 'paper id']
            ]
            G.add_node(new_node_idx, title=paper_title, year=paper_year)
        
        # Add edges
        for edge_idx in range(sub_graph_edge_index.size(1)):
            src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
            dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
            G.add_edge(src, dst)
        
        # Metadata
        metadata = {
            "query": f"Please infer the subject area of the query paper (node ID {node_mapping[sample_id]}).",
            "label": label,
            "target_node": node_mapping[sample_id]
        }

        return G, metadata
