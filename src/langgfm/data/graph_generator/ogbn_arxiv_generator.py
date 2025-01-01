
import torch
import pandas as pd
import networkx as nx

from torch_geometric.utils import k_hop_subgraph

from .utils import CustomPygNodePropPredDataset
from .utils.sampling import k_hop_sampled_subgraph
from .utils.shuffle_graph import shuffle_nodes_randomly

from .base_generator import InputGraphGenerator

@InputGraphGenerator.register("ogbn_arxiv")
class OgbnArxivGraphGenerator(InputGraphGenerator):
    """
    OgbnArxivGraphGenerator: A generator for creating k-hop subgraphs 
    from the OGBN-Arxiv dataset using NetworkX format.
    """

    def __init__(self, num_hops=2, sampling=False, neighbor_size:int=None, random_seed:int=None):
        self.num_hops = num_hops
        self.sampling = sampling
        self.neighbor_size = neighbor_size
        self.random_seed = random_seed
        if self.sampling:
            assert neighbor_size is not None, "neighbor_size should be specified"
            assert random_seed is not None, "random_seed should be specified"

        self.graph = None
        self.all_idx = None
        self.load_data()

    def load_data(self):
        """
        Load the OGBN-Arxiv dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = '/home/tlin4/projects/LangGFM/data' # run script in LangGFM/
        dataset = CustomPygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
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
            
            for line in fw:
                try:
                    paper_id, title, _ = line.split('\t')
                    self.paper_mag_id_title_mapping[int(paper_id)] = title
                except:
                    continue
       

        # Load node-to-paper ID mapping
        self.paper_node_id_mag_id_mapping = pd.read_csv(f'{self.root}/ogbn_arxiv/mapping/nodeidx2paperid.csv')
        self.paper_node_id_mag_id_mapping = dict(zip(self.paper_node_id_mag_id_mapping['node idx'].to_list(),self.paper_node_id_mag_id_mapping['paper id'].to_list()))

        # Map node_id to paper_title
        self.node_id_to_title_mapping = {
            node_id: self.paper_mag_id_title_mapping[mag_id]
            for node_id, mag_id in self.paper_node_id_mag_id_mapping.items()
            if mag_id in self.paper_mag_id_title_mapping
        }

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

    def generate_graph(self, sample_id: int, ) -> tuple[nx.Graph, dict]:
        """
        Generate a k-hop subgraph for the given sample ID in NetworkX format.
        """
        assert sample_id >= 0, "sample id should be greater than zero"
        assert sample_id < self.graph.num_nodes, f"sample id should be smaller than {self.graph.num_nodes}"

        if self.graph is None or self.all_idx is None:
            raise ValueError("Dataset not loaded. Call `load_data` first.")


        if self.sampling:
            # Generate k-hop subgraph with neighbor sampling
            src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_sampled_subgraph(
                node_idx = sample_id, num_hops=self.num_hops, edge_index=self.graph.edge_index, 
                relabel_nodes=False, flow='source_to_target',directed=False, 
                neighbor_size=self.neighbor_size, random_seed=self.random_seed
            )

            tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_sampled_subgraph(
                node_idx = sample_id, num_hops=self.num_hops, edge_index=self.graph.edge_index, 
                relabel_nodes=False, flow='target_to_source',directed=False,
                neighbor_size=self.neighbor_size, random_seed=self.random_seed
            )
        else:
            # Generate k-hop subgraph
            src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
                node_idx = sample_id, num_hops=self.num_hops, edge_index=self.graph.edge_index, 
                relabel_nodes=False, flow='source_to_target',directed=False
            )

            tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
                node_idx = sample_id, num_hops=self.num_hops, edge_index=self.graph.edge_index, 
                relabel_nodes=False, flow='target_to_source',directed=False
            )
        
        # Combine edges and nodes
        sub_graph_edge_index = self.graph.edge_index.T[torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)].T
        sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())
        
        # Create NetworkX graph
        G = nx.MultiDiGraph()
        node_mapping = {raw_node_idx: new_node_idx for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)}
        # print(f"{node_mapping=}")
        label = self.labelidx2arxivcategeory[self.graph.y[sample_id][0].item()]
        
        # Add nodes with attributes
        for raw_node_idx, new_node_idx in node_mapping.items():
            paper_year = self.graph.node_year[raw_node_idx][0].item()
            # print(f"{raw_node_idx=}")
            paper_title = self.node_id_to_title_mapping[raw_node_idx]
            G.add_node(new_node_idx, title=paper_title, year=paper_year)
        
        # Add edges
        for edge_idx in range(sub_graph_edge_index.size(1)):
            src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
            dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
            G.add_edge(src, dst)
            
        new_G, node_idx_mapping_old_to_new = shuffle_nodes_randomly(G)
        G = new_G
        target_node_idx = node_idx_mapping_old_to_new[node_mapping[sample_id]]
        
        # Metadata
        metadata = {
            "raw_sample_id": sample_id,
            "num_hop": self.num_hops,
            "sampling": {
                "enable": self.sampling,
                "neighbor_size": self.neighbor_size,
                "random_seed": self.random_seed
            },
            "main_task": {
                "query": f"Please infer the subject area of the paper with node id {target_node_idx}. The available areas are: {self.labelidx2arxivcategeory}. ",
                "label": label,
                "target_node": target_node_idx
            }
        }

        return G, metadata
