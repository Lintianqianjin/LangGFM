import torch
import pandas as pd
import networkx as nx

from .utils.ogb_dataset import CustomPygNodePropPredDataset

from ._base_generator import NodeTaskGraphGenerator

@NodeTaskGraphGenerator.register("ogbn_arxiv")
class OgbnArxivGraphGenerator(NodeTaskGraphGenerator):
    """
    OgbnArxivGraphGenerator: A generator for creating k-hop subgraphs 
    from the OGBN-Arxiv dataset using NetworkX format.
    """

    def load_data(self):
        """
        Load the OGBN-Arxiv dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data' # run script in LangGFM/
        dataset = CustomPygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
        
        self.graph = dataset[0]
        # print(f"{self.graph=}")
        # self.graph = RemoveSelfLoops()(self.graph)
        # print(f"{self.graph=} after remove self loops")
        # transform = RemoveDuplicatedEdges()
        # self.graph = transform(self.graph)
        # print(f"{self.graph=} after remove duplicated edges")
        # to_undirected = ToUndirected()
        # self.graph = to_undirected(self.graph)
        # print(f"{self.graph=} after to undirected")
        
        # Get split indices
        # split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # self.all_idx = torch.cat([train_idx, valid_idx, test_idx], dim=0)
        self.all_samples = set(range(self.graph.num_nodes))
        
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
            "Discrete Mathematics"
        ]


    def get_query(self, target_node_idx:int) -> str:
        """
        Get the query for the main task based on the sample ID.
        """
        return f"Please infer the subject area of the paper with node id {target_node_idx}. The available areas are: {self.labelidx2arxivcategeory}. "
    
    
    def get_answer(self, sample_id, target_node_idx:int) -> str:
        """
        Get the label of a node based on the sample ID.
        """
        return f"The paper with node id {target_node_idx} likely belongs to the subject area '{self.labelidx2arxivcategeory[self.graph.y[sample_id][0].item()]}'."
    
    
    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask) -> nx.Graph:
        """
        Create a NetworkX graph from the sampled subgraph.
        """
        G = nx.MultiDiGraph()
        
        # Add nodes with attributes
        for raw_node_idx, new_node_idx in node_mapping.items():
            paper_year = self.graph.node_year[raw_node_idx][0].item()
            paper_title = self.node_id_to_title_mapping[raw_node_idx]
            G.add_node(new_node_idx, title=paper_title, year=paper_year)
        
        # Add edges
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            G.add_edge(src, dst)
            
        return G
    