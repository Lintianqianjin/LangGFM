import json
import networkx as nx
import pandas as pd
from torch_geometric.datasets import AMiner


from ._base_generator import NodeTaskGraphGenerator


@NodeTaskGraphGenerator.register("aminer")
class AMinerGraphGenerator(NodeTaskGraphGenerator):
    """
    AMinerGraphGenerator: A generator for creating k-hop subgraphs 
    from the AMiner dataset using NetworkX format.
    """

    def load_data(self):
        """
        Load the AMiner dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        
        dataset = AMiner(root=f'{self.root}/AMiner/')
        self.graph = dataset[0]
        # print(f"{self.graph=}")
        # number of authors is 1693531
        # number of labelled authors is 246678
        self.author_labelled_index = self.graph['author']['y_index']
        self.all_samples = set(self.author_labelled_index.tolist())
        # print(f"{self.all_sample_ids=}")
        # print(f"{self.author_labelled_index=}")
        # author_labels is a tensor of size 246678, corresponding to the author_labelled_index
        self.author_labels = self.graph['author']['y']
        
        # create a dict that maps the author_labelled_index to the author_labels
        self.labelled_author_index_to_label = {}
        for i in range(self.author_labelled_index.size(0)):
            self.labelled_author_index_to_label[self.author_labelled_index[i].item()] = self.author_labels[i].item()
        
        del self.graph['author']['y_index']
        del self.graph['author']['y']
        del self.graph['venue']['y_index']
        del self.graph['venue']['y']
        del self.graph[('paper', 'written_by', 'author')]
        del self.graph[('venue', 'publishes', 'paper')]

        self.area_mapping = {
            3: 'Computational Linguistics', 
            4: 'Computer Graphics', 
            2: 'Computer Networks & Wireless Communication', 
            7: 'Computer Vision & Pattern Recognition', 
            1: 'Computing Systems', 
            8: 'Databases & Information Systems',
            5: 'Human Computer Interaction',
            6: 'Theoretical Computer Science'
        }

        self.graph = self.graph.to_homogeneous()

        self.node_type_mapping = {0:'author', 1:'venue', 2:'paper'}
        self.edge_type_mapping = {0:'writes', 1:'published_in'}

        self.papers = pd.read_csv(f'{self.root}/AMiner/raw/paper.txt',sep='\t',names=['idx','title'], index_col=0)
        self.venues = pd.read_csv(f'{self.root}/AMiner/raw/id_conf.txt',sep='\t',names=['raw_idx','name'])
        self.authors = pd.read_csv(f'{self.root}/AMiner/raw/id_author.txt',sep='\t',names=['raw_idx','name'])
    
    def get_query(self, target_node_idx):
        query = (f"Please infer the research area of the author with node id of {target_node_idx}. " 
            f"The available research areas are: {list(self.area_mapping.values())}.")
        return query
    
    def get_answer(self, sample_id, target_node_idx):
        # area_mapping starts from 1
        author_label = self.area_mapping[self.labelled_author_index_to_label[sample_id]+1] # 
        answer = f"The research area of the author with node id of {target_node_idx} is '{author_label}'."
        return answer
    
    def create_networkx_graph(self, sub_graph_edge_index, node_mapping, sub_graph_edge_mask):
        
        G = nx.MultiDiGraph()
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_type = self.node_type_mapping[self.graph.node_type[raw_node_idx].item()]
            if node_type == 'paper':
                paper_idx = raw_node_idx - 1693531 - 3883
                try:
                    G.add_node(new_node_idx, type = 'paper', title=self.papers.at[paper_idx,'title'])
                except KeyError:
                    G.add_node(new_node_idx, type = 'paper')
            elif node_type == 'venue':
                venue_idx = raw_node_idx - 1693531
                G.add_node(new_node_idx, type = 'venue', name = self.venues.at[venue_idx,'name'][1:])
        
        sub_graph_edge_type = self.graph.edge_type[sub_graph_edge_mask]
        for edge_idx in range(sub_graph_edge_index.size(1)):
            src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
            dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
            edge_type = self.edge_type_mapping[sub_graph_edge_type[edge_idx].item()]
            G.add_edge(src, dst, type = edge_type)
    
        return G