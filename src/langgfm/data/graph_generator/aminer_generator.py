import json
import networkx as nx
import pandas as pd
from torch_geometric.datasets import AMiner

from .utils.sampling import generate_node_centric_k_hop_subgraph
from .utils.shuffle_graph import shuffle_nodes_randomly

from .base_generator import InputGraphGenerator


@InputGraphGenerator.register("aminer")
class AMinerGraphGenerator(InputGraphGenerator):
    """
    AMinerGraphGenerator: A generator for creating k-hop subgraphs 
    from the AMiner dataset using NetworkX format.
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
        Load the AMiner dataset and preprocess required mappings.
        """
        # Load dataset and graph
        self.root = './data'
        
        dataset = AMiner(root=f'{self.root}/AMiner/')
        graph = dataset[0]

        author_labelled_index = graph['author']['y_index']
        author_labels =  graph['author']['y']

        del graph['author']['y_index']
        del graph['author']['y']
        del graph['venue']['y_index']
        del graph['venue']['y']
        del graph[('paper', 'written_by', 'author')]
        del graph[('venue', 'publishes', 'paper')]

        area_mapping = {
            3: 'Computational Linguistics', 
            4: 'Computer Graphics', 
            2: 'Computer Networks & Wireless Communication', 
            7: 'Computer Vision & Pattern Recognition', 
            1: 'Computing Systems', 
            8: 'Databases & Information Systems',
            5: 'Human Computer Interaction',
            6: 'Theoretical Computer Science'
        }

        graph = graph.to_homogeneous()
        # print(graph)

        node_type_mapping = {0:'author', 1:'venue', 2:'paper'}
        edge_type_mapping = {0:'writes', 1:'published_in'}

        papers = pd.read_csv('./GraphData/PYG/AMiner/raw/paper.txt',sep='\t',names=['idx','title'], index_col=0)
        venues = pd.read_csv('./GraphData/PYG/AMiner/raw/id_conf.txt',sep='\t',names=['raw_idx','name'])
        authors = pd.read_csv('./GraphData/PYG/AMiner/raw/id_author.txt',sep='\t',names=['raw_idx','name'])