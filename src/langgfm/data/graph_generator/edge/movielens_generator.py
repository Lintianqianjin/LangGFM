import json
import datetime
import networkx as nx
import torch
import pandas as pd

from ..utils.graph_utils import get_node_slices

from .._base_generator import EdgeTaskGraphGenerator
from torch_geometric.datasets import MovieLens1M

import pgeocode
from tqdm import tqdm
    
    
@EdgeTaskGraphGenerator.register("movielens1m")
class Movielens1MGraphGenerator(EdgeTaskGraphGenerator):
    """
    MovielensGraphGenerator: A generator for creating k-hop subgraphs 
    from the MovieLens dataset using NetworkX format.
    """
    
    directed = True
    has_node_attr = True
    has_edge_attr = True
    
    def load_data(self):
        """
        Load the MovieLens dataset and preprocess required mappings.
        """
        self.root = './data/MovieLens1M'
        self.dataset = MovieLens1M(root=self.root)
        # print(self.dataset[0])
        hetero_graph = self.dataset[0]
        del hetero_graph[('movie', 'rated_by', 'user')]
        self.node_slices = get_node_slices(hetero_graph.num_nodes_dict)
        # print(f"{self.node_slices=}")
        self.graph = hetero_graph.to_homogeneous()
        # print(f"{self.graph.time=}")
        # print(self.graph)
        self.node_type_mapping = {0:'movie', 1:'user'}
        # print(f"{torch.unique(self.graph.node_type,return_counts=True)}")
        # print(f"{torch.unique(self.graph.edge_type,return_counts=True)}")
        # print(f"{self.graph.time=}")

        
        self.users, self.user_raw_id_to_id_in_pyg_graph = self.__load_user_df()
        # print("\n")
        # print(self.users)
        # print("\n")
        self.movies, self.movie_raw_id_to_id_in_pyg_graph = self.__load_movie_df()
        # print("\n")
        # print(self.movies)
        # print("\n")
        self.ratings = self.__load_rating_df()
        # change UserID and MovieID in self.ratings to the corresponding node index in the graph
        self.ratings['UserID'] = self.ratings['UserID'].map(self.user_raw_id_to_id_in_pyg_graph)
        self.ratings['MovieID'] = self.ratings['MovieID'].map(self.movie_raw_id_to_id_in_pyg_graph)
        # print(f"{self.ratings=}")
        
        # samples are all the edges in the graph, i.e., all the ratings
        self.all_samples = [
            (int(self.ratings.at[i, 'UserID']), int(self.ratings.at[i, 'MovieID'])) 
            for i in range(self.ratings.shape[0])
        ]
    

    def __load_user_df(self):
        # 定义映射字典
        age_mapping = {
            1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 
            45: "45-49", 50: "50-55", 56: "56+"
        }
        
        occupation_mapping = {
            0: "unknown", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
            5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker",
            10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist",
            16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
        }
        
        gender_mapping = {"M": "Male", "F": "Female"}
        
        # load raw data
        df = pd.read_csv(
            f'{self.root}/raw/users.dat', sep='::', 
            names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
            index_col='UserID',
            encoding='ISO-8859-1',
            engine='python'
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}
        
        # replace value
        df['Gender'] = df['Gender'].map(gender_mapping)
        df['Age'] = df['Age'].map(age_mapping)
        df['Occupation'] = df['Occupation'].map(occupation_mapping)
        
        # process Zip-code to real location name
        nomi = pgeocode.Nominatim('US')

        def get_location(zip_code):
            result = nomi.query_postal_code(zip_code)
            # print(f"{result=}")
            if pd.notna(result.place_name):
                return f"{result.place_name}, {result.state_name}"
            return "Unknown"

        # print(df['Zip-code'])
        df['Location'] = df['Zip-code'].map(get_location)
        del df['Zip-code']
        
        return df, user_mapping


    def __load_movie_df(self):
        # 读取数据
        df = pd.read_csv(
            f'{self.root}/raw/movies.dat', sep='::', 
            names=['MovieID', 'Title', 'Genres'], 
            index_col='MovieID',
            engine='python', encoding='ISO-8859-1')
        
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}
        
        # 替换分隔符
        df['Genres'] = df['Genres'].str.replace('|', ', ')
        
        return df, movie_mapping

    
    def __load_rating_df(self):
        '''
        '''
        # 读取数据
        df = pd.read_csv(
            f'{self.root}/raw/ratings.dat', sep='::', 
            names=['UserID', 'MovieID', 'Rating', 'Timestamp'], 
            engine='python', encoding='ISO-8859-1'
        )
        
        # 转换时间戳为 年-月-日 时:分:秒 格式
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
        
        return df


    def get_query(self, target_src_node_idx:int, target_dst_node_idx:int):
        '''
        '''
        """
        Generate a query for rating inference.
        """
        query = (f"Each user rates a movie on a five-star scale, with only whole stars allowed. "
                 f"Please infer the rating of the user with node id {target_src_node_idx} "
                 f"for the movie with node id {target_dst_node_idx}.")
        return query
    
    
    def get_answer(self, edge:tuple, target_src_node_idx:int, target_dst_node_idx:int):
        '''
            edge: (user_node_idx_in_homo_pyg_graph, movie_node_idx_in_homo_pyg_graph)
        '''
        # find the rating score in self.ratings dataframe by edge
        rating = self.ratings[
            (self.ratings['UserID'] == edge[0]-self.node_slices['user'][0]) & 
            (self.ratings['MovieID'] == edge[1])
        ]['Rating'].values[0]
        
        answer = (f"The user with node id {target_src_node_idx} is likely to rate the movie "
          f"with node id {target_dst_node_idx} <answer> {rating} {'stars' if rating > 1 else 'star'} </answer>.")
    
        return answer
        
    
    def create_networkx_graph(self, node_mapping:dict, sub_graph_edge_mask, edge, **kwargs):
        '''
        '''
        G = nx.MultiDiGraph()
        for raw_node_idx, new_node_idx in node_mapping.items():
            # print(f"{raw_node_idx=}, {new_node_idx=}")
            node_type = 'movie' if self.graph.node_type[raw_node_idx] == 0 else 'user'
            if node_type == 'movie':
                G.add_node(
                    new_node_idx, type = 'movie', 
                    title=self.movies['Title'].iloc[raw_node_idx],  # raw_node_idx-th row
                    genres=self.movies['Genres'].iloc[raw_node_idx]
                )
            else:
                G.add_node(
                    new_node_idx, type = 'user', 
                    gender = self.users['Gender'].iloc[raw_node_idx-self.node_slices['user'][0]],
                    age = self.users['Age'].iloc[raw_node_idx-self.node_slices['user'][0]],
                    occupation = self.users['Occupation'].iloc[raw_node_idx-self.node_slices['user'][0]],
                    location = self.users['Location'].iloc[raw_node_idx-self.node_slices['user'][0]]
                )
        
        target_user_id, target_movie_id = edge
        # find this edge in self.graph.edge_index
        target_edge_mask = (self.graph.edge_index[0] == target_user_id) & (self.graph.edge_index[1] == target_movie_id)
        # find timestamp of this edge
        target_edge_timestamp = self.graph.time[target_edge_mask]
        target_edge_timestamp = pd.to_datetime(target_edge_timestamp, unit='s').strftime('%Y-%m-%d %H:%M')
        # print(f"{target_edge_timestamp=}")
        
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            edge_attrs = self.ratings[
                (self.ratings['UserID'] == raw_src-self.node_slices['user'][0]) & 
                (self.ratings['MovieID'] == raw_dst)
            ]
            # print(f"{edge_attrs['Timestamp'].item()=}, {target_edge_timestamp.item()=}")
            # print(f"{(edge_attrs['Timestamp'].item()<target_edge_timestamp.item())=}")
            if edge_attrs['Timestamp'].item()<target_edge_timestamp:
                G.add_edge(
                    src, dst, type="Rate", score = int(edge_attrs['Rating'].values[0]), 
                    time = edge_attrs['Timestamp'].values[0]
                )
            else:
                continue # do not add edges that are rated after the target edge
        
        return G
    
    
    def generate_graph(self, sample: tuple, edge_index = None):
        '''
        '''
        # sample id (user id and movie id in pyg heterogeneous graph object)
        # convert it into ids in homogeneous graph object
        user_id, movie_id = sample
        user_id += self.node_slices['user'][0] # in homogeneous graph movies is before users
        new_sample = (user_id, movie_id) # node idx in homo graph
        
        new_G, metadata = super().generate_graph(new_sample, edge_index)
        metadata["raw_sample_id"] = sample
        return new_G, metadata
        
    @property
    def graph_description(self):
        return "This graph is a heterogeneous graph where users rate movies. "\
            "Each node represents a user or a movie. "\
            "Each edge represents a rating record. "