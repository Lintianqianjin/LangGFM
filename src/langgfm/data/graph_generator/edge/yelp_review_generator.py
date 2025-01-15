import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

import random
import networkx as nx
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from .._base_generator import EdgeTaskGraphGenerator
from ..utils.graph_utils import get_node_slices, get_edge_idx_in_graph, get_edge_idx_in_etype
from ....utils.io import load_jsonl

@EdgeTaskGraphGenerator.register("yelp_review")
class YelpReviewGraphGenerator(EdgeTaskGraphGenerator):
    """
    YelpReviewGraphGenerator: A generator for creating k-hop subgraphs 
    from the Yelp dataset using NetworkX format.
    """
    
    directed = True

    def load_data(self):
        """
        1) 加载 Yelp 数据 (HeteroData 或处理后的对象)；
        2) 将异质图转换为同质图；
        3) 准备各种映射（如用户/商家在原始数据与同质图中的映射）；
        4) 整理出所有可能的 (user, business) 样本对，即 self.all_samples；
        """
        self.root = "./data/Yelp"
        # Corresponding one-to-one with (user, review, business) edges in sequence
        self.raw_reviews_info = load_jsonl(f"{self.root}/yelp_academic_dataset_review.json", return_type="dataframe")
        self.raw_reviews_info["date"] = pd.to_datetime(self.raw_reviews_info["date"], errors='coerce')
        # print(f"{self.raw_reviews_info.shape=}")
        
        # Corresponding one-to-one with user nodes in sequence
        self.raw_users_info = load_jsonl(f"{self.root}/yelp_academic_dataset_user.json", return_type="dataframe")
        self.raw_users_info["yelping_since"] = pd.to_datetime(self.raw_users_info["yelping_since"], errors='coerce')
        # print(f"{self.raw_users_info.shape=}")
        
        # Corresponding one-to-one with business nodes in sequence
        self.raw_businesses_info = load_jsonl(f"{self.root}/yelp_academic_dataset_business.json", return_type="dataframe")
        # print(f"{self.raw_businesses_info.shape=}")
        
        # Corresponding one-to-one with (user, tip, business) edges in sequence
        self.raw_tips_info = load_jsonl(f"{self.root}/yelp_academic_dataset_tip.json", return_type="dataframe")
        self.raw_tips_info["date"] = pd.to_datetime(self.raw_tips_info["date"], errors='coerce')
        # print(f"{self.raw_tips_info.shape=}")

        #   - Nodes: user, business
        #   - Edges: (user, 'friend', user), (user, 'review', business), (user, 'tip', business)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            self.hetero_data = torch.load(f"{self.root}/yelp.pt")
        # print(f"{self.hetero_data=}")
        self.node_slices = get_node_slices(self.hetero_data.num_nodes_dict)
    
        # 将异质图转为同质图
        self.graph = self.hetero_data.to_homogeneous()
        # print(f"{self.graph=}")
        self.node_type_mapping = {0: 'user', 1: 'business'}
        self.edge_type_mapping = {0:'friend', 1:'review', 2:'tip'}
        
        # save all the edges in (user, review, business) edges in self.hetero_data
        edge_count = defaultdict(int)  # Dictionary to store the number of times each (src, dst) pair appears
        edge_dict = defaultdict(int)  # Dictionary to store the mapping from (src, dst, multiplex_id) to edge idx in the edge_index
        # Iterate through all edges
        target_edge_type_index = self.hetero_data['user', 'review', 'business'].edge_index
        for idx, (src, dst) in enumerate(zip(target_edge_type_index[0], target_edge_type_index[1])):
            key = (int(src), int(dst))
            multiplex_id = edge_count[key]  # Get the current multiplex_id for this pair
            edge_dict[(int(src), int(dst), multiplex_id)] = idx  # Store the edge index
            edge_count[key] += 1   # Increment the count for this (src, dst) pair
            
        self.all_samples = list(edge_dict.keys())

    @property
    def graph_description(self):
        """
        Get the description of the graph.
        """
        desc = "This graph is a heterogeneous graph about reviews on the Yelp platform. "\
            "Each node is either a business or a user. Each edge represents that "\
            "two users are friends, or a user reviewed a business, or a user left a tip for a business."
        return desc
    
    def get_query(self, target_src_node_idx: int, target_dst_node_idx: int):
        """
        生成一个提示/问题，询问 user 对 business 的评论内容。
        """
        query = (f"User with node id {target_src_node_idx} is going to leave a review for business with node id {target_dst_node_idx}. "
        f"Please generate the review text that mimics the writing style of the user.")
        return query


    def get_answer(self, edge: tuple, target_src_node_idx: int, target_dst_node_idx: int):
        """
        给定一个 (user_node_idx_in_homo_pyg_graph, business_node_idx_in_homo_pyg_graph) 边，
        在原始数据中找到对应的评论文本，作为 ground truth。
        """
        user_id, business_id, multiplex_id = edge
        # find the order/index of this edge in self.graph.
        edge_idx = get_edge_idx_in_graph(src=user_id, dst=business_id,edge_index=self.graph.edge_index,multiplex_id=multiplex_id)
        review_edge_idx, edge_type = get_edge_idx_in_etype(edge_idx=edge_idx, edge_types=self.graph.edge_type, return_etype=True)
        
        if edge_type != 1:  # Ensure it's a review edge
            raise ValueError(f"Edge ({user_id}, {business_id}) is not a review edge.")
        
        # get raw review info
        review_info = self.raw_reviews_info.iloc[review_edge_idx]
        # rating = review_info['stars']
        review = review_info['text']
        answer = (
            f"User with node id {target_src_node_idx} may leave a review for Business with node id {target_dst_node_idx} as follows: "
            f"{review}"
        )
        return answer


    def create_networkx_graph(self, node_mapping: dict, edge, sub_graph_edge_mask, **kwargs):
        """
            Create a NetworkX graph from the subgraph edge index and node mapping.
        """
        G = nx.MultiDiGraph()
        # target_edge time
        user_id, business_id, multiplex_id = edge
        edge_idx = get_edge_idx_in_graph(src=user_id, dst=business_id,edge_index=self.graph.edge_index,multiplex_id=multiplex_id)
        target_edge_idx, edge_type = get_edge_idx_in_etype(edge_idx=edge_idx, edge_types=self.graph.edge_type, return_etype=True)
        # get raw review info
        review_info = self.raw_reviews_info.iloc[target_edge_idx]
        # get date
        target_edge_date = review_info['date']
        
        
        # adding nodes with attributes
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_type = self.node_type_mapping[self.graph.node_type[raw_node_idx].item()]
            
            if node_type == 'user':
                user_idx = raw_node_idx - self.node_slices['user'][0]
                
                # get user info
                df_row = self.raw_users_info.iloc[user_idx]
                feats = {
                    "type": "user",
                    "yelping_since": df_row["yelping_since"].strftime("%Y-%m"),
                    "review_count": df_row["review_count"],
                    "average_stars": df_row["average_stars"],
                    "fans": df_row["fans"],
                    "send_useful_votes": df_row["useful"],
                    "send_funny_votes": df_row["funny"],
                    "send_cool_votes": df_row["cool"],
                    "elite_year": df_row["elite"],
                    "compliment_hot": df_row["compliment_hot"],
                    "compliment_more": df_row["compliment_more"],
                    "compliment_profile": df_row["compliment_profile"],
                    "compliment_cute": df_row["compliment_cute"],
                    "compliment_list": df_row["compliment_list"],
                    "compliment_note": df_row["compliment_note"],
                    "compliment_plain": df_row["compliment_plain"],
                    "compliment_cool": df_row["compliment_cool"],
                    "compliment_funny": df_row["compliment_funny"],
                    "compliment_writer": df_row["compliment_writer"],
                    "compliment_photos": df_row["compliment_photos"]
                }
                G.add_node(new_node_idx, **feats)
            
            elif node_type == 'business':
                offset_business_idx = raw_node_idx - self.node_slices['business'][0]
                df_row = self.raw_businesses_info.iloc[offset_business_idx]
                feats = {
                    "type": "business",
                    "categories": df_row["categories"],
                    "stars": df_row["stars"],
                    "location": f"{df_row['city'], df_row['state']}",
                    "review_count": df_row["review_count"]
                }
                G.add_node(new_node_idx, **feats)
            else:
                raise ValueError(f"Unknown node type: {node_type}")
        
        # adding edges with attributes
        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            # edge_idx in self.graph (homo_graph of the hetero_data)
            edge_idx_within_etype, edge_type_id = get_edge_idx_in_etype(edge_idx=edge_idx, edge_types=self.graph.edge_type, return_etype=True)
            etype_name = self.edge_type_mapping[edge_type_id]
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            
            if etype_name == 'friend':
                feats = {
                    "type": "Friend",
                }
                
            elif etype_name == 'review':
                # find edge info in raw_reviews_info
                review_info = self.raw_reviews_info.iloc[edge_idx_within_etype]
                feats = {
                    "type": "Review",
                    "stars": review_info['stars'],
                    "date": review_info['date'].strftime("%Y-%m-%d"),
                    "text": review_info['text'],
                    "receive_useful_votes": review_info['useful'],
                    "receive_funny_votes": review_info['funny'],
                    "receive_cool_votes": review_info['cool']
                }
                # compare the date of the edge with the target edge
                # if later than the target edge, remove the edge
                if review_info['date'] >= target_edge_date:
                    continue
                
            elif etype_name == 'tip':
                tip_info = self.raw_tips_info.iloc[edge_idx_within_etype]
                feats = {
                    "type": "Tip",
                    "date": tip_info['date'].strftime("%Y-%m-%d"),
                    "text": tip_info['text'],
                    "compliment_count": tip_info['compliment_count']
                }
                # compare the date of the edge with the target edge
                # if later than the target edge, remove the edge
                if tip_info['date'] >= target_edge_date:
                    continue
                
            else:
                raise ValueError(f"Unknown edge type: {etype_name}")

            G.add_edge(src, dst, **feats)
        
        return G

    def generate_graph(self, sample, edge_index=None):
        '''
        Generate a k-hop subgraph for the given sample.
        '''
        # sample id (user id and business id in pyg heterogeneous graph object)
        # convert it into ids in homogeneous graph object
        user_id, business_id, multiplex_id = sample
        business_id += self.node_slices['business'][0]
        sample = (user_id, business_id, multiplex_id)
        
        return super().generate_graph(sample, edge_index)

    @property
    def YelpReviewGeneration(self):
        return "This is a heterogeneous graph about reviews on the Yelp platform. "\
            "Each node is either a business or a user. Each edge represents that "\
            "two users are friends, or a user reviewed a business, or a user left a tip for a business."
