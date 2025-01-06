import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

import random
import networkx as nx
import torch
import pandas as pd
from tqdm import tqdm

from .base_generator import EdgeGraphGenerator
from .utils.hetero_graph_utils import get_node_slices
from langgfm.utils.io import load_jsonl

@EdgeGraphGenerator.register("yelp_review")
class YelpReviewGraphGenerator(EdgeGraphGenerator):
    """
    YelpReviewGraphGenerator: A generator for creating k-hop subgraphs 
    from the Yelp dataset using NetworkX format.
    """

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
        print(f"{self.raw_reviews_info.shape=}")
        
        # Corresponding one-to-one with user nodes in sequence
        self.raw_users_info = load_jsonl(f"{self.root}/yelp_academic_dataset_user.json", return_type="dataframe")
        print(f"{self.raw_users_info.shape=}")
        
        # Corresponding one-to-one with business nodes in sequence
        self.raw_businesses_info = load_jsonl(f"{self.root}/yelp_academic_dataset_business.json", return_type="dataframe")
        print(f"{self.raw_businesses_info.shape=}")
        
        # Corresponding one-to-one with (user, tip, business) edges in sequence
        self.raw_tips_info = load_jsonl(f"{self.root}/yelp_academic_dataset_tip.json", return_type="dataframe")
        print(f"{self.raw_tips_info.shape=}")
        # 其中包含如下类型：
        #   - 节点: user, business
        #   - 边: (user, 'friend', user), (user, 'review', business), (user, 'tip', business)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            self.hetero_data = torch.load(f"{self.root}/yelp.pt")
        print(f"{self.hetero_data=}")
        self.node_slices = get_node_slices(self.hetero_data.num_nodes_dict)
    
        # 将异质图转为同质图
        self.graph = self.hetero_data.to_homogeneous()
        print(f"{self.graph=}")
        self.node_type_mapping = {0: 'user', 1: 'business'}
        self.edge_type_mapping = {0:'friend', 1:'review', 2:'tip'}
        
        # save all the edges in (user, review, business) edges in self.hetero_data
        self.all_samples = [
            (user_id.item(), business_id.item()) 
            for user_id, business_id in self.hetero_data.edge_index('user', 'review', 'business').T
        ]


    def get_query(self, target_src_node_idx: int, target_dst_node_idx: int):
        """
        生成一个提示/问题，询问 user 对 business 的评论内容。
        """
        query = (f"It is known that user with node id {target_src_node_idx} commented on "
                 f"business with node id {target_dst_node_idx}. Please generate a review text "
                 f"by mimicking the user {target_src_node_idx}'s style.")
        return query


    def get_answer(self, edge: tuple, target_src_node_idx: int, target_dst_node_idx: int):
        """
        给定一个 (user_node_idx_in_homo_pyg_graph, business_node_idx_in_homo_pyg_graph) 边，
        在原始数据中找到对应的评论文本，作为 ground truth。
        """
        # 比如在 self.reviews 中找到对应的评论
        # 需要结合 raw user / raw business id 的映射进行查找，这里仅示意
        # edge 传进来的是 (src_user_nid, dst_business_nid) 在同质图中的索引
        # 通常需要去除 offset 后，再去 self.reviews 里匹配
        
        # 注意：edge[0] - self.node_slices['user'][0] 如果 users 在 homogeneous graph 的 offset 是 self.node_slices['user'][0] 
        # 这里示例认为 user/biz 刚好是 0 ~ #user - 1, #user ~ #user+#biz - 1
        
        user_homo_idx, business_homo_idx = edge
        # 
        
        # 这里仅简单示意直接从 self.reviews 中找到一个文本
        # 在实际情况下，你需要通过 user_homo_idx => user_raw_id => user_id => 再和 df_reviews merge
        # 同理 business
        # 下面假设可以直接拿到 star, text ...
        # 仅供参考
        # 假设 user_homo_idx, biz_homo_idx 对应 df_reviews 的一行
        rating = 4
        text = "This place is absolutely amazing!"
        
        answer = (f"User (node id {target_src_node_idx}) gave business (node id {target_dst_node_idx}) "
                  f"a {rating}-star review and said: \"{text}\"")
        return answer


    def create_networkx_graph(self, sub_graph_edge_index, node_mapping: dict, **kwargs):
        """
        将子图 (通过 edge_index 选出来的节点和边) 构建为一个 NetworkX 的 MultiDiGraph，
        并将相关的节点/边属性加入。
        """
        G = nx.MultiDiGraph()
        
        # 添加节点 (为每个节点设定属性)
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_type = self.node_type_mapping[self.graph.node_type[raw_node_idx].item()]
            
            if node_type == 'user':
                # 如果在同质图中，raw_node_idx - self.node_slices['user'][0] 是用户在 DataFrame 里的行
                offset_user_idx = raw_node_idx - self.node_slices['user'][0]
                
                # 取一些示例属性
                df_row = self.users.iloc[offset_user_idx]  # 这里仅示意
                feats = {
                    "type": "user",
                    "review_count": df_row["review_count"],
                    "average_stars": df_row["average_stars"],
                    "fans": df_row["fans"],
                    "send_useful_votes": df_row["useful"],
                    "send_funny_votes": df_row["funny"],
                    "send_cool_votes": df_row["cool"]
                }
                G.add_node(new_node_idx, **feats)
            
            elif node_type == 'business':
                offset_business_idx = raw_node_idx - self.node_slices['business'][0]
                df_row = self.businesses.iloc[offset_business_idx]
                feats = {
                    "type": "business",
                    "categories": df_row["categories"],
                    "stars": df_row["stars"],
                    "city": df_row["city"],
                    "review_count": df_row["review_count"]
                }
                G.add_node(new_node_idx, **feats)
            else:
                raise ValueError(f"Unknown node type: {node_type}")
        
        # 添加边
        for raw_src, raw_dst in sub_graph_edge_index.T:
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            edge_type_id = self.graph.edge_type[raw_src, raw_dst].item()  # 仅示意
            etype = self.edge_type_mapping[edge_type_id]
            
            if etype == 'friend':
                G.add_edge(src, dst, type="Friend")
            elif etype == 'review':
                # 在实际的 Yelp 数据里，需要去找到具体的 review 内容 (stars, text, date, votes...) 
                # 这里简单示意
                feats = {
                    "type": "Review",
                    "stars": 4,
                    "date": "2020-01-01",
                    "text": "A placeholder review text",
                    "receive_useful_votes": 1,
                    "receive_funny_votes": 0,
                    "receive_cool_votes": 1
                }
                G.add_edge(src, dst, **feats)
            elif etype == 'tip':
                feats = {
                    "type": "Tip",
                    "date": "2022-05-10",
                    "text": "Try their new dish!",
                    "compliment_count": 3
                }
                G.add_edge(src, dst, **feats)
            else:
                raise ValueError(f"Unknown edge type: {etype}")
        
        return G


    def generate_graph(self, sample: tuple, edge_index=None):
        """
        生成针对某一个 (user_homo_idx, business_homo_idx) 的子图，以及对应的 query, answer。
        与 Movielens1MGraphGenerator 中类似，通过对目标样本的时间戳或其他逻辑筛选出相应子图。
        这里示例仅展示简单用法。
        """
        user_id, business_id = sample
        
        # 1) 在这里，你可以像 Movielens1M 一样，根据该条边的时间戳或其他属性做截断
        #    或者像老版本 YelpReviewGeneration 一样，做 k-hop 邻居采样
        #    这里为了演示，我们不做时间筛选，而是简单直接把“已有的 homogeneous graph”中
        #    与这条边相关的 2-hop 邻居拿出来
        # --------------------------
        # 示例：只做 k-hop subgraph
        # (你需要自己实现或者引入类似 torch_geometric.utils.k_hop_subgraph 的函数)
        
        # 如果你跟 movielens 保持风格一致，也可以参考:
        #   timestamp = self.graph.time[target_edge_mask]
        #   edge_mask = self.graph.time < timestamp
        #   ...
        
        # 这里仅演示: 假设做 2-hop
        # from torch_geometric.utils import k_hop_subgraph
        # subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
        #     node_idx=[user_id, business_id],
        #     num_hops=2,
        #     edge_index=self.graph.edge_index,
        #     relabel_nodes=False
        # )
        #
        # 为了不使示例过长，这里直接把整图都拿来
        sub_edge_index = self.graph.edge_index
        # 做一个最简单的 node_mapping
        node_mapping = {nid: i for i, nid in enumerate(torch.unique(sub_edge_index))}

        # 2) 用 create_networkx_graph 构建 NXG
        G = self.create_networkx_graph(sub_edge_index, node_mapping)

        # 3) 构造 query, answer
        query = self.get_query(user_id, business_id)
        answer = self.get_answer((user_id, business_id), user_id, business_id)
        
        # 4) 返回 (graphs, query, answer, meta_info)
        # 可以与 Movielens1MGraphGenerator 一样返回
        # 注意，base_generator 通常会在外部遍历 self.all_samples，然后分别调用 generate_graph
        # 因此你也可以在外部一次性生成
        return [G], query, answer, f"User({user_id})-Business({business_id})"
