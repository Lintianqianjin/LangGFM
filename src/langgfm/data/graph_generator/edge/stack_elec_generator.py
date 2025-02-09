import re
import os
import sys
import json
import networkx as nx
import torch
import pandas as pd

from torch_geometric.data import Data
from collections import Counter

# from ..utils.graph_utils import get_edge_idx_in_graph, represent_edges_with_multiplex_id
from langgfm.data.graph_generator.utils.graph_utils import get_edge_idx_in_graph, represent_edges_with_multiplex_id
from langgfm.data.graph_generator._base_generator import EdgeTaskGraphGenerator


@EdgeTaskGraphGenerator.register("stack_elec")
class StackElecGraphGenerator(EdgeTaskGraphGenerator):
    """
    StackElecGraphGenerator: A generator for creating k-hop subgraphs from the StackExchange dataset using NetworkX format.
    """
    
    directed = True
    has_node_attr = True
    has_edge_attr = True
    
    def load_data(self):
        self.root = "data/stack_elec"
        
        self.entity_text = pd.read_csv(f"{self.root}/entity_text.csv", index_col=0)
        self.edge_list = pd.read_csv(f"{self.root}/edge_list.csv", index_col=0)
        # logger.debug(self.edge_list.loc[self.edge_list['i'] < 67155])
        self.relation_text = pd.read_csv(f"{self.root}/relation_text.csv", index_col=0)
        
        edge_index = torch.tensor(self.edge_list[['u','i']].values, dtype=torch.long).t().contiguous()
        relation_idx = torch.tensor(self.edge_list['r'].values, dtype=torch.long)
        edge_label = torch.tensor(self.edge_list['label'].values, dtype=torch.long)
        
        # create pyg data object
        if not os.path.exists(f"{self.root}/data.pt"):
            data = Data(edge_index=edge_index, edge_relation_idx=relation_idx, edge_label=edge_label)
            torch.save(data, f"{self.root}/data.pt")
        else:
            data = torch.load(f"{self.root}/data.pt", weights_only=False)
            
        self.graph = data
        resutls = represent_edges_with_multiplex_id(self.graph.edge_index, range(self.graph.edge_index.size(1)))
        # logger.debug(f"{resutls=}")
        self.all_samples = list(resutls)
        # self.all_samples = set([(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))])
    
    @property
    def graph_description(self):
        desc = ("This is a graph constructed from question-and-answer data related to electronic techniques of the Stack Exchange platform, "
        "forming a dynamic bipartite structure that captures multi-round interactions between users and questions. "
        "Nodes are users and questions. User nodes represent individuals who ask, answer, or comment on questions, "
        "while question nodes correspond to technical inquiries on the platform. "
        "Edges represent interactions between users and questions, including answers and comments.")

        return desc
    
    def get_query(self, target_src_node_idx, target_dst_node_idx):
        query = (f"Each edge, i.e., answer or comment, is classified as 'Useful' if its voting count is greater than 1, and 'Useless' otherwise."
        f"Please infer the category of the edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx}.")
        return query
    
    def get_answer(self, edge, target_src_node_idx, target_dst_node_idx):
        """
        Get the answer for a specific edge.
        Args:
            edge (tuple): The node indices of the src and dst node.
        """
        src, dst, multiplex_id = edge
        edge_idx = get_edge_idx_in_graph(src, dst, self.graph.edge_index, multiplex_id=multiplex_id)
        label = self.graph.edge_label[edge_idx].item()
        
        if label == 1:
            answer = (f"The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} "
            "is likely to be classified as 'Useful'.")
            
        else: 
            answer = (f"The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} "
            "is likely to be classified as 'Useless'.")
        
        return answer

    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None, edge=None, **kwargs):
        """
        Create a NetworkX graph from the sampled subgraph.
        """
        
        G = nx.MultiDiGraph()
        
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_text = self.entity_text.at[raw_node_idx,'text']
            node_type, features = self.__parse_node_text(node_text)
            G.add_node(new_node_idx, type=node_type, **features)
            

        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            # get relation text
            relation_idx = self.graph.edge_relation_idx[edge_idx].item()
            relation_text = self.relation_text.at[relation_idx, 'text']
            
            G.add_edge(src, dst, type="reply", content=relation_text)
            
        
        return G
        
    def __parse_node_text(self, node_text):
    
        """
        Parse the input text to extract node type (node_type) and corresponding feature information (features).
        :param text: str, input text content
        :return: tuple, (node_type, features)
        """
        user_pattern = re.compile(r"Name:\s*(.*?)\s*Location:\s*(.*?)\s*Introduction:\s*(.*)", re.DOTALL)
        question_pattern = re.compile(r"Title:\s*(.*?)\s*Post:\s*(.*)", re.DOTALL)
        
        user_match = user_pattern.match(node_text)
        
        if user_match:
            # remove "Name": user_match.group(1), 
            return "user", {"Location": user_match.group(2), "Introduction": user_match.group(3)}
        
        question_match = question_pattern.match(node_text)
        if question_match:
            return "question", {"Title": question_match.group(1), "Post": question_match.group(2)}
        
        return None, {}  # Return None and an empty dictionary if no pattern is matched


if __name__ == "__main__":
    edges = [[637, 106735, 4], [28712, 166303, 0], [55, 84573, 1], [2676, 189542, 2], [35, 105888, 0], [6086, 196230, 3], [9019, 95491, 0], [67, 98494, 1], [713, 166102, 6], [103, 74190, 0], [44244, 295169, 0], [50, 292290, 0], [27, 103409, 0], [73, 217694, 0], [97, 217247, 0], [19789, 268909, 1], [725, 323176, 0], [54, 162992, 0], [433, 133286, 1], [109, 227863, 2], [3668, 228154, 0], [3505, 273039, 0], [69, 163250, 0], [36, 148316, 0], [14142, 117392, 0], [6121, 241882, 0], [617, 372169, 0], [23498, 152364, 1], [117, 75168, 3], [31304, 205661, 1], [5160, 336109, 0], [63, 136940, 1], [1703, 322547, 1], [54, 292575, 0], [23003, 229272, 0], [89, 186946, 1], [551, 70755, 3], [193, 99860, 0], [15801, 124735, 1], [7987, 94182, 2], [818, 180805, 1], [3017, 371735, 0], [15313, 99502, 2], [12, 225445, 0], [260, 76070, 0], [2280, 326917, 0], [1666, 76480, 1], [72, 96557, 1], [3, 195621, 0], [3152, 97567, 0], [514, 228030, 0], [12506, 166837, 5], [698, 155763, 0], [38046, 243888, 0], [39371, 149395, 2], [765, 227916, 1], [53563, 330190, 0], [426, 140689, 2], [11708, 106834, 0], [27, 106656, 0], [27, 114048, 0], [145, 157887, 2], [1085, 110379, 2], [150, 142704, 0], [7350, 89017, 2], [846, 198681, 0], [71, 208497, 0], [3728, 80980, 0], [22385, 129622, 0], [20171, 146907, 3], [452, 180955, 2], [52784, 256884, 1], [1875, 115960, 1], [1226, 145971, 1], [50, 362559, 0], [217, 205706, 3], [217, 122101, 0], [24342, 187524, 0], [17, 121448, 2], [3793, 205085, 0], [107, 367565, 0], [1116, 208277, 2], [815, 391091, 0], [140, 93045, 0], [53334, 387171, 0], [567, 202774, 0], [19534, 379946, 0], [567, 134093, 1], [72, 89492, 2], [57041, 395112, 0], [239, 266563, 0], [35, 140118, 0], [145, 155716, 0], [18099, 118221, 0], [601, 107953, 1], [763, 235101, 0], [103, 276057, 0], [1177, 217900, 1], [433, 195914, 1], [190, 263544, 0], [217, 262840, 0], [19208, 80166, 0], [835, 160169, 0], [1070, 267988, 1], [145, 179642, 0], [62, 212012, 1], [97, 116168, 1], [885, 166132, 0], [1479, 158629, 1], [17, 242278, 0], [518, 157826, 0], [7957, 218333, 0], [36, 219619, 1], [169, 274484, 1], [746, 108391, 0], [54, 173387, 2], [5128, 146629, 0], [107, 177946, 1], [206, 148615, 2], [41892, 265876, 0], [9943, 99264, 9], [855, 84296, 2], [12, 113094, 0], [30, 108974, 0], [128, 119266, 1], [30197, 199641, 1], [33, 320974, 0], [1522, 197528, 0], [1000, 236522, 0], [27755, 190936, 0], [51, 121647, 0], [962, 105317, 0], [703, 118933, 0], [824, 181659, 0], [19195, 124988, 0], [26365, 76013, 0], [55, 144441, 1], [2493, 248084, 0], [52286, 160685, 1], [35, 365406, 0], [1999, 88702, 1], [4280, 111394, 0], [54, 395149, 0], [187, 359477, 0], [158, 124904, 0], [32468, 150682, 0], [846, 284464, 0], [55, 323167, 0], [53, 131182, 3], [34181, 221988, 0], [29178, 175737, 0], [32498, 224708, 1], [128, 267753, 0], [1568, 158564, 0], [107, 208868, 1], [6478, 394213, 0], [55, 126685, 0], [2871, 75087, 0], [6745, 248242, 1], [35, 243335, 0], [50, 122965, 1], [23180, 162140, 3], [27, 160251, 1], [440, 123849, 0], [239, 129625, 0], [14830, 173261, 0], [10722, 167597, 4], [61739, 129101, 0], [57690, 128935, 0], [35, 178232, 0], [924, 333983, 0], [1030, 220919, 0], [140, 122022, 2], [8468, 113344, 1], [3863, 349428, 0], [18564, 138560, 3], [79, 85981, 0], [2010, 276345, 0], [1343, 203692, 0], [2820, 258189, 2], [52263, 116849, 0], [151, 283882, 0], [72, 290858, 0], [189, 92751, 0], [11183, 318465, 0], [24204, 240911, 0], [14510, 185374, 0], [2050, 183453, 0], [72, 71735, 7], [140, 223556, 0], [72, 76882, 0], [34416, 223369, 0], [2707, 287364, 0], [7281, 109610, 3], [63, 122558, 1], [76, 72307, 0], [27, 163700, 2], [650, 93043, 4], [79, 210229, 1], [1709, 76811, 0]]
    generator = StackElecGraphGenerator()
    generator.load_data()
    edge_index = []
    for edge in edges:
        src, dst, multiplex_id = edge
        edge_idx = get_edge_idx_in_graph(src, dst, generator.graph.edge_index, multiplex_id=multiplex_id)
        edge_index.append(edge_idx)
    with open("edge_index.json", "w") as f:
        json.dump(edge_index, f)