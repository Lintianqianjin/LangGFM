import torch
import pandas as pd
import networkx as nx

from .._base_generator import NodeTaskGraphGenerator

@NodeTaskGraphGenerator.register("cora_node")
class CoraNodeGraphGenerator(NodeTaskGraphGenerator):
    '''
    '''



# def Cora(self, number_samples=2708, num_hops=2, text_mode='keywords', neighbor_size=4, sampling_seed=42):
#         # dataset = torch.load(f"./GraphData/Cora/cora_random_sbert.pt", map_location = 'cpu') 
#         cora_choice = dataset_choice_config['Cora']
#         if cora_choice == "OFA":
#             dataset = torch.load(f"../../../Baselines/OneForAll/data/single_graph/Cora/cora.pt", map_location = 'cpu')
#         elif cora_choice == "LLaGA":
#             dataset = torch.load(f"../../../Baselines/LLaGA/dataset/cora/processed_data.pt", map_location = 'cpu')
#             dataset.category_names = [dataset.label_texts[y] for y in dataset.y]
#             dataset.label_names = dataset.label_texts
#             # print(dataset.category_names[:10])
#             # os._exit(-1)
#         elif cora_choice == "InstructGLM":
#             dataset = torch.load(f"", map_location = 'cpu')
#         elif cora_choice == "GraphLLM": # tangjiliang kdd explorations
#             dataset = torch.load(f"./GraphData/Cora/cora_keywords.pt", map_location = 'cpu')

#         graph = dataset # pyg Data object
        
#         ## only title
#         text_mode = node_feature_choice_config['Cora']
#         if text_mode == 'keywords':
#             node_idx_text_mapping = {node_idx: text for node_idx, text in enumerate(graph.keywords_20)}
#         elif text_mode == 'only_title':
#             node_idx_text_mapping = {node_idx: split_title(raw_text) for node_idx, raw_text in enumerate(graph.raw_texts)}
#         elif text_mode == 'title_and_or_abstract':
#             node_idx_text_mapping = {node_idx: title for node_idx, title in enumerate(graph.raw_texts)}
#         ## done
#         node_idx_label_mapping = {node_idx: label for node_idx, label in enumerate(graph.category_names)}

#         labels = dataset.label_names
#         labels_dict = {idx: label for idx, label in enumerate(labels)}
#         labels_dict_rev = {label: idx for idx, label in enumerate(labels)}

#         # for paper_idx in random.choices(train_idx.numpy().tolist(), k=number_samples):
#         print(f'\nSampling with seed {sampling_seed}...')
#         for node_idx in list(node_idx_text_mapping.keys()):
#             src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_sampled_subgraph(
#                 node_idx = node_idx, num_hops=num_hops, edge_index=graph.edge_index, 
#                 relabel_nodes=False, flow='source_to_target', directed=False,
#                 neighbor_size=neighbor_size, random_seed=sampling_seed,
#             )
#             tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_sampled_subgraph(
#                 node_idx = node_idx, num_hops=num_hops, edge_index=graph.edge_index, 
#                 relabel_nodes=False, flow='target_to_source', directed=False,
#                 neighbor_size=neighbor_size, random_seed=sampling_seed,
#             )

#             sub_graph_edge_index = graph.edge_index.T[torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)].T
#             # print(f"{sub_graph_edge_index=}")
#             sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())
            
#             # print(f"{node_idx=}")
#             # print(sub_graph_nodes)
#             G = nx.MultiDiGraph()
#             # raw_node_idx node idx in pyg graph
#             # new_node_idx: node idx in nxg graph
#             node_mapping = {raw_node_idx: new_node_idx for new_node_idx, raw_node_idx in enumerate(sub_graph_nodes)}
            
#             target_node_idx = node_idx
#             # label = labelidx2arxivcategeory[graph.y[paper_idx][0].item()]
#             label = node_idx_label_mapping[node_idx]
#             # print(node_mapping)
#             for raw_node_idx, new_node_idx in node_mapping.items():
#                 if text_mode == 'keywords':
#                     G.add_node(new_node_idx, keywords=node_idx_text_mapping[raw_node_idx]) # first char is `v`
#                     G.add_node(new_node_idx, keywords=node_idx_text_mapping[raw_node_idx]) # first char is `v`
#                 elif text_mode == 'only_title':
#                     G.add_node(new_node_idx, title=node_idx_text_mapping[raw_node_idx]) # first char is `v`
#                     G.add_node(new_node_idx, title=node_idx_text_mapping[raw_node_idx]) # first char is `v`
#                 elif text_mode == 'title_and_or_abstract':
#                     G.add_node(new_node_idx, title_and_or_abstract=node_idx_text_mapping[raw_node_idx]) # first char is `v`
#                     G.add_node(new_node_idx, title_and_or_abstract=node_idx_text_mapping[raw_node_idx]) # first char is `v`
                
#             for edge_idx in range(sub_graph_edge_index.size(1)):
#                 src = node_mapping[sub_graph_edge_index[0][edge_idx].item()]
#                 dst = node_mapping[sub_graph_edge_index[1][edge_idx].item()]
#                 # print(f"{src=}, {dst=}")
#                 G.add_edge(src, dst)
            

#             new_G, node_idx_mapping_old_to_new = self.shuffle_nodes_randomly(G)
#             # print(node_idx_mapping_old_to_new)
#             G = new_G
#             target_node_idx = node_idx_mapping_old_to_new[node_mapping[target_node_idx]]

#             nxgs = [G]
# ###################################ONLY OPTION START#####################################
#             label_strs = '.\n'.join([f'({k}) {v.replace("_"," ")}' for k,v in labels_dict.items()])
#             query = f"""Please infer the subject area of the query paper, i.e., node with id of {target_node_idx}. 

# Here are the available subject areas:
# {label_strs}.

# Please respond with EXACTLY one of the above options. 
# """
#             answer = f"The subject area of the query paper is ({labels_dict_rev[label]})."
# ###################################ONLY OPTION END#####################################

# # ###################################COPY ANSWER START#####################################
# #             label_strs_list = '.\n'.join([f'({k}) {v}' for k,v in labels_dict.items()])
# #             query = f"""Please infer the subject area of the query paper, i.e., node with id of {target_node_idx}. 

# # Please select one response from the following options:
# # {label_strs_list}.
# # """
# #             answer = f"({labels_dict_rev[label]}) {label}."
# # ###################################COPY ANSWER END#####################################

#             yield nxgs, query, answer, f"NodeIndex({node_idx})"
