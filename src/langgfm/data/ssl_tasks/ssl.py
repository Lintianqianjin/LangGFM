import networkx as nx
import json
import copy
import random
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class AuxiliaryTasks:

    def __init__(self):
        self.idx_token_mapping = {0:'first', 1: 'second', 2: 'third'}
        pass

    def check_number_of_graphs(self, nxg):
        more_than_one_graph = False
        choose_nxg_idx = None
        if isinstance(nxg, list): # for tasks like graph similarity, two graph will be provided
            more_than_one_graph = True
            # random choose one graph
            choose_nxg_idx = random.choice(list(range(len(nxg))))
            nxg = nxg[choose_nxg_idx]

        return more_than_one_graph, nxg, choose_nxg_idx
    
    def get_node_features(self, G):
        node_features = {}
        for node, attrs in G.nodes(data=True):
            if attrs:  # 检查是否有属性
                node_features[node] = attrs
        return node_features

    def get_edge_features(self, G):
        # if isinstance(nx.classes.multidigraph.MultiDiGraph, G):

        # print(f"{type(G)=}, {G.is_multigraph()=}")
        # print(f"{G.edges(data=True)=}")
        edge_features = {}
        if G.is_multigraph():
            for u, v, k, attrs in G.edges(data=True, keys=True):
                # print(f"{u=}, {v=}, {attrs=}")
                if attrs:  # 检查是否有属性
                    edge_features[(u, v, k)] = attrs
        else:
            for u, v, attrs in G.edges(data=True):
                # print(f"{u=}, {v=}, {attrs=}")
                if attrs:  # 检查是否有属性
                    edge_features[(u, v)] = attrs

        return edge_features


    def GraphMAE2_data_preparation(self, nxg, mask_node_ratio=0.2, mask_edge_ratio=0.2):
        
        G = copy.deepcopy(nxg)
        # print(f"{type(G)}=")
        # print(G.nodes(data=True))
        # print(G.edges(data=True))
        # exit()
        node_features = self.get_node_features(G) # empty when no node has any feature
        edge_features = self.get_edge_features(G) # empty when no edge has any feature
        # print(f"{node_features=}")
        # print(f"{edge_features=}")
        # exit()

        masked_nodes_features = None
        # 处理节点特征
        if node_features:
            nodes = list(node_features.keys())
            num_nodes_to_mask = int(len(nodes) * mask_node_ratio)
            # print(f"{num_nodes_to_mask=}")
            masked_nodes = random.sample(nodes, num_nodes_to_mask)

            masked_nodes_features = {}
            
            for node in masked_nodes:
                masked_nodes_features[node] = copy.deepcopy(node_features[node])
                # print(f"{G.nodes[node]=}")
                # G.nodes[node].clear()  # 删除原始特征
                for key in G.nodes[node]:
                    G.nodes[node][key] = "Unknown"  # 将每个特征的值MASK为 "Unknown" 
                # print(f"{G.nodes[node]=}")
                # exit()

        masked_edges_features = None
        # 处理边特征
        if edge_features:
            edges = list(edge_features.keys())
            num_edges_to_mask = int(len(edges) * mask_edge_ratio)
            # print(f"{num_edges_to_mask=}")
            masked_edges = random.sample(edges, num_edges_to_mask)

            masked_edges_features = {}

            # for edge in edges:
            for edge in masked_edges:
                masked_edges_features[edge] = copy.deepcopy(edge_features[edge])
                # print(type(G))
                # G.edges[edge].clear()  # 删除原始特征
                for key in G.edges[edge]:
                    G.edges[edge][key] = "Unknown"  # 将每个特征的值MASK为 "Unknown" 
                # print(G.edges[edge])

        return G, masked_nodes_features, masked_edges_features
    

    
    def connectivity_detection(self, nxg):
        '''
            # random choose two nodes
            # return if they are connected

            return: the two nodes and connectivity
        '''
        # print(nxg)
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)
        if nxg.number_of_edges == 0:
            return None, None

        if np.random.rand(1)[0] > 0.5:
            # print('edge', random.choice(list(nxg.edges)))
            # key in MultiDiGraph
            # https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.add_edge.html#networkx.MultiDiGraph.add_edge
            nxg_edges = list(nxg.edges)
            if nxg_edges:
                node_a, node_b, key = random.choice(nxg_edges) 
                # print(node_a, node_b, key)
                result = nxg.has_edge(node_a, node_b)
                assert result==True, 'error in real edge selection' 
            else:
                return None, None
        else:
            node_a = random.choice(list(nxg.nodes))
            # print(list(nxg.nodes))
            # print(set(nxg.neighbors(node_a)))
            # print(node_a)
            non_connected_neighbors = list(set(nxg.nodes) - set(nxg.neighbors(node_a)))
            if non_connected_neighbors:
                node_b = random.choice(non_connected_neighbors)
                result = nxg.has_edge(node_a, node_b)
                assert result==False, 'error in fake edge selection' 
            else:
                return None, None

        if more_than_one_graph:
            
            query = f'Are node {node_a} and node {node_b} in the {self.idx_token_mapping[choose_nxg_idx]} graph directly connected?'
            if result:
                answer = f"Yes, node {node_a} and node {node_b} in the {self.idx_token_mapping[choose_nxg_idx]} graph are directly connected."
            else:
                answer = f"No, node {node_a} and node {node_b} in the {self.idx_token_mapping[choose_nxg_idx]} graph are not directly connected."
        else:
            query = f'Are node {node_a} and node {node_b} directly connected?'
            if result:
                answer = f"Yes, node {node_a} and node {node_b} are directly connected."
            else:
                answer = f"No, node {node_a} and node {node_b} are not directly connected."
            
        return query, answer

    #  # this task may generate too many paths, and this task is similar to shortest_path_detection
    # def simple_path_detection(self, nxg, cutoff=5):
    #     # random choose two nodes
    #     node_a, node_b = random.choices(list(nxg.nodes), k=2)
    #     query = f"Please generate all simple paths (with a length <= {cutoff}) in the graph from node {node_a} to node {node_b} in a list format "\
    #             f"if there is at least one simple path between the two nodes. " \
    #             f"A simple path is a path with no repeated nodes."\
                
    #     paths = []
    #     for path in nx.all_simple_paths(nxg, source=node_a, target=node_b, cutoff=cutoff):
    #         paths.append(path)

    #     if len(paths) == 1: 
    #         answer = f"There is only one simple path from node {node_a} to node {node_b}. It is {json.dumps(paths[0], cls=NpEncoder)}."
    #     elif len(paths) > 1: 
    #         answer = f"All the simple paths from node {node_a} to node {node_b} are as follows: {json.dumps(paths, cls=NpEncoder)}."
    #     else:
    #         answer = f"There is no simple path between the two nodes."
        
    #     return query, answer

    
    def shortest_path_detection(self, nxg):
        
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)
        # random choose two nodes
        node_a, node_b = random.choices(list(nxg.nodes), k=2)

        paths = []
        try:
            for path in nx.all_shortest_paths(nxg, source=node_a, target=node_b):
                paths.append(path)
        except nx.exception.NetworkXNoPath:
            paths = None

        if more_than_one_graph:
            query = f"Please generate all shortest paths in the {self.idx_token_mapping[choose_nxg_idx]} graph from node {node_a} to node {node_b} in a list format "\
                "if there is at least one shortest path between the two nodes." 

            if paths is not None:
                # print(paths)
                if len(paths) == 1: 
                    answer = f"There is only one shortest path from node {node_a} to node {node_b} in the {self.idx_token_mapping[choose_nxg_idx]} graph, i.e., {json.dumps(paths[0], cls=NpEncoder)}."
                elif len(paths) > 1: 
                    answer = f"All the shortest paths from node {node_a} to node {node_b} in the {self.idx_token_mapping[choose_nxg_idx]} graph are as follows: {json.dumps(paths, cls=NpEncoder)}."
            else:
                answer = f"There is no path between the two nodes."
        else:
            # print('shortest_path_detection', node_a, node_b)
            query = f"Please generate all shortest paths in the graph from node {node_a} to node {node_b} in a list format "\
                    f"if there is at least one shortest path between the two nodes."
            if paths is not None:
                if len(paths) == 1: 
                    answer = f"There is only one shortest path from node {node_a} to node {node_b}, i.e., {json.dumps(paths[0], cls=NpEncoder)}."
                elif len(paths) > 1: 
                    answer = f"All the shortest paths from node {node_a} to node {node_b} are as follows: {json.dumps(paths, cls=NpEncoder)}."
                # else:
                #     answer = f"There is no path between the two nodes."
            else:
                answer = f"There is no path between the two nodes."
        return query, answer

    
    
    def neighbor_retrieval(self, nxg):

        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)

        node_a = random.choice(list(nxg.nodes))

        neighbors = list(set(nx.all_neighbors(nxg, node_a)))
        # print('neighbors',neighbors)

        if more_than_one_graph:
            
            query = f'Plese list the neighbors of node {node_a} in the {self.idx_token_mapping[choose_nxg_idx]} graph in ascending order of their node IDs.'
            if len(neighbors)==1:
                answer = f"Node {node_a} has only one neighbor in the {self.idx_token_mapping[choose_nxg_idx]} graph, i.e., node {neighbors[0]}."
            elif len(neighbors)>1:
                answer = f"The neighbors of node {node_a} in the {self.idx_token_mapping[choose_nxg_idx]} graph is {json.dumps(neighbors, cls=NpEncoder)}."
            else:
                answer = f"Node {node_a} in the {self.idx_token_mapping[choose_nxg_idx]} graph is an isolated node and it has no neighbor."
        else:
            query = f'Plese list the neighbors of node {node_a} in ascending order in a list.'
            if len(neighbors)==1:
                answer = f"Node {node_a} has only one neighbor, i.e., node {neighbors[0]}."
            elif len(neighbors)>1:
                # answer = f"The neighbors of node {node_a} is {json.dumps(neighbors, cls=NpEncoder)}."
                answer = f"The neighbors of node {node_a} is {sorted([int(item) for item in neighbors])}."
            else:
                answer = f"Node {node_a} is an isolated node and it has no neighbor."        

        return query, answer

    
    def degree_computing(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)
        node_a = random.choice(list(nxg.nodes))
        degree = nxg.degree(node_a)

        if more_than_one_graph:

            query = f"What is the degree of node {node_a} in the {self.idx_token_mapping[choose_nxg_idx]} graph?"
            answer = f"The degree of node {node_a} in the {self.idx_token_mapping[choose_nxg_idx]} graph is {degree}."
        else:
            query = f"What is the degree of node {node_a}?"
            answer = f"The degree of node {node_a} is {degree}."

        return query, answer

    
    def simple_cycle_detection(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)

        detected_cycles = list(nx.simple_cycles(nxg, length_bound=3)) # , orientation="ignore"
        # detected_chordless_cycles = list(nx.chordless_cycles(nxg)) # defaul including edges (due to two directed edges)
        detected_cycles = list(filter(lambda x: len(x)>2, detected_cycles))

        if more_than_one_graph:
            query = f"A cycle is a closed path where no node appears twice. "\
                    f"Are there any cycles in the {self.idx_token_mapping[choose_nxg_idx]} graph that exactly contain three nodes? "\
                    f"If there are any cycles that meet the criteria, please return the lists of nodes along the cycles. It the number of the cycles exceeds three, you can only return three cycles."

            if len(detected_cycles)==1:
                answer = f"There is only one cycle that exactly contain three nodes in the {self.idx_token_mapping[choose_nxg_idx]} graph, i.e., {json.dumps(detected_cycles[0],cls=NpEncoder)}."
            elif len(detected_cycles)>1 and len(detected_cycles)<=3:
                answer = f"The detected cycles that exactly contain three nodes in the {self.idx_token_mapping[choose_nxg_idx]} graph are as follows: {json.dumps(detected_cycles,cls=NpEncoder)}."
            elif len(detected_cycles)>3:
                detected_cycles = random.choices(detected_cycles,k=3)
                answer = f"The number of the detected cycles that exactly contain three nodes exceeds three in the {self.idx_token_mapping[choose_nxg_idx]} graph. These are three examples: {json.dumps(detected_cycles,cls=NpEncoder)}."
            else:
                answer = f"The the {self.idx_token_mapping[choose_nxg_idx]} graph contains no cycles that exactly contain three nodes."
        else:
            query = f"A cycle is a closed path where no node appears twice. "\
                    f"Are there any cycles in the graph that exactly contain three nodes? "\
                    f"If there are any cycles that meet the criteria, please return the lists of nodes along the cycles."

            if len(detected_cycles)==1:
                answer = f"There is only one cycle that exactly contain three nodes, i.e., {json.dumps(detected_cycles[0],cls=NpEncoder)}."
            elif len(detected_cycles)>1 and len(detected_cycles)<=3:
                answer = f"The detected cycles that exactly contain three nodes are as follows: {json.dumps(detected_cycles,cls=NpEncoder)}."
            elif len(detected_cycles)>3:
                detected_cycles = random.choices(detected_cycles,k=3)
                answer = f"The number of the detected cycles that exactly contain three nodes exceeds three. These are three examples: {json.dumps(detected_cycles,cls=NpEncoder)}."
            else:
                answer = f"The graph contains no cycles that exactly contain three nodes."

        return query, answer

    
    def diameter_computing(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)

        error_message = None
        try: 
            diameter = nx.diameter(nxg)
        except nx.exception.NetworkXError as error:
            # Handling the exception and capturing the error message
            error_message = str(error)
            error_message = error_message.replace('Found infinite path length ', '')
            # Found infinite path length because the digraph is not strongly connected
            # Found infinite path length because the graph is not connected
        

        if more_than_one_graph:
            query = f"The diameter is the maximum eccentricity and the eccentricity of a node is the maximum distance from it to all other nodes in a graph. "\
                    f"Please return the diameter of the {self.idx_token_mapping[choose_nxg_idx]} graph. If the graph is directed, please convert it to undirected before computing."
            
            if error_message is not None:
                answer = f"The diameter of the {self.idx_token_mapping[choose_nxg_idx]} graph is infinite {error_message}."
            else:
                answer = f"The diameter of the {self.idx_token_mapping[choose_nxg_idx]} graph is {diameter}."
        else:
            query = f"The diameter is the maximum eccentricity and the eccentricity of a node is the maximum distance from it to all other nodes in a graph. "\
                    f"Please return the diameter of the graph. If the graph is directed, please convert it to undirected before computing."
            if error_message is not None:
                answer = f"The diameter of the graph is infinite {error_message}."
            else:
                answer = f"The diameter of the graph is {diameter}."

        return query, answer



    def edge_size_computing(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)
        number_of_edges = nxg.number_of_edges()

        if more_than_one_graph:
            query = f"How many edges are there in the {self.idx_token_mapping[choose_nxg_idx]} graph?"
            answer = f"There are {number_of_edges} edges in the {self.idx_token_mapping[choose_nxg_idx]} graph."
        else:
            query = f"How many edges are there in the graph?"
            answer = f"There are {number_of_edges} edges in the graph."
        
        return query, answer

    
    def node_size_computing(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)
        number_of_nodes = nxg.number_of_nodes()
        if more_than_one_graph:
            query = f"How many nodes are there in the {self.idx_token_mapping[choose_nxg_idx]} graph?"
            answer = f"There are {number_of_nodes} nodes in the {self.idx_token_mapping[choose_nxg_idx]} graph."
        else:
            query = f"How many nodes are there in the graph?"
            answer = f"There are {number_of_nodes} nodes in the graph."
        return query, answer

    
    def node_attribute_retrieval(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)

        query = None
        answer = None

        nodes = list(nxg.nodes(data=True))
        query_node = random.choice(nodes)
        # print(query_node)
        node_idx, attrs = query_node
        if attrs:
            attr, value = random.choice(list(attrs.items()))
            if more_than_one_graph:
                query = f"What is the value of attribute `{attr}` for node {node_idx} in the {self.idx_token_mapping[choose_nxg_idx]} graph?"
                answer = f"The value of attribute `{attr}` for node {node_idx} in the {self.idx_token_mapping[choose_nxg_idx]} graph is {value}."
            else:
                query = f"What is the value of attribute `{attr}` for node {node_idx}?"
                answer = f"The value of attribute `{attr}` for node {node_idx} is {value}."
        return query, answer


    def node_attribute_mae(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)

        query = None
        answer = None

        nodes = list(nxg.nodes(data=True))
        query_node = random.choice(nodes)
        # print(query_node)
        node_idx, attrs = query_node
        if attrs:
            attr, value = random.choice(list(attrs.items()))
            if more_than_one_graph:
                query = f"What is the value of attribute `{attr}` for node {node_idx} in the {self.idx_token_mapping[choose_nxg_idx]} graph?"
                answer = f"The value of attribute `{attr}` for node {node_idx} in the {self.idx_token_mapping[choose_nxg_idx]} graph is {value}."
            else:
                query = f"What is the value of attribute `{attr}` for node {node_idx}?"
                answer = f"The value of attribute `{attr}` for node {node_idx} is {value}."
        return query, answer

    
    def edge_attribute_retrieval(self, nxg):
        more_than_one_graph, nxg, choose_nxg_idx = self.check_number_of_graphs(nxg)

        query = None
        answer = None

        edges = list(nxg.edges(data=True))
        if len(edges) == 0:
            return None, None
        query_edge = random.choice(edges)
        src, dst, attrs = query_edge
        if attrs:
            attr, value = random.choice(list(attrs.items()))
            if more_than_one_graph:
                query = f"What is the value of attribute {attr} for the edge from node {src} to node {dst} in the {self.idx_token_mapping[choose_nxg_idx]} graph?"
                answer = f"The value of attribute {attr} for the edge from node {src} to node {dst} in the {self.idx_token_mapping[choose_nxg_idx]} graph is {value}."
            else:
                if nxg.is_directed():
                    query = f"What is the value of attribute `{attr}` for the edge from node {src} to node {dst}?"
                    answer = f"The value of attribute `{attr}` for the edge from node {src} to node {dst} is {value}."
                else:
                    if np.random.rand(1)[0] > 0.5:
                        query = f"What is the value of attribute `{attr}` for the edge between node {src} and node {dst}?"
                        answer = f"The value of attribute `{attr}` for the edge between node {src} and node {dst} is {value}."
                    else:
                        query = f"What is the value of attribute `{attr}` for the edge between node {dst} and node {src}?"
                        answer = f"The value of attribute `{attr}` for the edge between node {dst} and node {src} is {value}."
        return query, answer

    # 
    # def clustering_coefficient_computing(self, nxg):
    #     query = f"Compute the clustering coefficient for nodes."
    #     pass
    
def generate_graphmae_ssl_instructions(filename, mask_node_ratio=0.1, mask_edge_ratio=0.1, max_samples = 1000):
    dataset_name, graph_format, split, max_length = extract_variables(filename)
    print(f"{dataset_name=}, {graph_format=}, {split=}, {max_length=}")
    # exit()
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/91c1b4bce6/models/LLMs/Meta-Llama-3-8B-Instruct")
    # for split in ['train','val','test']:
        # ugm@OgbnArxivForAblation@@@main_max_length=8000_GML_samples_val.json
    main_instructions = json.load(open(f"./GraphData/SamplesV5/OFFICIAL/{filename}"))
    ssl_instructions = []
    for sample in tqdm(main_instructions, desc = f"{dataset_name}, {graph_format}, {split}, {max_length}" ):
        graph_text = sample["graph_text"][0]
        nxg = load_graph_from_text(graph_text, graph_format, directed=False, multigraph=True)
        masked_G, masked_nodes_features, masked_edges_features = TASKS_SFT.GraphMAE2_data_preparation(nxg, mask_node_ratio=mask_node_ratio, mask_edge_ratio=mask_edge_ratio)

        # print(f"{masked_nodes_features, masked_edges_features=}")
        
        if graph_format == 'Table':
            g_texts = generate_table(masked_G, undirected=is_graph_directed_config[dataset_name])
        
        elif graph_format == 'GraphML':
            graphml_string = '\n'.join(nx.generate_graphml(masked_G))
            g_texts = graphml_string
        
        elif graph_format == 'GML':
            gml_string = '\n'.join(nx.generate_gml(masked_G))
            g_texts = gml_string
        
        elif graph_format == 'JSON':
            json_string = nx.node_link_data(masked_G)
            json_string = json.dumps(json_string, cls=MyJsonEncoder, indent=2)
            g_texts = json_string
        else:
            g_texts = None

        if masked_nodes_features is not None:
            # print(f"{masked_nodes_features=}")
            for node_id, node_attrs in masked_nodes_features.items():
                
                query = f"The attribute(s) of node {node_id} seems to be missing. Please infer the attribute value(s) and return them in dictionary form (with attribute names as keys and inferred values as values)."
                answer = f"The attribute(s) of node {node_id} should be {node_attrs}."
                # print()
                ssl_sample = copy.deepcopy(sample)
                ssl_sample['graph_text'] = [g_texts]
                ssl_sample["query"] = query
                ssl_sample["output"] = answer
                ssl_sample["task_type"] = "GraphMAE_Node"

                prompt = PROMPT_DICT["prompt_template"].format_map(ssl_sample)
                example = prompt + ssl_sample["output"]
                example = tokenizer.encode(example)
                example.append(tokenizer.eos_token_id)

                ssl_sample["#tokens"] = len(example)

                if ssl_sample["#tokens"]>max_length:
                    print(f"{ssl_sample['#tokens']=}")
                    continue
                else:
                    ssl_instructions.append(ssl_sample)

        if masked_edges_features is not None:
            for edge, edge_attrs in masked_edges_features.items():
                if masked_G.is_multigraph(): 
                    (src, dst, key) = edge
                else:
                    (src, dst) = edge

                # for attr_name, attr_value in edge_attrs.items():
                query = f"The attribute(s) of the edge between node {src} and node {dst} seems to be missing. Please infer the attribute value(s) and return them in dictionary form (with attribute names as keys and inferred values as values)."
                answer = f"The attribute(s) of the edge between node {src} and node {dst} should be {edge_attrs}."

                ssl_sample = copy.deepcopy(sample)
                ssl_sample["query"] = query
                ssl_sample["output"] = answer
                ssl_sample["task_type"] = "GraphMAE_Edge"

                prompt = PROMPT_DICT["prompt_template"].format_map(ssl_sample)
                example = prompt + ssl_sample["output"]
                example = tokenizer.encode(example)
                example.append(tokenizer.eos_token_id)

                ssl_sample["#tokens"] = len(example)

                if ssl_sample["#tokens"]>max_length:
                    print(f"{ssl_sample['#tokens']=}")
                    continue
                else:
                    ssl_instructions.append(ssl_sample)

    # 确保原始列表的长度至少为1000
    if len(ssl_instructions) < 1000:
        max_samples = len(ssl_instructions)
        # raise ValueError("原始列表的长度小于1000，无法随机抽取1000个字典。")

    # 随机抽取1000个字典
    ssl_instructions = random.sample(ssl_instructions, max_samples)

    with open(f"./GraphData/SamplesV5/OFFICIAL/{dataset_name}_GraphMAE_{max_length}_{graph_format}_{split}.json", 'w') as fw:
        json.dump(ssl_instructions, fw, indent=4)


def generate_topology_ssl_instructions(filename):
    # print(filename)
    dataset_name, graph_format, split, max_length = extract_variables(filename)
    # print(f"{dataset_name=}, {graph_format=}, {split=}, {max_length=}")
    if os.path.exists(f"./GraphData/SamplesV5/OFFICIAL/{dataset_name}_TopoSSL_{max_length}_{graph_format}_{split}.json"):
        # print(f"Already Exist ./GraphData/SamplesV5/OFFICIAL/{dataset_name}_TopoSSL_{max_length}_{graph_format}_{split}.json")
        return  
    else:
        print(f"Processing ./GraphData/SamplesV5/OFFICIAL/{dataset_name}_TopoSSL_{max_length}_{graph_format}_{split}.json")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/91c1b4bce6/models/LLMs/Meta-Llama-3-8B-Instruct")
    # for split in ['train','val','test']:
        # ugm@OgbnArxivForAblation@@@main_max_length=8000_GML_samples_val.json
    main_instructions = json.load(open(f"./GraphData/SamplesV5/OFFICIAL/{filename}"))
    ssl_instructions = []
    for sample in tqdm(main_instructions, desc = f"{dataset_name}, {graph_format}, {split}, {max_length}"):
        graph_text = sample["graph_text"][0]
        nxg = load_graph_from_text(graph_text, graph_format, directed=False, multigraph=True)
        
        query, answer = get_ssl_instruction(nxg, "neighbor_retrieval")
        
        ssl_sample = copy.deepcopy(sample)
        ssl_sample["query"] = query
        ssl_sample["output"] = answer
        ssl_sample["task_type"] = "neighbor_retrieval"

        prompt = PROMPT_DICT["prompt_template"].format_map(ssl_sample)
        example = prompt + ssl_sample["output"]
        example = tokenizer.encode(example)
        example.append(tokenizer.eos_token_id)

        ssl_sample["#tokens"] = len(example)

        if ssl_sample["#tokens"]>max_length:
            print(f"{ssl_sample['#tokens']=}")
            continue
        else:
            ssl_instructions.append(ssl_sample)

    with open(f"./GraphData/SamplesV5/OFFICIAL/{dataset_name}_TopoSSL_{max_length}_{graph_format}_{split}.json", 'w') as fw:
        json.dump(ssl_instructions, fw, indent=4)

