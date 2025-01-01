

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

