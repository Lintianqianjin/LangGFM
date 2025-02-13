import json
import torch
from tqdm import tqdm
from collections import defaultdict
from langgfm.data.graph_generator import InputGraphGenerator
from langgfm.data.graph_generator.utils.graph_utils import get_edge_idx_in_graph

datasets = [ "ogbl_vessel"] # "ogbl_vessel" "fb15k237","stack_elec"


# load edge indices
indices = defaultdict(lambda: defaultdict(list))
for dataset in datasets:
    indices[dataset]["train"] = json.load(open(f"experiments/langgfm_i/{dataset}/train_800/indices.json"))[dataset]
    indices[dataset]["test"] = json.load(open(f"experiments/langgfm_i/{dataset}/test_200/indices.json"))[dataset]

# Load graph generator
graph_generators = {}
for dataset in datasets:
    graph_generators[dataset] = InputGraphGenerator.create(dataset)
    # print(f"{graph_generators[dataset].graph=}")


new_indices = defaultdict(lambda: defaultdict(list))
# Generate baseline edge indices
for dataset in datasets:
    for split in ["train", "test"]:
        if dataset == "ogbl_vessel":
            pos_edge_indices = []
            neg_edge_indices = []
            
        edge_indices = indices[dataset][split]
        for edge in tqdm(edge_indices):
            if len(edge) == 2:
                src, dst = edge
                multiplex_id = None
            elif len(edge) == 3:
                src, dst, multiplex_id = edge
            
            if dataset == "ogbl_vessel":
                label = graph_generators[dataset].edge_label_mapping[(src,dst)]
                if label == 1:
                    pos_edge_indices.append([src, dst])
                elif label == 0:
                    neg_edge_indices.append([src, dst])
            else:
                edge_idx = get_edge_idx_in_graph(src, dst, graph_generators[dataset].graph.edge_index, multiplex_id=multiplex_id)
                new_indices[dataset][split].append(edge_idx)
                # print(f"Dataset: {dataset}, Split: {split}, Edge: {edge}, Edge Index: {edge_idx}")
                
        if dataset == "ogbl_vessel":
            new_indices[dataset][split] = (pos_edge_indices, neg_edge_indices)
            
# Save new indices
for dataset in datasets:
    new_indices[dataset]["valid"] = new_indices[dataset]["test"]
    json.dump(new_indices[dataset], open(f"{dataset}_split.json", "w"), indent=4)
    print(f"Saved indices for {dataset} {split} split")