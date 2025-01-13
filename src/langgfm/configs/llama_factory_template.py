import os
import re
import copy
import json
import random
import argparse
from tqdm import tqdm

PROMPT_TEMPLATE = (
        "You are a cross-domain, cross-task graph mining expert. "
        "You are proficient in general graph theory knowledge "
        "and have the ability to apply relevant domain-specific knowledge in specific tasks. "
        "You are required to answer specific questions based on the input graph(s) "
        "and the physical meaning description of the graph(s)\n"
        "Below is the physical meaning description of the input graph(s):\n"
        "```\n"
        "{graph_description}\n"
        "```\n"
        "Below is the input graphs(s):\n"
        "```\n"
        "{graph_text}\n"
        "```\n"
        "Below is the question:\n"
        "{query}\n"
)

def convert_unit(sample):
    prompt = PROMPT_TEMPLATE.format_map(sample)
    label = sample["output"]

    return prompt, label

def convert_dataset(dataset_name, load_from_dir="", write_to_dir = ".."):
    # for stage in ['train', 'val', 'test']:
    for stage in ['test']:
        dataset = json.load(open(f"{load_from_dir}/{dataset_name}_{stage}.json"))
        new_dataset = []
        if len(dataset) == 0:
            print(f"{dataset_name}_{stage} is empty!")
        for sample in tqdm(dataset, leave=False, desc=f"{dataset_name}_{stage}"):
            prompt, label = convert_unit(sample)
            new_sample = {
                "instruction": prompt,
                "input": "",
                "output": label
            }
            new_dataset.append(new_sample)

        # os.path.join(target_path, f"../")
        with open(f"{write_to_dir}/{dataset_name}_{stage}.json", 'w') as json_file:
            # Step 4: Dump the data into the file
            json.dump(new_dataset, json_file, indent=4)

    dataset_info = json.load(open(f"../dataset_info.json"))
    dataset_info.update({
        f'{dataset_name}_train': {"file_name": f'{dataset_name}_train.json'}, 
        f'{dataset_name}_val': {"file_name": f'{dataset_name}_val.json'}, 
        f'{dataset_name}_test': {"file_name": f'{dataset_name}_test.json'}, 
    })
    
    with open(f"../dataset_info.json", 'w') as file:
        json.dump(dataset_info, file, indent=4)
# ugm@FB15K237@@@main_max_length=1600_Table_samples_train
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some input parameters.')
    parser.add_argument('--file_name', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--load_from_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--write_to_dir', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    convert_dataset(args.file_name, args.load_from_dir, args.write_to_dir)
    # convert_dataset("ugm@HIV@@@main@ofa_original_single_answer_Table_samples")
    # convert_dataset("ugm@NodeAttributeRetrieval@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@EdgeAttributeRetrieval@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@NodeDegreeCounting@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@ShortestPath@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@MaxTriangleSum@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@HamiltonPath@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@SubgraphMatching@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@GraphStructure@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@GraphAutomorphic@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@SubgraphMatching@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@BBBP@@@mainmax_length=16000_GraphML_samples")
    # convert_dataset("ugm@FreeSolv@@@mainmax_length=16000_GraphML_samples")
    # convert_dataset("ugm@USAAirport@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@Twitch@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@OgbnArxiv@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@OgbnArxiv@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@HamiltonPath@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@SubgraphMatching@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@Twitch@@@main_Table_max_length=16000_samples")
    
    # convert_dataset("ugm@WikiCS@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@WikiCS@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@ShortestPath@@@0@main@OO-42@length8000_GML_samples")
    # convert_dataset("ugm@ShortestPath@@@0@main@OO-42@length8000_GraphML_samples")
    # convert_dataset("ugm@ShortestPath@@@0@main@OO-42@length8000_JSON_samples")
    # convert_dataset("ugm@ShortestPath@@@0@main@OO-42@length8000_Table_samples")
    # convert_dataset("ugm@AMiner@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@MovieLens@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@Fingerprint@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@OgblVessel@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@BACE@@@mainmax_length=16000_Table_samples")
    # convert_dataset("ugm@BBBP@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@BBBP@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@BBBP@@@main_JSON_max_length=16000_samples")
    # convert_dataset("ugm@BBBP@@@main_GML_max_length=16000_samples")
    # convert_dataset("ugm@BACE@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@BACE@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@BACE@@@main_JSON_max_length=16000_samples")
    # convert_dataset("ugm@BACE@@@main_GML_max_length=16000_samples")
    # convert_dataset("ugm@ChEBI20@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@ChEBI20@@@main_GraphML_max_length=16000_samples")
    # convert_dataset("ugm@ChEBI20@@@main_JSON_max_length=16000_samples")
    # convert_dataset("ugm@ChEBI20@@@main_GML_max_length=16000_samples")
    # convert_dataset("ugm@GraphStructure@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@SubgraphMatching@@@main_Table_max_length=16000_samples")

    # convert_dataset("ugm@MaxTriangleSum@@@formatablation_max_length=8000_GML_samples")
    # convert_dataset("ugm@MaxTriangleSum@@@formatablation_max_length=8000_GraphML_samples")
    # convert_dataset("ugm@MaxTriangleSum@@@formatablation_max_length=8000_JSON_samples")
    # convert_dataset("ugm@MaxTriangleSum@@@formatablation_max_length=8000_Table_samples")
    # convert_dataset("ugm@MetaQA@@@main_Table_max_length=16000_samples")
    
    #=================================================================================#
    #============================= Format Ablation Start =============================#
    #=================================================================================#
    # ShortestPath
    # convert_dataset("ugm@ShortestPath@@@formatablation_max_length=8000_GML_samples")
    # convert_dataset("ugm@ShortestPath@@@formatablation_max_length=8000_GraphML_samples")
    # convert_dataset("ugm@ShortestPath@@@formatablation_max_length=8000_JSON_samples")
    # convert_dataset("ugm@ShortestPath@@@formatablation_max_length=8000_Table_samples")
    # OgbnArxiv
    # convert_dataset("ugm@OgbnArxivForAblation@@@main_max_length=8000_GML_samples")
    # convert_dataset("ugm@OgbnArxivForAblation@@@main_max_length=8000_GraphML_samples")
    # convert_dataset("ugm@OgbnArxivForAblation@@@main_max_length=8000_Table_samples")
    # convert_dataset("ugm@OgbnArxivForAblation@@@main_max_length=8000_JSON_samples")
    #=================================================================================#
    #============================== Format Ablation End ==============================#
    #=================================================================================#

    #==========================================================================================#
    #============================= Self-Supervised Ablation Start =============================#
    #==========================================================================================#
    # ShortestPath
    # convert_dataset("ugm@ShortestPath@@@edge_attribute_retrieval_max_length=8000_Table_samples")
    # convert_dataset("ugm@ShortestPath@@@neighbor_retrieval_max_length=8000_Table_samples_all")
    # OgbnArxiv
    # convert_dataset("ugm@OgbnArxivForAblation@@@neighbor_retrieval_max_length=8000_Table_samples")
    # convert_dataset("ugm@OgbnArxivForAblation@@@node_attribute_retrieval_max_length=8000_Table_samples")
    #==========================================================================================#
    #============================= Self-Supervised Ablation Start =============================#
    #==========================================================================================#
    
    # convert_dataset("ugm@SocialCircle@@@main_Table_max_length=16000_samples")
    # convert_dataset("ugm@ESOL@@@mainmax_length=16000_Table_samples")
    
    
    

    
    

    
    
    
    
    

    