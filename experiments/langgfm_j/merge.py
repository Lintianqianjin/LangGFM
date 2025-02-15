import os
import yaml
import json
from collections import defaultdict
import re
import jsbeautifier


def save_beautiful_json(data, file_path):
    """
    Save a JSON file with indentation for human readability
    """
    json_string = json.dumps(data)
    formatted_json = jsbeautifier.beautify(json_string)

    formatted_json = re.sub(r"\],\s*\n\s*\[", "], [", formatted_json)

    # 替换 "[\n        [" 为 "[["
    formatted_json = re.sub(r'\[\s*\n\s*\[', '[[', formatted_json)

    # 替换 "]\n    ]" 为 "]]"
    formatted_json = re.sub(r'\]\s*\n\s*\]', ']]', formatted_json)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(formatted_json)



# Define source and destination paths
source_base = "experiments/langgfm_i"
dest_base = "experiments/langgfm_j"

# List of tasks to combine
<<<<<<< HEAD
tasks = [
    "node_counting", "edge_counting", "node_attribute_retrieval",
    "edge_attribute_retrieval", "degree_counting", "shortest_path",
    "cycle_checking", "hamilton_path", "graph_automorphic",
    "graph_structure_detection", "edge_existence", "connectivity"
]

# Ensure destination directories exist
os.makedirs(f"{dest_base}/train", exist_ok=True)
os.makedirs(f"{dest_base}/test", exist_ok=True)
=======
# tasks = [
#     "node_counting", "edge_counting", "node_attribute_retrieval",
#     "edge_attribute_retrieval", "degree_counting", "shortest_path",
#     "cycle_checking", "hamilton_path", "graph_automorphic",
#     "graph_structure_detection", "edge_existence", "connectivity"
# ]
tasks = [
    "ogbn_arxiv", "wikics", "re_europe", "twitch", "oag_scholar_interest",
    "fb15k237", "movielens1m", "ogbl_vessel", "stack_elec", "yelp_review",
    "fingerprint", "explagraphs", "bace", "esol", "chebi20"
]

# Ensure destination directories exist
# os.makedirs(f"{dest_base}/train", exist_ok=True)
# os.makedirs(f"{dest_base}/test", exist_ok=True)
os.makedirs(f"{dest_base}/train_800", exist_ok=True)
os.makedirs(f"{dest_base}/test_200", exist_ok=True)
>>>>>>> origin/main

# Function to load YAML file while preserving references
def load_yaml(file_path):
    # if not os.path.exists(file_path):
    #     return {}
    with open(file_path, "r") as file:
        return yaml.safe_load(file) or {}

# Function to load JSON file
def load_json(file_path):
    # if not os.path.exists(file_path):
    #     return {}
    with open(file_path, "r") as file:
        return json.load(file) or {}

# Function to save YAML file while preserving references
def save_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

# Function to save JSON file
def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

# Initialize storage for combined YAML content (avoid duplicates)
combined_yaml_train = {}
combined_yaml_test = {}

# Initialize storage for combined JSON indices
<<<<<<< HEAD
combined_json_train = defaultdict(set)
combined_json_test = defaultdict(set)

# Process each task
for task in tasks:
    for split in ["train", "test"]:
=======
combined_json_train = defaultdict(list)
combined_json_test = defaultdict(list)

# Process each task
for task in tasks:
    for split in ["train_800", "test_200"]:
>>>>>>> origin/main
        source_path = f"{source_base}/{task}/{split}"
        
        yaml_file = os.path.join(source_path, "data_generation.yaml")
        json_file = os.path.join(source_path, "indices.json")

        # print("path:")
        # print(os.getcwd())
        # print(os.listdir())
        print(yaml_file)
        print(json_file)

<<<<<<< HEAD
        yaml_content = load_yaml(yaml_file)
=======
        # yaml_content = load_yaml(yaml_file)
>>>>>>> origin/main
        json_content = load_json(json_file)

        # print(yaml_content)
        # print(json_content)
        # exit()

        # Merge YAML data while preserving references
<<<<<<< HEAD
        for key, value in yaml_content.items():
            if key not in ["common_format"]:  # Avoid duplicating common references
                if split == "train":
                    combined_yaml_train[key] = value
                else:
                    combined_yaml_test[key] = value
            else:
                # Preserve the reference notation (e.g., `&id001`)
                if split == "train" and "common_format" not in combined_yaml_train:
                    combined_yaml_train["common_format"] = value
                elif split == "test" and "common_format" not in combined_yaml_test:
                    combined_yaml_test["common_format"] = value

        # Merge JSON indices by keeping task-specific keys
        for key, values in json_content.items():
            if split == "train":
                combined_json_train[key].update(values)
            else:
                combined_json_test[key].update(values)
=======
        # for key, value in yaml_content.items():
        #     if key not in ["common_format"]:  # Avoid duplicating common references
        #         if split == "train_800":
        #             combined_yaml_train[key] = value
        #         else:
        #             combined_yaml_test[key] = value
        #     else:
        #         # Preserve the reference notation (e.g., `&id001`)
        #         if split == "train_800" and "common_format" not in combined_yaml_train:
        #             combined_yaml_train["common_format"] = value
        #         elif split == "test_200" and "common_format" not in combined_yaml_test:
        #             combined_yaml_test["common_format"] = value
        # print(json_content)
        # Merge JSON indices by keeping task-specific keys
        for key, values in json_content.items():
            print(f"{key=}")
            print(f"{values=}")
            if split == "train_800":
                combined_json_train[key]=values
            else:
                combined_json_test[key]=values
>>>>>>> origin/main

# Convert sets back to lists for JSON compatibility
for key in combined_json_train:
    combined_json_train[key] = sorted(list(combined_json_train[key]))
for key in combined_json_test:
    combined_json_test[key] = sorted(list(combined_json_test[key]))

# Save combined files
# save_yaml(combined_yaml_train, f"{dest_base}/train/data_generation.yaml")
<<<<<<< HEAD
save_beautiful_json(combined_json_train, f"{dest_base}/train/indices.json")

# save_yaml(combined_yaml_test, f"{dest_base}/test/data_generation.yaml")
save_beautiful_json(combined_json_test, f"{dest_base}/test/indices.json")
=======
save_beautiful_json(combined_json_train, f"{dest_base}/train_800/indices.json")

# save_yaml(combined_yaml_test, f"{dest_base}/test/data_generation.yaml")
save_beautiful_json(combined_json_test, f"{dest_base}/test_200/indices.json")
>>>>>>> origin/main

print("✅ Successfully combined data into:", dest_base)
