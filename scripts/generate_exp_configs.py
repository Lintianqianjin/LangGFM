import random
import json
import os
import yaml
from ruamel.yaml import YAML

from langgfm.data.graph_generator import InputGraphGenerator
from langgfm.utils.io import save_beautiful_json, load_yaml


# Base directory for experiments
output_dir = "./experiments"
os.makedirs(output_dir, exist_ok=True)

def build_joint_exp_config_by_size():
    # Training sizes for experiments
    training_sizes = [3200, 1600, 800, 400, 200, 100]
    test_size = 200

    # Ensure subdirectories for each training size exist
    for size in training_sizes:
        os.makedirs(os.path.join(output_dir, f"training_size_{size}"), exist_ok=True)

    # Dictionary to store test indices for each dataset
    test_indices = {}

    # Process each dataset in the registry
    for dataset in InputGraphGenerator.registry:
        if dataset in {"aminer", "usa_airport"}:  # Skip specific datasets
            continue

        print(f"Processing dataset: {dataset}")

        # Create the generator for the dataset
        generator = InputGraphGenerator.create(dataset)
        all_samples = generator.all_samples

        # Shuffle and sample a maximum of 5000 samples
        random.shuffle(all_samples)
        samples = all_samples[:min(5000, len(all_samples))]

        # Ensure there are enough samples to create the subsets
        total_required_samples = test_size + max(training_sizes)
        if len(samples) < total_required_samples:
            print(f"Not enough samples in dataset {dataset}. Required: {total_required_samples}, Available: {len(samples)}")
            # raise ValueError(f"Not enough samples in dataset {dataset}. Required: {total_required_samples}, Available: {len(samples)}")

        # Split the samples
        test_set = samples[:test_size]
        largest_training_set = samples[test_size:test_size + max(training_sizes)]

        # Store the test indices
        test_indices[dataset] = test_set

        # Generate training subsets for each size
        for size in training_sizes:
            training_set = largest_training_set[:size]

            # Path for the indices file in the specific training size folder
            indices_file = os.path.join(output_dir, f"training_size_{size}", "indices.json")

            # Load existing data or create a new dictionary
            training_indices = {}
            # Add the training indices for this dataset
            training_indices[dataset] = training_set

            # Save the updated indices
            save_beautiful_json(training_indices, indices_file)
            # with open(indices_file, "w") as f:
            #     json.dump(training_indices, f, indent=4)

        print(f"Saved indices for dataset: {dataset}")

    # Save test indices
    test_file = os.path.join(output_dir, "test_indices.json")
    save_beautiful_json(test_indices, test_file)
    # with open(test_file, "w") as f:
    #     json.dump(test_indices, f, indent=4)

    print("All indices saved successfully.")


# The main function
def build_individual_exp_config_by_size(output_dir):
    # Training sizes for experiments
    training_sizes = [3200, 1600, 800, 400, 200, 100]
    test_size = 200

    # Dictionary to store test indices for each dataset
    test_indices = {}

    # Load the config template
    config_template = load_yaml(os.path.join(output_dir, "config_template.yaml"))
    common_fields = {
        "common_format": config_template["common_format"],
        "common_node_feature_masked_autoencoder": config_template["common_node_feature_masked_autoencoder"],
        "common_edge_feature_masked_autoencoder": config_template["common_edge_feature_masked_autoencoder"]
    }

    yaml_writer = YAML()
    yaml_writer.default_flow_style = False
    yaml_writer.allow_unicode = True

    # Process each dataset in the registry
    for dataset in InputGraphGenerator.registry:
        # Skip specific datasets
        if dataset in {"aminer", "usa_airport"}:
            continue

        print(f"Processing dataset: {dataset}")

        # Create the generator for the dataset
        generator = InputGraphGenerator.create(dataset)
        all_samples = generator.all_samples

        # Shuffle and sample a maximum of 5000 samples
        random.shuffle(all_samples)
        samples = all_samples[:min(5000, len(all_samples))]

        # Ensure there are enough samples to create the subsets
        total_required_samples = test_size + max(training_sizes)
        if len(samples) < total_required_samples:
            print(f"Not enough samples in dataset {dataset}. Required: {total_required_samples}, Available: {len(samples)}")
            continue  # Skip this dataset if insufficient samples

        # Split the samples
        test_set = samples[:test_size]
        largest_training_set = samples[test_size:test_size + max(training_sizes)]

        # Store the test indices
        test_indices[dataset] = test_set

        # Generate training subsets for each size
        for size in training_sizes:
            cur_root = os.path.join(output_dir, dataset, f"train_{size}")
            os.makedirs(cur_root, exist_ok=True)

            training_set = largest_training_set[:size]

            # Path for the indices file in the specific training size folder
            indices_file = os.path.join(cur_root, f"indices.json")
            save_beautiful_json({dataset:training_set}, indices_file)
        
            # --- save the config file ---
            # Create and save the config file for each dataset and size
            config_dataset = {}  # Start with the entire template
            config_dataset.update(common_fields)    # Add the common fields
            config_dataset.update({dataset: config_template.get(dataset, {})})  # Add dataset-specific config

            # Save the YAML file
            with open(os.path.join(cur_root, "data_generation.yaml"), "w") as file:
                yaml_writer.dump(config_dataset, file)

        # Save test indices and config
        test_root = os.path.join(output_dir, dataset, f"test_{test_size}")
        os.makedirs(test_root, exist_ok=True)

        with open(os.path.join(test_root, "data_generation.yaml"), "w") as file:
            yaml_writer.dump(config_dataset, file)
        
        test_file = os.path.join(test_root, "indices.json")
        save_beautiful_json({dataset: test_indices[dataset]}, test_file)

        print(f"Saved indices for dataset: {dataset}")


if __name__ == "__main__":
    # build_joint_exp_config_by_size()
    build_individual_exp_config_by_size(output_dir='./experiments')
    print("Experiment configurations generated successfully.")