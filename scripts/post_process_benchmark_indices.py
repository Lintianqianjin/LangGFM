import json
import os
import random
import fire
import warnings

from langgfm.utils.random_control import set_seed

def process_and_partition(
    dataset_name,
    max_token_cnt=1200,
    train_sizes="[200,400,800,1600,3200]",
    test_size=500
):
    """
    Loads the JSON file from experiments/benchmark/{dataset_name}/pre_selection/instruction_dataset.json,
    extracts valid sample IDs (i.e. those for which every instance's "#tokens" is <= max_token_cnt),
    saves these valid IDs into a file, partitions them into nested training sets and a disjoint test set,
    and then saves each partition into separate JSON files.
    
    The valid sample IDs are saved in "valid_indices.json" (formatted as {dataset_name: id_list}),
    and for each partition (e.g. "train_200", "test"), a separate file is created with a name like
    "train_200_{dataset_name}_partition.json".
    
    Args:
        dataset_name (str): The dataset name. The input JSON file is assumed to be at:
            experiments/benchmark/{dataset_name}/pre_selection/instruction_dataset.json
        max_token_cnt (int): Maximum allowed token count (default: 1200).
        train_sizes (list or str): A list (or string representation) of desired training set sizes.
            Default is "[200,400,800,1600,3200]". Larger sets will include the IDs of smaller sets.
        test_size (int): Number of samples to reserve for the test set (default: 500).
    """
    # Construct the input JSON file path
    json_file = f"experiments/benchmark/{dataset_name}/pre_selection/instruction_dataset.json"
    
    # Load the JSON data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract valid sample IDs:
    # A valid sample ID is one for which every instance (across all graph_formats) has "#tokens" <= max_token_cnt.
    valid_dict = dict()
    for entry in data:
        # Retrieve metadata and token count from the entry
        metadata = entry.get("metadata", {})
        raw_id = metadata.get("raw_sample_id")
        if isinstance(raw_id, list):
            raw_id = tuple(raw_id)
        token_cnt = entry.get("#tokens", 0)
        # print(f"raw_id: {raw_id}, token_cnt: {token_cnt}")
        
        # Skip the entry if raw_id is missing
        if raw_id is None:
            continue
        
        # Initialize or update the validity for the raw_id
        if raw_id not in valid_dict:
            valid_dict[raw_id] = (token_cnt <= max_token_cnt)
        else:
            valid_dict[raw_id] = valid_dict[raw_id] and (token_cnt <= max_token_cnt)
    
    # Get the list of valid sample IDs
    valid_sample_ids = [raw_id for raw_id, valid in valid_dict.items() if valid]
    
    # Determine the output directory (same as input file's directory)
    output_dir = os.path.dirname(os.path.abspath(json_file))
    
    # Save the valid sample IDs to "valid_indices.json" in the format {dataset_name: id_list}
    valid_indices_file = os.path.join(output_dir, "valid_indices.json")
    with open(valid_indices_file, "w", encoding="utf-8") as f:
        json.dump({dataset_name: valid_sample_ids}, f, ensure_ascii=False, indent=4)
    print(f"Valid sample IDs saved to: {valid_indices_file}")
    
    # --- Partitioning ---
    # Parse train_sizes if provided as a string
    if isinstance(train_sizes, str):
        train_sizes = [int(x.strip()) for x in train_sizes.strip("[]").split(",") if x.strip()]
    train_sizes = sorted(train_sizes)
    
    # Ensure there are enough valid samples: test_size + maximum training size
    required_samples = test_size + max(train_sizes)
    if len(valid_sample_ids) < required_samples:
        # raise ValueError(
        #     f"Not enough valid sample IDs ({len(valid_sample_ids)}) to form a test set of {test_size} "
        #     f"and a training set of size {max(train_sizes)}."
        # )
        warnings.warn(
            f"Not enough valid sample IDs ({len(valid_sample_ids)}) to form a test set of {test_size} "
            f"and a training set of size {max(train_sizes)}. The maximum training size will be adjusted to {(len(valid_sample_ids) - test_size)=}.",
            UserWarning
        )
        
        
    
    # Shuffle the valid_sample_ids randomly
    random.shuffle(valid_sample_ids)
    
    # Reserve the first test_size samples for the test set (ensuring disjointness)
    test_ids = valid_sample_ids[:test_size]
    
    # Use the next max(train_sizes) samples for the training sets
    training_pool = valid_sample_ids[test_size:test_size + max(train_sizes)]
    
    # Build nested training sets: each larger training set includes all IDs from the smaller one.
    train_sets = {}
    for size in train_sizes:
        if size > len(training_pool):
            size = len(training_pool)
            train_sets[f"train_{size}"] = training_pool[:size]
            break
        else:
            train_sets[f"train_{size}"] = training_pool[:size]
        
    
    # Build the partition result dictionary.
    # Each partition is stored as {dataset_name: [id_list]}.
    partition_result = {}
    for key, ids in train_sets.items():
        partition_result[key] = {dataset_name: ids}
    partition_result[f"test_{test_size}"] = {dataset_name: test_ids}
    
    # Save each partition in a separate JSON file.
    # For each partition key, create a file named "{partition_key}_partition.json"
    for partition_key, partition_data in partition_result.items():
        # Create a folder for the partition (e.g., output_dir/train_200)
        partition_folder = os.path.join(output_dir, partition_key)
        os.makedirs(partition_folder, exist_ok=True)
        partition_file = os.path.join(partition_folder, "indices.json")
        with open(partition_file, "w", encoding="utf-8") as f:
            json.dump(partition_data, f, ensure_ascii=False, indent=4)
        print(f"{partition_key} partition saved to: {partition_file}")
    
    # Optionally, print a summary of the partition results
    print("Partitioning complete. Summary:")
    for key, part in partition_result.items():
        ids = part.get(dataset_name, [])
        print(f"  {key}: {len(ids)} samples")

if __name__ == "__main__":
    set_seed(0)
    fire.Fire(process_and_partition)
