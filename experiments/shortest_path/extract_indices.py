import os 
import json
from langgfm.utils.io import save_beautiful_json

def extract_indices(file_path: str):
    """
    Extract the indices of the samples in the input JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    indices = [sample['metadata']['raw_sample_id'] for sample in samples]
    indices = {"shortest_path": list(set(indices))}
    # Save the indices to a new JSON file
    save_beautiful_json(indices, os.path.join(file_path.split('instruction_dataset.json')[0], "indices.json"))
    # output_file = f"""{os.path.join(file_path.split('instruction_dataset.json')[0], "indices.json")}"""
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(indices, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import fire
    fire.Fire(extract_indices)