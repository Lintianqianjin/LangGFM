import os
import json
import fire
import shutil
from langgfm.data.dataset_generation_coordinator import DatasetGenerationCoordinator

LLAMA_FACTORY_DIR = "LLaMA-Factory-LangGFM"

def exp_dir_to_file_name(exp_dir: str) -> str:
    return exp_dir.replace("/", "__") + ".json"

def update_dataset_info(file_name: str):
    dataset_info_path = f"{LLAMA_FACTORY_DIR}/data/dataset_info.json"
    dataset_info = json.load(open(dataset_info_path))

    new_dataset_info = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system"
        }
    }
    dataset_name = file_name.removesuffix(".json")
    dataset_info[dataset_name] = new_dataset_info

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    return dataset_name

def copy_instruction_dataset_to_data_dir(exp_dir, file_name: str):
    ins_ds = os.path.join(exp_dir, "instruction_dataset.json")
    shutil.copy(ins_ds, f"{LLAMA_FACTORY_DIR}/data/{file_name}")


def main(job_path: str, continue_flag: bool = False, return_token_length: bool = True, tokenizer_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Run the dataset generation pipeline.

    Args:
        job_path (str): Path to the dataset generation job directory.
        continue_flag (bool): Continue from the last checkpoint if set to True.
    """
    job_path = job_path.rstrip('/')
    
    coordinator = DatasetGenerationCoordinator(
        job_path=job_path,
        is_continue=continue_flag,
        return_token_length=return_token_length,
        tokenizer_name_or_path=tokenizer_name_or_path
    )
    coordinator.pipeline()
    print("Dataset generation pipeline completed successfully.")
    
    fname = exp_dir_to_file_name(job_path)
    dataset_name = update_dataset_info(fname)
    copy_instruction_dataset_to_data_dir(job_path, fname)
    

if __name__ == "__main__":
    fire.Fire(main)
