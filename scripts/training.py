import os
import json
import shutil
import argparse
import subprocess

# define a function to translate a exp_dir to a file_name
def exp_dir_to_file_name(exp_dir: str) -> str:
    """
        keep the dir info in file name
    """
    return exp_dir.replace("/", "__") + ".json"


# register in dataset_info.json in LLaMA-Factory
def update_dataset_info(file_name: str):
    ## load dataset_info.json
    dataset_info = json.load(open("LLaMA-Factory/data/dataset_info.json"))

    ## append new dataset info to dataset_info.json
    ## refer to https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
    #
    # "dataset_name": {
    #   "file_name": "data.json",
    #   "columns": {
    #     "prompt": "instruction",
    #     "query": "input",
    #     "response": "output",
    #     "system": "system",
    #     "history": "history"
    #   }
    # }
    #
    new_dataset_info = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            # "system": "system",
            # "history": "history"
        }
    }
    dataset_info[file_name.strip(".json")] = new_dataset_info
    # save dataset_info.json
    with open("LLaMA-Factory/data/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=4)

## copy the target instruction_dataset.json to LLaMA-Factory/src/llamafactory/data/
def copy_instruction_dataset_to_data_dir(exp_dir, file_name: str):
    shutil.copy(f"{exp_dir}/instruction_dataset.json", f"LLaMA-Factory/data/{file_name}")
    
# define a function to excute training commands based on os and subprocess
def run_llamafactory_training(exp_dir: str):
    absolute_exp_dir = os.path.abspath(exp_dir)
    
    llama_factory_dir = "LLaMA-Factory"
    os.chdir(llama_factory_dir)
    
    command =  f"DISABLE_VERSION_CHECK=1 llamafactory-cli train {absolute_exp_dir}/training.yaml"
    
    # Use subprocess.Popen with shell=True
    process = subprocess.Popen(
        command,
        shell=True,               # Enable shell mode
        stdout=subprocess.PIPE,   # Capture standard output
        stderr=subprocess.STDOUT, # Merge standard error output into standard output
        text=True                 # Treat output as a string instead of bytes
    )

    # Read and print the output in real-time
    for line in process.stdout:
        print(line, end="")  # Print the command's real-time output

    # Wait for the process to complete
    process.wait()

    # Check the return code
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    else:
        print("Command executed successfully!")


if __name__ == "__main__":
    
    
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    
    
    fname = exp_dir_to_file_name(args.exp_dir)
    update_dataset_info(fname)
    copy_instruction_dataset_to_data_dir(args.exp_dir, fname)
    run_llamafactory_training(args.exp_dir)