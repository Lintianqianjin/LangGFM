import os
import subprocess
import fire

import logging
logger = logging.getLogger("root")


def run_llamafactory_vllm_infer(model_name_or_path, adapter_name_or_path, dataset, output_dir, **kwargs):
    if model_name_or_path.startswith("."):
        absolute_model_path = os.path.abspath(model_name_or_path)
    else:
        absolute_model_path = model_name_or_path
        
    if adapter_name_or_path is not None:
        absolute_adapter_path = os.path.abspath(adapter_name_or_path)
    else:
        absolute_adapter_path = None
        
    output_dir = os.path.abspath(output_dir)
    logger.info(f"{output_dir=}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    absolute_output_dir_path = os.path.join(output_dir,"predictions.json")
    
    llama_factory_dir = "LLaMA-Factory"
    os.chdir(llama_factory_dir)

    command = f"VLLM_SKIP_P2P_CHECK=1 DISABLE_VERSION_CHECK=1 python scripts/vllm_infer.py --model_name_or_path {absolute_model_path} --adapter_name_or_path {absolute_adapter_path} --dataset {dataset} --save_name {absolute_output_dir_path} --cutoff_len 16000 --top_k 1 --temperature 0.01"

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    else:
        print("Command executed successfully!")


def main(model_name_or_path: str, adapter_name_or_path:str = None, dataset:str = None, output_dir:str = None, **kwargs):
    """
    """
    
    run_llamafactory_vllm_infer(model_name_or_path, adapter_name_or_path, dataset, output_dir, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
    