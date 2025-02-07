import os
import torch
import json
import shutil
import subprocess
import fire
import yaml
import psutil
from pathlib import Path

def generate_yaml_file(file_path=None, **kwargs):
    # run_name = file_path.replace("/", "__")
    
    # get absolute path of file_path
    if file_path is not None:
        file_path = os.path.abspath(file_path)
    
    # get max processor count using psutil
    max_processor_count = psutil.cpu_count(logical=True)
    
    # create a subdir for output
    model_name_or_path = kwargs.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    lora_rank = kwargs.get("lora_rank", 8)
    lora_alpha = kwargs.get("lora_alpha", 16)
    lora_dropout = kwargs.get("lora_dropout", 0.)
    num_train_epochs = kwargs.get("num_train_epochs", 10.0)
    learning_rate = kwargs.get("learning_rate", 1.0e-4)
    warmup_ratio = kwargs.get("warmup_ratio", 0.4)
    
    per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 1)
    
    num_gpus = torch.cuda.device_count()
    batch_size = kwargs.get("batch_size", 32)
    gradient_accumulation_steps = batch_size//num_gpus//per_device_train_batch_size
    # num_gpus*per_device_train_batch_size*gradient_accumulation_steps
    gradient_accumulation_steps = gradient_accumulation_steps
    
    output_dir = os.path.join(
        file_path,
        "ckpts",
        model_name_or_path.split('/')[1],
        f"{lora_rank=}",
        f"{lora_alpha=}",
        f"{lora_dropout=}",
        f"{learning_rate=}",
        f"{num_train_epochs=}",
        f"{warmup_ratio=}",
        f"{batch_size=}"
    )
    
    
    path = Path(output_dir)
    
    # Check if the path exists
    if path.exists():
        print("This experiment has already been run.")
        # return
    else:
        # Create the path
        path.mkdir(parents=True, exist_ok=True)
    # print("Path created. Starting the experiment.")
    
    data = {
        # model
        "model_name_or_path": kwargs.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct"),
        "trust_remote_code": kwargs.get("trust_remote_code", True),
        
        # method
        "stage": kwargs.get("stage", "sft"),
        "do_train": kwargs.get("do_train", True),
        "finetuning_type": kwargs.get("finetuning_type", "lora"),
        "lora_alpha": kwargs.get("lora_alpha", 16),
        "lora_dropout": kwargs.get("lora_dropout", 0.),
        "lora_rank": kwargs.get("lora_rank", 8),
        "lora_target": kwargs.get("lora_target", "all"),
        "use_rslora": kwargs.get("use_rslora", False),

        # dataset
        "dataset": kwargs.get("dataset", ""),
        "template": kwargs.get("template", "qwen"),
        "cutoff_len": kwargs.get("cutoff_len", 15000),
        "max_samples": kwargs.get("max_samples", 100000),
        "overwrite_cache": kwargs.get("overwrite_cache", True),
        "preprocessing_num_workers": kwargs.get("preprocessing_num_workers", max_processor_count),
        
        # output
        "output_dir": kwargs.get("output_dir", output_dir),
        "logging_steps": kwargs.get("logging_steps", 2),
        "save_steps": kwargs.get("save_steps", 100),
        "plot_loss": kwargs.get("plot_loss", True),
        "overwrite_output_dir": kwargs.get("overwrite_output_dir", True),
        
        # train
        "per_device_train_batch_size": kwargs.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 16),
        "learning_rate": kwargs.get("learning_rate", 1.0e-4),
        "num_train_epochs": kwargs.get("num_train_epochs", 5.0),
        "lr_scheduler_type": kwargs.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": kwargs.get("warmup_ratio", 0.4),
        "bf16": kwargs.get("bf16", True),
        "ddp_timeout": kwargs.get("ddp_timeout", 180000000),
        "flash_attn": kwargs.get("flash_attn", "fa2"),
        "enable_liger_kernel": kwargs.get("enable_liger_kernel", True),
        
        "report_to": kwargs.get("report_to", "wandb"),
        "run_name": kwargs.get("dataset", ""), # keep runname same as dataset name
        
        "load_best_model_at_end": kwargs.get("load_best_model_at_end", True),
        "metric_for_best_model": kwargs.get("metric_for_best_model", "eval_accuracy"),

        # eval
        "eval_dataset": kwargs.get("eval_dataset", None),
        "bf16_full_eval": kwargs.get("bf16_full_eval", True),
        "val_size": kwargs.get("val_size", 0.0),
        "per_device_eval_batch_size": kwargs.get("per_device_eval_batch_size", 1),
        "eval_strategy": kwargs.get("eval_strategy", "steps"),
        "eval_steps": kwargs.get("eval_steps", 100),
        "compute_accuracy": kwargs.get("compute_accuracy", True)
    }

    yaml_path = f"{output_dir}/traning.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)
    
    return yaml_path

def exp_dir_to_file_name(exp_dir: str) -> str:
    return exp_dir.replace("/", "__") + ".json"


def update_dataset_info(file_name: str):
    dataset_info_path = "LLaMA-Factory/data/dataset_info.json"
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
    shutil.copy(ins_ds, f"LLaMA-Factory/data/{file_name}")


def run_llamafactory_training(ymal_path: str):
    absolute_ymal_path = os.path.abspath(ymal_path)
    llama_factory_dir = "LLaMA-Factory"
    os.chdir(llama_factory_dir)

    command = f"DISABLE_VERSION_CHECK=1 llamafactory-cli train {absolute_ymal_path}"

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


def main(train_dir: str, eval_dir:str = None, **kwargs):
    """
    Run the training pipeline basd on Llamafactory.
    Args:
        train_dir (str): Path to the training dataset, where contains `instrcution_dataset.json`
        eval_dir (str): Path to the evaluation dataset, where contains `instrcution_dataset.json`
    """
    train_dir = train_dir.rstrip('/')
    eval_dir = eval_dir.rstrip('/')
    
    if eval_dir is not None:
        eval_fname = exp_dir_to_file_name(eval_dir)
        eval_dataset_name = update_dataset_info(eval_fname)
        copy_instruction_dataset_to_data_dir(eval_dir, eval_fname)
        kwargs['eval_dataset'] = eval_dataset_name
        
    train_fname = exp_dir_to_file_name(train_dir)
    train_dataset_name = update_dataset_info(train_fname)
    copy_instruction_dataset_to_data_dir(train_dir, train_fname)
    
    yaml_file_path = generate_yaml_file(file_path=f"{train_dir}", dataset=train_dataset_name, **kwargs)
    
    run_llamafactory_training(yaml_file_path)


if __name__ == "__main__":
    fire.Fire(main)
