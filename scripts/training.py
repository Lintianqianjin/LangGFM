import os
import torch
import random
import json
import shutil
import shlex
import subprocess
import fire
import yaml
import psutil
from pathlib import Path


def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

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
    assert batch_size >= per_device_train_batch_size*num_gpus, "Batch size should be >= than per_device_train_batch_size*num_gpus"
    assert batch_size % (per_device_train_batch_size*num_gpus) == 0, "Batch size should be divisible by per_device_train_batch_size*num_gpus"
    gradient_accumulation_steps = batch_size//num_gpus//per_device_train_batch_size
    
    # if 
    output_dir = os.path.join(
        file_path,
        "ckpts",
        model_name_or_path,
        f"{lora_rank=}",
        f"{lora_alpha=}",
        f"{lora_dropout=}",
        f"{learning_rate=}",
        f"{num_train_epochs=}",
        f"{warmup_ratio=}",
        f"{batch_size=}"
    )
    
    if "Qwen" in model_name_or_path:
        template = "qwen"
    elif "Llama-3" in model_name_or_path:
        template = "llama3"
    else:
        template = None
    
    path = Path(output_dir)
    
    # Check if the path exists
    if path.exists():
        print("This experiment has already been run.")
        # return
    else:
        # Create the path
        path.mkdir(parents=True, exist_ok=True)
    # print("Path created. Starting the experiment.")
    
    _model = {
        # model
        "model_name_or_path": model_name_or_path,
        "trust_remote_code": kwargs.get("trust_remote_code", True),
    }    
    
    _method = {
        "stage": kwargs.get("stage", "sft"),
        "do_train": kwargs.get("do_train", True),
        "do_eval": kwargs.get("do_eval", False),
        "do_predict": kwargs.get("do_predict", False),
        "finetuning_type": kwargs.get("finetuning_type", "lora")
    }
    if _method['finetuning_type'] == "lora":
        _lora_config = {
            "lora_alpha": kwargs.get("lora_alpha", 16),
            "lora_dropout": kwargs.get("lora_dropout", 0.),
            "lora_rank": kwargs.get("lora_rank", 8),
            "lora_target": kwargs.get("lora_target", "all"),
            "use_rslora": kwargs.get("use_rslora", False),
        }
        _method = _method | _lora_config
    
    if kwargs.get("deepspeed", None) is not None:
        _method["deepspeed"] = kwargs.get("deepspeed", None)
    
    _dataset = {
        "dataset_dir": kwargs.get("dataset_dir", None),
        "dataset": kwargs.get("dataset", ""),
        "template": template,
        "cutoff_len": kwargs.get("cutoff_len", 15000),
        "max_samples": kwargs.get("max_samples", 1000000),
        "overwrite_cache": kwargs.get("overwrite_cache", True),
        "preprocessing_num_workers": kwargs.get("preprocessing_num_workers", max_processor_count),
    }
    
    _output = {
        "output_dir": kwargs.get("output_dir", output_dir),
        "logging_steps": kwargs.get("logging_steps", 2),
        "save_steps": kwargs.get("save_steps", 100),
        "plot_loss": kwargs.get("plot_loss", True),
        "overwrite_output_dir": kwargs.get("overwrite_output_dir", True),
    }    
    _train = {
        "per_device_train_batch_size": kwargs.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": kwargs.get("learning_rate", 1.0e-4),
        "num_train_epochs": kwargs.get("num_train_epochs", 5.0),
        "lr_scheduler_type": kwargs.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": kwargs.get("warmup_ratio", 0.4),
        "bf16": kwargs.get("bf16", True),
        "ddp_timeout": kwargs.get("ddp_timeout", 180000000),
        "flash_attn": kwargs.get("flash_attn", "fa2"),
        "enable_liger_kernel": kwargs.get("enable_liger_kernel", True),
       
        "report_to": kwargs.get("report_to", "wandb"),
        "run_name": kwargs.get("run_name", kwargs.get("dataset")), # keep runname same as dataset name if not provided
        
        # "load_best_model_at_end": kwargs.get("load_best_model_at_end", False),
        # "metric_for_best_model": kwargs.get("metric_for_best_model", None),
    }
    _eval = {
        "eval_dataset": kwargs.get("eval_dataset", None),
        "bf16_full_eval": kwargs.get("bf16_full_eval", True),
        "val_size": kwargs.get("val_size", 0.0),
        "per_device_eval_batch_size": kwargs.get("per_device_eval_batch_size", 1),
        "eval_strategy": kwargs.get("eval_strategy", "steps"),
        "eval_steps": kwargs.get("eval_steps", 100),
        # "compute_accuracy": kwargs.get("compute_accuracy", True),
        "predict_with_generate": kwargs.get("predict_with_generate", True),
        # "do_sample": kwargs.get("do_sample", False),
        # "max_new_tokens": kwargs.get("max_new_tokens", 4),
    }
    
    # if "output_logits" in kwargs:
    #     _eval["output_logits"] = kwargs.get("output_logits", False)
    #     if kwargs["output_logits"]:
    #         _eval["return_dict_in_generate"] = True
    

    data = _model | _method | _dataset | _output | _train | _eval
    
    yaml_path = f"{output_dir}/traning.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)
    
    return yaml_path

def exp_dir_to_file_name(exp_dir: str) -> str:
    return exp_dir.replace("/", "__") + ".json"

def run_llamafactory_training(ymal_path: str):
    absolute_ymal_path = os.path.abspath(ymal_path)
    yaml_config = load_yaml_config(absolute_ymal_path)
    
    # llama_factory_dir = "LangGFM-SFT"
    # os.chdir(llama_factory_dir)

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
    nnodes=os.getenv("NNODES", "1")
    node_rank=os.getenv("NODE_RANK", "0")
    nproc_per_node=os.getenv("NPROC_PER_NODE", str(torch.cuda.device_count()))
    # add all the key-value pairs in the yaml file as command line arguments
    cmd_args = " ".join(f"--{key} {shlex.quote(str(value))}" for key, value in yaml_config.items())

    # full cmd
    command = (f"DISABLE_VERSION_CHECK=1 torchrun --nnodes {nnodes} --node_rank {node_rank} "
               f"--nproc_per_node {nproc_per_node} --master_addr {master_addr} --master_port {master_port} "
               f"training/llamafactory/src/train.py {cmd_args}")
    
    print("\n",command,"\n")
    
    # exit()
    # command = f"DISABLE_VERSION_CHECK=1 python src/train.py {absolute_ymal_path}"

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # env={"DISABLE_VERSION_CHECK": "1"}
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    else:
        print("Command executed successfully!")

def process_single_dir(dir_path: str):
    dir_path = dir_path.rstrip('/')
    fname = exp_dir_to_file_name(dir_path)
    fname = fname.removesuffix('.json')
    return fname

def main(train_dir: str, eval_dir:str = None, exp_dir:str = None, **kwargs):
    """
    Run the training pipeline basd on Llamafactory.
    Args:
        train_dir (str): "path1,path2,path3" or "path1" Path to the training dataset, where contains `instrcution_dataset.json`
        eval_dir (str): Path to the evaluation dataset, where contains `instrcution_dataset.json`
        exp_dir (str): Path to the experiment directory, where yaml and ckpt will be saved.
    """
    
    train_dirs = train_dir.strip(',').split(',')
    if len(train_dirs) == 1:
        train_dir = train_dirs[0]
        exp_dir = train_dir if exp_dir is None else exp_dir
        train_fname = process_single_dir(train_dir)
        # train_dir = train_dir.rstrip('/')
        # train_fname = exp_dir_to_file_name(train_dir)
        # train_fname = train_fname.removesuffix('.json')
    else: # multiple training datasets
        assert exp_dir is not None, "exp_dir must be provided for multiple training datasets."
        train_fnames = [process_single_dir(d) for d in train_dirs]
        train_fname = ','.join(train_fnames)
    
    eval_dirs = eval_dir.strip(',').split(',')
    if len(eval_dirs) == 1:
        eval_dir = eval_dirs[0]
        eval_fname = process_single_dir(eval_dir)
        # eval_dir = eval_dir.rstrip('/')
        # eval_fname = exp_dir_to_file_name(eval_dir)
        # eval_fname = eval_fname.removesuffix('.json')
    else:
        eval_fnames = [process_single_dir(d) for d in eval_dirs]
        eval_fname = ','.join(eval_fnames)
        
    kwargs['eval_dataset'] = eval_fname

    yaml_file_path = generate_yaml_file(file_path=f"{exp_dir}", dataset=train_fname, **kwargs)
    
    run_llamafactory_training(yaml_file_path)
    
if __name__ == "__main__":
    fire.Fire(main)
