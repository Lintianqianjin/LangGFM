import os
import time
import subprocess
import requests
import sys
import socket
import json
import fire

from eval_utils import get_best_trained_model



def check_vllm_server(host="localhost", port=8016, max_retries=5, delay=3):
    """
    检查 vLLM API 服务器是否正常运行
    :param host: 服务器地址
    :param port: 服务器端口
    :param max_retries: 最大重试次数
    :param delay: 每次重试之间的间隔时间（秒）
    :return: 是否启动成功（True/False）
    """
    timeout = 600
    # api_url = f"{host}:{port}/v1/models"
    start_time = time.time()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                # 如果能连接上，说明端口已被监听
                return True
            except ConnectionRefusedError:
                time.sleep(2)

        if (time.time() - start_time) > timeout:
            return False


def pipeline(datasets, model_name, hg_model, gpu_id=0, port=8016, min_ckpt_idx=50, max_ckpt_idx=2425, exp_prefix="langgfm_i", **kwargs):
    """Runs the vLLM server, performs inference, and shuts down the server."""

    print(f"\n🚀 Starting Pipeline for Exp: {exp_prefix} | Model: {model_name} | GPU: {gpu_id} | Port: {port}")
    # safe, for file/module... names
    safe_exp_prefix = exp_prefix.replace("/","-")
    safe_hg_model = hg_model.split("/")[-1]
    # result = []
    # code for iter list of epochs, 50 epochs per list, epoch interval is 25, min epoch is 25 ,max epoch is 1600
    def epochs_iter():
        batch_size = 50
        for i in range(min_ckpt_idx, max_ckpt_idx, 25*batch_size):
            yield [i+j*100 for j,_ in enumerate(range(min_ckpt_idx, min(min_ckpt_idx+100*batch_size, max_ckpt_idx), 100))]
    
    # Log file for the server
<<<<<<< HEAD
    log_file = f"logs/LangGFM-{safe_exp_prefix}-{safe_hg_model}.log"
=======
    log_file = f"LangGFM-J-{safe_exp_prefix}-{safe_hg_model}.log"
>>>>>>> origin/main
    
    warmup_ratio = kwargs.get("warmup_ratio", 0.2)
    num_train_epochs = kwargs.get("num_train_epochs", 50)
    batch_size = kwargs.get("batch_size", 64)

    for epoch_list in epochs_iter():
        # lora_rank=256/lora_alpha=1024/lora_dropout=0.1/learning_rate=2e-05/num_train_epochs=20/warmup_ratio=0.4/batch_size=128/checkpoint-300
        lora_modules = [
            {"name": f"LangGFM-{safe_exp_prefix}-{safe_hg_model}-{epoch}", 
            #  "path": f"experiments/{exp_prefix}/train/ckpts/{hg_model}/lora_rank=256/lora_alpha=1024/lora_dropout=0.1/learning_rate=2e-05/num_train_epochs=20/warmup_ratio=0.4/batch_size=128/checkpoint-{epoch}", 
             "path": f"experiments/{exp_prefix}/ckpts/{hg_model}/lora_rank=256/lora_alpha=1024/lora_dropout=0.1/learning_rate=2e-05/num_train_epochs=20/warmup_ratio=0.4/batch_size=128/checkpoint-{epoch}", 
             "base_model_name": hg_model}
            for epoch in epoch_list
        ]

        print("Epochs to be evaluated:", epoch_list)
        # Construct the vLLM server command
        vllm_command = (
            f"nohup vllm serve {hg_model} "
            f"""--enable-lora --lora-modules '{"' '".join([json.dumps(lora) for lora in lora_modules])}' """
            f"--api-key 12345 --host 0.0.0.0 --port {port} "
            f"--max-model-len 16000 --max-lora-rank 256 > {log_file} 2>&1 &"
        )
        print("\n")
        print(vllm_command)
        print("\n")
        # exit()
        print(f"🚀 Initiating vLLM Server of {exp_prefix} on GPU {gpu_id} (Port {port})...")
        # os.system(vllm_command)
        process = subprocess.Popen(vllm_command, shell=True, stdout=subprocess.PIPE, executable="/bin/bash")
        # pid = process.stdout.read().strip().decode()
        print(f"✅ Server started, logs are being written to `{log_file}`.")



        # Give the server time to start before running inference
        time.sleep(15)  # Adjust if needed
        if check_vllm_server(port=port):
            print("✅ Server is running and ready for inference.")
        else:
            print("❌ Server failed to start (TIMEOUT in check_vllm_server). Exiting...")
            return
        
        for dataset in datasets:
            print(f"🔍 Running inference on dataset {dataset}...")
            for epoch in epoch_list: 
                # Construct the inference command
                # tmp: infer from langgfm-i
                inference_command = (
                    f"python scripts/eval_langgfm_api.py "
                    f"--api_key 12345 --url http://localhost:{port}/v1 "
<<<<<<< HEAD
                    f"--file_path experiments/langgfm_i/{dataset}/test/instruction_dataset.json "
=======
                    f"--file_path experiments/langgfm_i/{dataset}/test_200/instruction_dataset.json "
>>>>>>> origin/main
                    f"--model_name LangGFM-{safe_exp_prefix}-{safe_hg_model}-{epoch}"
                )
                
                print()
                print(inference_command)
                print()

                try:
                    inference_process = subprocess.run(inference_command, shell=True, check=True)
                    print("✅ Inference completed successfully.")
                except subprocess.CalledProcessError as e:
                    print("❌ Inference failed with error code:", e.returncode)
            
        print("🛑 Shutting down the vLLM server...")
        os.system(f"pkill -f vllm")
        
        # os.system(f"pkill -f vllm serve")  # Shut down only the specific instance
        # os.system(f"kill -9 {pid}")  # Shut down only the specific instance
        time.sleep(10)  # Ensure processes are fully terminated

        print(f"🎉 Pipeline completed for Dataset: | Model: {model_name} | GPU: {gpu_id} | Port: {port}\n")


def main(model_name, datasets=None, min_ckpt_idx=50, max_ckpt_idx=1200, exp_prefix="langgfm_i", **kwargs):
    """Runs the pipeline for the given model on a specific GPU."""
    
    # datasets = structure_datasets
    datasets = [
<<<<<<< HEAD
        "node_counting",
        "edge_counting",
        "node_attribute_retrieval",
        "edge_attribute_retrieval",
        "degree_counting",
        "edge_existence",
        "cycle_checking",
        "hamilton_path",
        "connectivity",
        "graph_automorphic",
        "shortest_path",
        "graph_structure_detection"
=======
        # "node_counting",
        # "edge_counting",
        # "node_attribute_retrieval",
        # "edge_attribute_retrieval",
        # "degree_counting",
        # "edge_existence",
        # "cycle_checking",
        # "hamilton_path",
        # "connectivity",
        # "graph_automorphic",
        # "shortest_path",
        # "graph_structure_detection"
        "ogbn_arxiv",
        # "wikics",
        # "re_europe"
>>>>>>> origin/main
    ]

    gpu_ports = {
        "meta-llama/Llama-3.1-8B-Instruct": {"gpu_id": "0", "port": 8018, "HG_MODEL":"meta-llama/Llama-3.1-8B-Instruct"},
        "Qwen/Qwen2.5-7B-Instruct": {"gpu_id": "0", "port": 8019, "HG_MODEL":"Qwen/Qwen2.5-7B-Instruct"}, 
    }

    if model_name not in gpu_ports:
        print(f"❌ Unsupported model: {model_name}. Choose from {list(gpu_ports.keys())}")
        return

    gpu_id = gpu_ports[model_name]["gpu_id"]
    port = gpu_ports[model_name]["port"]
    hg_model = gpu_ports[model_name]["HG_MODEL"]
    

    # for dataset in datasets:
    pipeline(datasets=datasets, model_name=model_name, hg_model=hg_model, gpu_id=gpu_id, port=port, min_ckpt_idx=min_ckpt_idx, max_ckpt_idx=max_ckpt_idx, exp_prefix=exp_prefix,**kwargs)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python pipeline_script.py <model_name>")
    #     sys.exit(1)

    # model_name = sys.argv[1]
    # main(model_name)
    fire.Fire(main)
    # main("meta-llama/Llama-3.1-8B-Instruct", datasets)
