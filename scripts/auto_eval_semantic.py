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


def pipeline(dataset, model_name, hg_model, gpu_id=0, port=8016, min_ckpt_idx=50, max_ckpt_idx=2425, exp_prefix="langgfm_i", **kwargs):
    """Runs the vLLM server, performs inference, and shuts down the server."""

    print(f"\n🚀 Starting Pipeline for Dataset: {dataset} | Model: {model_name} | GPU: {gpu_id} | Port: {port}")

    # result = []
    # code for iter list of epochs, 50 epochs per list, epoch interval is 25, min epoch is 25 ,max epoch is 1600
    def epochs_iter():
        batch_size = 50
        epochs = list(range(min_ckpt_idx, max_ckpt_idx, 25))
        for i in range(len(epochs)//batch_size+1):
            yield epochs[i*batch_size:i*batch_size+batch_size]
    
    # Log file for the server
    log_file = f"{exp_prefix}_{dataset}_{hg_model.split('/')[1]}.log"
    
    warmup_ratio = kwargs.get("warmup_ratio", 0.2)
    batch_size = kwargs.get("batch_size", 64)
    lora_alpha = kwargs.get("lora_alpha", 256)
    for epoch_list in epochs_iter():
        lora_modules = [
            {"name": f"{hg_model}-LangGFM-{exp_prefix}-{epoch}", 
             "path": f"experiments/{exp_prefix}/{dataset}/train_800/ckpts/{hg_model}/lora_rank=64/lora_alpha={lora_alpha}/lora_dropout=0.0/learning_rate=2e-05/num_train_epochs=50/warmup_ratio={warmup_ratio}/batch_size={batch_size}/checkpoint-{epoch}", 
             "base_model_name": hg_model}
            for epoch in epoch_list
        ]

        
        # Construct the vLLM server command
        vllm_command = (
            f"nohup vllm serve {hg_model} "
            f"--enable-lora --lora-modules '{"' '".join([json.dumps(lora) for lora in lora_modules])}' "
            f"--api-key 12345 --host 0.0.0.0 --port {port} "
            f"--max-model-len 16000 --max-lora-rank 64 > {log_file} 2>&1 &"
        )
        print("\n")
        print(vllm_command)
        print("\n")

        print(f"🚀 Initiating vLLM Server for {dataset} of {exp_prefix} on GPU {gpu_id} (Port {port})...")
        # os.system(vllm_command)
        process = subprocess.Popen(vllm_command, shell=True, stdout=subprocess.PIPE, executable="/bin/bash")
        # pid = process.stdout.read().strip().decode()
        print(f"✅ Server started, logs are being written to `{log_file}`.")



        # Give the server time to start before running inference
        time.sleep(15)  # Adjust if needed

        # 检查服务器是否成功启动
        # if not check_vllm_server(port=port):  # Check corresponding port
        #     print("❌ Server failed to start. Exiting...")
        #     return

        if check_vllm_server(port=port):
            print("✅ Server is running and ready for inference.")
        else:
            print("❌ Server failed to start (TIMEOUT in check_vllm_server). Exiting...")
            return

        for epoch in epoch_list: 
            # Construct the inference command
            inference_command = (
                f"python scripts/eval_langgfm_api.py "
                # f"--api_key 12345 --port {port} "
                f"--file_path experiments/{exp_prefix}/{dataset}/test_200/instruction_dataset.json "
                f"--model_name {hg_model}-LangGFM-{exp_prefix}-{epoch}"
            )
            
            print()
            print(inference_command)
            print()

            print("🔍 Running inference...")
            try:
                inference_process = subprocess.run(inference_command, shell=True, check=True)
                print("✅ Inference completed successfully.")
            except subprocess.CalledProcessError as e:
                print("❌ Inference failed with error code:", e.returncode)
            

            # if inference_process.returncode == 0:
            #     print("✅ Inference completed successfully.")
            # else:
            #     print("❌ Inference failed.")

                    
            # Shut down the vLLM server gracefully
        print("🛑 Shutting down the vLLM server...")
        os.system(f"pkill -f vllm")
        
        # os.system(f"pkill -f vllm serve")  # Shut down only the specific instance
        # os.system(f"kill -9 {pid}")  # Shut down only the specific instance
        time.sleep(10)  # Ensure processes are fully terminated

        print(f"🎉 Pipeline completed for Dataset: {dataset} | Model: {model_name} | GPU: {gpu_id} | Port: {port}\n")


def main(model_name, dataset, min_ckpt_idx=50, max_ckpt_idx=1200, exp_prefix="langgfm_i",**kwargs):
    """Runs the pipeline for the given model on a specific GPU."""

    datasets = [
        dataset
    ]

    gpu_ports = {
        "meta-llama/Llama-3.1-8B-Instruct": {"gpu_id": "0", "port": 8016, "HG_MODEL":"meta-llama/Llama-3.1-8B-Instruct"},
        "Qwen/Qwen2.5-7B-Instruct": {"gpu_id": "0", "port": 8016, "HG_MODEL":"Qwen/Qwen2.5-7B-Instruct"}, 
    }

    if model_name not in gpu_ports:
        print(f"❌ Unsupported model: {model_name}. Choose from {list(gpu_ports.keys())}")
        return

    gpu_id = gpu_ports[model_name]["gpu_id"]
    port = gpu_ports[model_name]["port"]
    hg_model = gpu_ports[model_name]["HG_MODEL"]

    for dataset in datasets:
        pipeline(dataset=dataset, model_name=model_name, hg_model=hg_model, gpu_id=gpu_id, port=port, min_ckpt_idx=min_ckpt_idx, max_ckpt_idx=max_ckpt_idx, exp_prefix=exp_prefix,**kwargs)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python pipeline_script.py <model_name>")
    #     sys.exit(1)

    # model_name = sys.argv[1]
    # main(model_name)
    fire.Fire(main)
