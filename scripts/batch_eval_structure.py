import os
import time
import subprocess
import requests
import sys
import socket

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
    timeout = 30
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


def get_all_best_models(datasets, model_name):
    """Retrieve the best trained model path for each dataset."""
    best_models = {}
    for dataset in datasets:
        best_model = get_best_trained_model(dataset, model_name)
        if best_model:
            best_models[dataset] = best_model.split("LangGFM/")[-1]
        else:
            print(f"⚠️ No best model found for dataset: {dataset}, model: {model_name}. Skipping...")
    return best_models


def start_vllm_servers(best_models, model_name, gpu_id, port, hg_model):
    """Starts the vLLM servers for all datasets."""
    log_file = f"logs/server_{model_name}_langgfm_i.log"
    
    lora_modules = " ".join([
        f"'{{\"name\": \"{dataset}\", \"path\": \"{model_path}\", \"base_model_name\": \"{hg_model}\"}}'"
        for dataset, model_path in best_models.items()
    ])
    
    vllm_command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} nohup vllm serve {hg_model} "
        f"--enable-lora --lora-modules {lora_modules} "
        f"--api-key 12345 --host 0.0.0.0 --port {port} "
        f"--max-model-len 16000 --max-lora-rank 64 > {log_file} 2>&1 &"
    )
    
    print(f"🚀 Initiating vLLM Server for {model_name} on GPU {gpu_id} (Port {port})...")
    process = subprocess.Popen(vllm_command, shell=True, stdout=subprocess.PIPE, executable="/bin/bash")
    print(f"✅ Server started, logs are being written to `{log_file}`.")
    
    time.sleep(120)  # Allow the server to fully start
    if check_vllm_server(port=port):
        print("✅ Server is running and ready for inference.")
    else:
        print("❌ Server failed to start. Exiting...")
        return False
    return True


def run_inference(datasets, port):
    """Runs inference for each dataset."""
    for dataset in datasets:
        inference_command = (
            f"python scripts/eval_langgfm_api.py "
            f"--api_key 12345 --port {port} "
            f"--file_path experiments/langgfm_i/{dataset}/test/instruction_dataset.json "
            f"--model_name {dataset} "
        )
        
        print(f"🔍 Running inference for {dataset}...")
        inference_process = subprocess.run(inference_command, shell=True, check=True)
        
        if inference_process.returncode == 0:
            print(f"✅ Inference completed successfully for {dataset}.")
        else:
            print(f"❌ Inference failed for {dataset}.")


def shutdown_vllm_server(hg_model):
    """Shuts down the vLLM server."""
    print("🛑 Shutting down the vLLM server...")
    os.system(f"pkill -f 'vllm serve {hg_model}'")
    time.sleep(3)
    print("✅ Server shut down successfully.")


def main(model_name):
    """Runs the pipeline for the given model on a specific GPU."""
    datasets = [
        "node_counting", "edge_counting", "node_attribute_retrieval",
        "edge_attribute_retrieval", "degree_counting", "shortest_path",
        "cycle_checking", "hamilton_path", "graph_automorphic",
        "graph_structure_detection", "edge_existence", "connectivity"
    ]
    
    gpu_ports = {
        "Llama-3.1-8B-Instruct": {"gpu_id": "0", "port": 8018, "hg_model":"meta-llama/Llama-3.1-8B-Instruct"},
        "Qwen2.5-7B-Instruct": {"gpu_id": "0", "port": 8017, "hg_model":"Qwen/Qwen2.5-7B-Instruct"},
    }
    
    if model_name not in gpu_ports:
        print(f"❌ Unsupported model: {model_name}. Choose from {list(gpu_ports.keys())}")
        return
    
    gpu_id = gpu_ports[model_name]["gpu_id"]
    port = gpu_ports[model_name]["port"]
    hg_model = gpu_ports[model_name]["hg_model"]
    
    best_models = get_all_best_models(datasets, model_name)
    if not best_models:
        print("❌ No valid best models found. Exiting...")
        return
    
    if start_vllm_servers(best_models, model_name, gpu_id, port, hg_model):
        # return "Success"
        run_inference(datasets, port)
        shutdown_vllm_server(hg_model)
    
    print(f"🎉 Pipeline completed for Model: {model_name} | GPU: {gpu_id} | Port: {port}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline_script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
