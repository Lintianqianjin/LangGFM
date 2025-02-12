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
    timeout = 120
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


def pipeline(dataset, model_name, gpu_id, port, hg_model):
    """Runs the vLLM server, performs inference, and shuts down the server."""

    print(f"\n🚀 Starting Pipeline for Dataset: {dataset} | Model: {model_name} | GPU: {gpu_id} | Port: {port}")

    # Get the best trained model path
    best_model = get_best_trained_model(dataset, model_name)
    if not best_model:
        print(f"⚠️ No best model found for dataset: {dataset}, model: {model_name}. Skipping...")
        return
    # drop the user path
    best_model = best_model.split("LangGFM/")[-1]
    print(f"✅ Best model: {best_model}")

    # Log file for the server
    log_file = f"server_{dataset}_{model_name}_langgfm_i.log"

    # Construct the vLLM server command
    vllm_command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} nohup vllm serve {hg_model} "
        f"--enable-lora --lora-modules '{{\"name\": \"{model_name}-LangGFM_I\", \"path\": \"{best_model}\", \"base_model_name\": \"{hg_model}\"}}' "
        f"--api-key 12345 --host 0.0.0.0 --port {port} "
        f"--max-model-len 16000 --max-lora-rank 64 > {log_file} 2>&1 &"
    )

    print(f"🚀 Initiating vLLM Server for {model_name}-LangGFM_I on GPU {gpu_id} (Port {port})...")
    # os.system(vllm_command)
    process = subprocess.Popen(vllm_command, shell=True, stdout=subprocess.PIPE, executable="/bin/bash")
    # pid = process.stdout.read().strip().decode()
    print(f"✅ Server started, logs are being written to `{log_file}`.")



    # Give the server time to start before running inference
    time.sleep(5)  # Adjust if needed

    # 检查服务器是否成功启动
    # if not check_vllm_server(port=port):  # Check corresponding port
    #     print("❌ Server failed to start. Exiting...")
    #     return

    if check_vllm_server(port=port):
        print("✅ Server is running and ready for inference.")
    else:
        print("❌ Server failed to start (TIMEOUT in check_vllm_server). Exiting...")
        return

    # Construct the inference command
    inference_command = (
        f"python scripts/eval_langgfm_api.py "
        f"--api_key 12345 --port {port} "
        f"--file_path experiments/langgfm_i/{dataset}/test/instruction_dataset.json "
        f"--model_name {model_name}-LangGFM_I "
    )

    print("🔍 Running inference...")
    inference_process = subprocess.run(inference_command, shell=True, check=True)

    if inference_process.returncode == 0:
        print("✅ Inference completed successfully.")
    else:
        print("❌ Inference failed.")

    # Shut down the vLLM server gracefully
    print("🛑 Shutting down the vLLM server...")
    os.system(f"pkill -f 'vllm serve {hg_model}'")
    # os.system(f"pkill -f vllm serve")  # Shut down only the specific instance
    # os.system(f"kill -9 {pid}")  # Shut down only the specific instance
    time.sleep(3)  # Ensure processes are fully terminated

    print(f"🎉 Pipeline completed for Dataset: {dataset} | Model: {model_name} | GPU: {gpu_id} | Port: {port}\n")


def main(model_name):
    """Runs the pipeline for the given model on a specific GPU."""

    datasets = [
        "node_counting", "edge_counting", "node_attribute_retrieval",
        "edge_attribute_retrieval", "degree_counting", "shortest_path",
        "cycle_checking", "hamilton_path", "graph_automorphic",
        "graph_structure_detection", "edge_existence", "connectivity"
    ]

    gpu_ports = {
        "Llama-3.1-8B-Instruct": {"gpu_id": "2", "port": 8018,"HG_MODEL":"meta-llama/Llama-3.1-8B-Instruct"},
        "Qwen2.5-7B-Instruct": {"gpu_id": "3", "port": 8017, "HG_MODEL":"Qwen/Qwen2.5-7B-Instruct"}, 
    }

    if model_name not in gpu_ports:
        print(f"❌ Unsupported model: {model_name}. Choose from {list(gpu_ports.keys())}")
        return

    gpu_id = gpu_ports[model_name]["gpu_id"]
    port = gpu_ports[model_name]["port"]
    hg_model = gpu_ports[model_name]["HG_MODEL"]

    for dataset in datasets:
        pipeline(dataset, model_name, gpu_id, port, hg_model)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline_script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    main(model_name)
