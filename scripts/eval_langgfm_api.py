import os
import fire
import json
from tqdm import tqdm
from openai import OpenAI

from eval_utils import extract_info, compute_metric

# LANUCH vLLM SERVER FIRST
# nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype auto --api-key 12345 &> server.log &

# 设置 OpenAI 兼容的 API Key 和 vLLM 基础 URL
openai_api_key = "12345"
openai_api_base = "http://localhost:8016/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def load_dataset(file_path: str):
    """
    加载数据集文件，要求文件为 JSON 格式，内容为包含若干字典的列表，
    每个字典中至少包含以下字段：
      - "instruction": 指令文本，用于生成预测 prompt 的前半部分
      - "input": 输入文本，用于生成预测 prompt 的后半部分
      - "output": 标注答案（ground truth label）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def query_vllm(instruction, user: str, model_name: str):
    """
    向 vLLM 服务发送 prompt，并返回模型响应内容。
    若出现异常，则返回字符串 "Error"。
    """
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            # messages=[{"role": "system", "content": instruction},{"role": "user", "content": user}],
            messages=[{"role": "system", "content": ""},{"role": "user", "content": instruction + "\n" + user}],
            temperature=0,  # 保证输出确定性
            max_tokens=32   # 限制输出 token 数量，防止返回冗余文本
        )
        return chat_response.choices[0].message.content.strip()  # 提取响应内容
    except Exception as e:
        print(f"Error querying vLLM: {e}")
        return "Error"

def run_inference(file_path: str, model_name: str):
    """
    对数据集中的每个样本依次进行：
      1. 拼接 "instruction" 和 "input" 构成生成预测的 prompt，调用模型获得预测结果；
      2. 使用生成的预测结果与 ground truth（output）构造验证 prompt，
         调用模型获得验证结果（应为 True 或 False）；
      3. 记录每个样本的预测、验证结果，并统计正确数目，最终计算准确率。
    """
    samples = load_dataset(file_path)
    # total = len(samples)
    # correct_count = 0

    preds = []
    labels = []
    
    for entry in tqdm(samples, desc="Processing samples"):
        # 1. 生成预测：构造初始 prompt（注意这里的格式可根据你的任务需求调整）
        # initial_prompt = entry["instruction"] + entry["input"]
        prediction = query_vllm(entry["instruction"], entry["input"], model_name)
        entry["prediction"] = prediction  # 保存预测结果
        
        entry["predicted_answer"] = extract_info(entry.get('dataset', ""), prediction)  # Extracted direct answer
        entry["answer"] = extract_info(entry.get('dataset', ""), entry['output'])  # Extracted label from prediction
        preds.append(entry["predicted_answer"])
        labels.append(entry["answer"])
        
        # Debug information: Print each sample's prediction, ground truth, and verification result.
        # print(f"Prediction: {prediction}")
        print(f"Ground Truth: {entry['output']}")
        print(f"predicted_answer: {entry['predicted_answer']}")
        print(f"answer: {entry['answer']}")
        # print(f"Verification: {verdict}")
        print("-" * 50)

    # Save the output file with the original file name plus a suffix _with_prediction under the folder ckpts/openllm/{model_name}
    output_dir = os.path.join(
        os.path.dirname(file_path), 
        "ckpts",
        "langgfm-i",
        model_name
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/{os.path.basename(file_path).split('.')[0]}_with_prediction.json"
    save_results(output_file, samples)
    
    metric = compute_metric(entry.get('dataset', ""), preds, labels)
    
    save_results(f"{output_dir}/metric.json", metric)

def save_results(file_path: str, data):
    """
    将更新后的数据（包含预测和验证结果）保存到指定的 JSON 文件中。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(run_inference)
