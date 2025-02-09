import fire
import json
from tqdm import tqdm
from openai import OpenAI

# LANUCH vLLM SERVER FIRST
# nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype auto --api-key 12345 &> server.log &

# 设置 OpenAI 兼容的 API Key 和 vLLM 基础 URL
openai_api_key = "12345"
openai_api_base = "http://localhost:8016/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def construct_prompt(prediction: str, label: str) -> str:
    """
    构造验证 prompt。
    
    该 prompt 指示模型验证 Prediction 是否正确地预测了 Label 类别。
    模型需严格按照规则判断，并只返回 True 或 False。
    """
    prompt_template = f"""
You are a strict text classification validator. Please verify whether the prediction results correctly predict the given category according to the following rules:

# Task  
Verify whether [Prediction] correctly predicts the [Label] category.

# Verification Rules  
1. Exact Match Rule:  
   - The Prediction must contain the complete and accurate Label text.  
   - The Label text must exist as an independent semantic unit in the Prediction  
     (it cannot be part of another word).

2. Prediction Intent Rule:  
   - The Prediction must explicitly classify the Label as the predicted result.  
   - Merely mentioning the Label without using it as the predicted result should be considered incorrect.  
   - The following cases must be excluded:  
     * The Label appears as a counterexample.  
     * The Label appears as part of a hypothetical discussion.  
     * The Label appears as background information.

# Output Format  
Return `True` if the prediction is correct and `False` if it is incorrect. Only return `True` or `False` without any additional text.

Input:  
Prediction: {prediction}  
Label: {label}  

Output:
"""
    return prompt_template.strip()

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

def query_vllm(prompt: str, model_name: str):
    """
    向 vLLM 服务发送 prompt，并返回模型响应内容。
    若出现异常，则返回字符串 "Error"。
    """
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
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
    total = len(samples)
    correct_count = 0

    for entry in tqdm(samples, desc="Processing samples"):
        # 1. 生成预测：构造初始 prompt（注意这里的格式可根据你的任务需求调整）
        initial_prompt = entry["instruction"] + "<|eot_id|>" + entry["input"]
        prediction = query_vllm(initial_prompt, model_name)
        entry["prediction"] = prediction  # 保存预测结果

        # 2. 验证预测：构造验证 prompt，将预测结果与 ground truth 进行对比
        verification_prompt = construct_prompt(prediction, entry["output"])
        verdict = query_vllm(verification_prompt, model_name)
        entry["judgement"] = verdict  # 保存验证结果

        # 3. 根据验证结果判断是否正确（这里对返回结果统一转小写处理）
        if verdict.strip().lower() == "true":
            entry["is_correct"] = True
            correct_count += 1
        elif verdict.strip().lower() == "false":
            entry["is_correct"] = False
        else:
            # 若返回结果不符合预期，则视为验证失败，并输出提示信息
            print(f"Unexpected verifier output for sample: {verdict}")
            entry["is_correct"] = False

        # 调试信息：打印每个样本的预测、ground truth 和验证结果
        print(f"Prediction: {prediction}")
        print(f"Ground Truth: {entry['output']}")
        print(f"Verification: {verdict}")
        print("-" * 50)

    accuracy = correct_count / total if total > 0 else 0
    print(f"\nTotal Samples: {total}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")

    # 如果需要将带有预测和验证结果的数据保存到文件，可以取消下面的注释
    output_file = file_path.replace(".json", f"_with_judgement_{correct_count}_{total}.json")
    save_results(output_file, samples)

def save_results(file_path: str, data):
    """
    将更新后的数据（包含预测和验证结果）保存到指定的 JSON 文件中。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(run_inference)
