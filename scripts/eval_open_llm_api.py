import re
import os
import fire
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from bert_score import score as bertscore
# LAUNCH vLLM SERVER FIRST
# Example: nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype auto --api-key 12345 &> server.log &

# Set the OpenAI-compatible API key and the base URL for vLLM
openai_api_key = "12345"
openai_api_base = "http://localhost:8016/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def rmse(predictions, targets):
    """
    Compute the root mean squared error (RMSE) between predictions and targets.
    
    Parameters:
        predictions (numpy.ndarray): Array of predicted values.
        targets (numpy.ndarray): Array of ground truth values.
        
    Returns:
        float: The RMSE value.
    """
    # Compute the squared error for each sample, average them, then take the square root.
    return np.sqrt(((predictions - targets) ** 2).mean())

def extract_info(dataset, text):
    """
    Extract required information from the text based on the dataset name using regex.
    
    Parameters:
        dataset (str): Name of the dataset, e.g., "movielens1m".
        text (str): The text to search within.
        
    Returns:
        int or None: For the "movielens1m" dataset, returns the extracted rating (as an integer)
                     if found; otherwise, returns None.
    """
    if dataset.lower() == "movielens1m":
        # Use regex to search for rating information in the format "<number> stars"
        match = re.search(r'(\d+)\s*star', text)
        if match:
            # Return the extracted rating
            return match.group(1).strip()
        else:
            return None
    elif dataset.lower() == "oag_scholar_interest":
        # Use regex to search for research interests information in the format "<interest1>, <interest2>, ..."
        match = re.search(r'The author with id \d+ would likely describe research interests as (.*).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    else:
        # Additional matching rules can be added for other datasets
        return None

def extract_answer(text):
    """
    Extract the answer from the text.
    
    Parameters:
        text (str): The text to search for the answer.
        
    Returns:
        str: The extracted answer text.
    """
    # Use regex to search for answer information in the format "<answer>...</answer>"
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        # Return the extracted answer text
        return match.group(1).strip()
    else:
        return None

def compute_metric(dataset, predictions, labels):
    """
    Compute the similarity metric between the model predictions and the labels.
    
    Parameters:
        dataset (str): The dataset name.
        predictions (iterable): An iterable of predictions.
        labels (iterable): An iterable of ground truth labels.
        
    Returns:
        float: The similarity score between predictions and labels.
    """
    if dataset == "movielens1m":
        # Convert predictions and labels to floats and compute RMSE
        print(f"{predictions=}")
        print(f"{labels=}")
        def safe_float(value, default=3.0):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        predictions = [safe_float(x) for x in predictions]
        labels = list(map(float, labels))
        error = rmse(np.array(list(predictions)), np.array(list(labels)))
        return error
    
    elif dataset == "oag_scholar_interest":
        # 使用 bert-score 计算文本相似度
        # 注意：确保 predictions 和 labels 都是字符串列表
        # 默认语言设为英语（lang="en"），可以根据需要调整
        P, R, F1 = bertscore(predictions, labels, lang="en", verbose=True)
        # 返回 F1 分数的平均值作为整体得分
        return F1.mean().item()
    
    return 0.0

def load_dataset(file_path: str):
    """
    Load the dataset file. The file should be in JSON format and contain a list of dictionaries,
    where each dictionary must include at least the following fields:
      - "instruction": Instruction text used for generating the first part of the prediction prompt.
      - "input": Input text used for generating the second part of the prediction prompt.
      - "output": The ground truth answer.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def query_vllm(prompt: str, model_name: str):
    """
    Send the prompt to the vLLM service and return the model's response.
    If an exception occurs, return the string "Error".
    """
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Ensure deterministic output
            # max_tokens=32   # Limit the number of output tokens to avoid redundant text
        )
        # Extract and return the response content
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying vLLM: {e}")
        return "Error"

def run_inference(file_path: str, model_name: str):
    """
    For each sample in the dataset, perform the following steps:
      1. Generate a prediction by concatenating the "instruction" and "input" to form a prompt, 
         and query the model for the prediction result.
      2. (Optional, commented out) Use the prediction and the ground truth ("output") to construct 
         a verification prompt, then query the model for a verification result (expected to be True or False).
      3. Record the prediction and verification results for each sample, and compute the overall metric.
    """
    samples = load_dataset(file_path)
    # total = len(samples)
    # correct_count = 0

    preds = []
    labels = []
    
    for entry in tqdm(samples, desc="Processing samples"):
        # 1. Generate prediction: Construct the initial prompt (adjust the format as needed for your task)
        initial_prompt = entry["instruction"] + "<|eot_id|>" + entry["input"] + \
            "\n\nYour response **must** contain a direct, clear, and unambiguous answer "+\
            "enclosed within the `<answer></answer>` tags.\n" + \
            "- You may perform reasoning and analysis **outside** the `<answer></answer>` tags.\n" +\
            "- The answer cannot be directly found in the input, you must infer the best possible estimate.\n" +\
            "- The answer **must not** include explanations, qualifiers, or any extraneous text.\n" +\
            "- The enclosed answer **must** be valid for direct use in subsequent calculations of machine learning metrics such as accuracy, RMSE, ROUGE, etc.\n" +\
            "- Responses like 'unable to determine', 'cannot be inferred', or 'None' or any other ambiguous statements are strictly prohibited.\n"
        prediction = query_vllm(initial_prompt, model_name)
        entry["prediction"] = prediction  # Prediction with reasoning
        entry["predicted_answer"] = extract_answer(prediction)  # Extracted direct answer
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
    output_file = os.path.join(
        os.path.dirname(file_path), 
        "ckpts",
        "openllm",
        model_name.split("/")[1]
    )
    
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    
    output_file = f"{output_file}/{os.path.basename(file_path).split('.')[0]}_with_prediction.json"
    save_results(output_file, samples)
    
    metric = compute_metric(entry.get('dataset', ""), preds, labels)
    print(f"Metric: {metric:.4f}")

def save_results(file_path: str, data):
    """
    Save the updated data (including predictions and verification results) to the specified JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(run_inference)
