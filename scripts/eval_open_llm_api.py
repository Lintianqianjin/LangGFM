import os
import fire
import json
from tqdm import tqdm
from openai import OpenAI
from eval_utils import extract_info, extract_answer, compute_metric, init_client

# LAUNCH vLLM SERVER FIRST
# Example: nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype auto --api-key 12345 &> server.log &

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

def query_vllm(client, prompt: str, model_name: str):
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

def run_inference(file_path: str, model_name: str,api_key="12345", port=8016):
    """
    For each sample in the dataset, perform the following steps:
      1. Generate a prediction by concatenating the "instruction" and "input" to form a prompt, 
         and query the model for the prediction result.
      2. (Optional, commented out) Use the prediction and the ground truth ("output") to construct 
         a verification prompt, then query the model for a verification result (expected to be True or False).
      3. Record the prediction and verification results for each sample, and compute the overall metric.
    """
    
    client = init_client(api_key, port)
    
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
        prediction = query_vllm(client, initial_prompt, model_name)
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
    output_dir = os.path.join(
        os.path.dirname(file_path), 
        "ckpts",
        "openllm",
        model_name.split("/")[1]
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/{os.path.basename(file_path).split('.')[0]}_with_prediction.json"
    save_results(output_file, samples)
    
    metric = compute_metric(entry.get('dataset', ""), preds, labels)
    
    save_results(f"{output_dir}/metric.json", metric)

def save_results(file_path: str, data):
    """
    Save the updated data (including predictions and verification results) to the specified JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(run_inference)
