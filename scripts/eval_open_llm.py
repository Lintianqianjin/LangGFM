import fire
import json
import os
from tqdm import tqdm
from openai import OpenAI

# LANUCH vLLM SERVER FIRST
# nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype auto --api-key 12345 &> server.log &

# Set OpenAI-compatible API key and base URL for vLLM
openai_api_key = "12345"
openai_api_base = "http://localhost:8016/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def construct_prompt(prediction: str, label: str) -> str:
    """Constructs the full prompt based on the given template."""
    prompt_template = f"""
You are a strict text classification validator. Please verify the prediction results according to the following rules:

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

def load_predictions(file_path: str):
    """Loads a JSONL file and returns a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_predictions(file_path: str, data):
    """Saves the updated data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def query_vllm(prompt: str, model_name: str):
    """Sends a prompt to the vLLM server and retrieves the model response."""
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0,  # Ensures deterministic output
            max_tokens=5  # Limits the output to prevent unnecessary text
        )
        return chat_response.choices[0].message.content.strip()  # Extract the response
    except Exception as e:
        print(f"Error querying vLLM: {e}")
        return "Error"

def run_verification(file_path: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
    """Runs vLLM to verify prediction results."""
    # Load prediction data
    data = load_predictions(file_path)

    correct_count = 0
    total_count = len(data)

    # Process each prediction
    for entry in tqdm(data, desc="Processing Predictions"):
        prediction = entry["predict"]
        label = entry["label"]

        # Construct the full prompt
        prompt = construct_prompt(prediction, label)

        # Get response from vLLM model
        judgement = query_vllm(prompt, model_name)

        # Store the judgement result
        entry["judgement"] = judgement
        if judgement.lower() == "true":
            correct_count += 1

    # Construct the new filename
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}_with_judgement_{correct_count}_{total_count}{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # Save results
    save_predictions(new_file_path, data)
    print(f"Results saved to: {new_file_path}")

    return new_file_path

if __name__ == "__main__":
    fire.Fire(run_verification)
    # run_verification("experiments/langgfm_i/wikics/test/ckpts/Qwen2.5-72B-Instruct/predictions.json", model_name="meta-llama/Llama-3.3-70B-Instruct")
# Example usage
# run_verification("your_prediction_file.jsonl")
