import os
import json
import math
from tqdm import tqdm
import fire
from transformers import pipeline

from eval_utils import (
    extract_info,
    extract_answer,
    compute_metric,
)

def query_llm_batch(pipe, prompts, max_new_tokens=128, batch_size=16):
    """
    Send a list of prompts to the pipeline in batches and return the model's responses.
    Each prompt is expected to receive a single response.

    :param pipe: The text-generation pipeline
    :param prompts: List of prompt strings
    :param max_length: Maximum output token length for text generation
    :param batch_size: Number of samples to process per batch
    :return: List of response strings
    """
    responses = []
    num_samples = len(prompts)
    num_batches = math.ceil(num_samples / batch_size)

    for i in tqdm(range(num_batches), desc="Batch Inference"):
        start = i * batch_size
        end = min(start + batch_size, num_samples)
        current_batch = prompts[start:end]

        try:
            # 'pipe' returns a list of lists of dicts for text-generation;
            # we only expect one item per prompt, so we'll take [0]['generated_text'] each time.
            batch_outputs = pipe(
                current_batch,
                max_new_tokens=max_new_tokens,
                early_stopping=True,     # Force stopping at max_new_tokens
                do_sample=False,         # Greedy decoding (set True & tune top_k, top_p, temperature, etc.)
                num_return_sequences=1
            )
            
            for out in batch_outputs:
                # out is typically a list of dicts: [{'generated_text': '...'}]
                # We only need the 'generated_text' from the first item
                print(out[0]["generated_text"])
                responses.append(out[0]["generated_text"])
        except Exception as e:
            print(f"Error querying pipeline in batch: {e}")
            # Fill in the batch with "Error" strings so indexes remain aligned
            for _ in current_batch:
                responses.append("Error")

    return responses

def run_inference(file_path: str, model_name="GraphWiz/LLaMA2-7B-DPO", batch_size=32, max_new_tokens=32):
    
    pipe = pipeline("text-generation", model=model_name)
    samples = json.load(open(file_path, "r", encoding="utf-8"))
    
    # 1. Build all prompts first
    prompts = []
    for entry in samples:
        initial_prompt = (
            # entry["instruction"]
            # + "<|eot_id|>"
            entry["input"]
            + "\n\nYour response **must** contain a direct, clear, and unambiguous answer "
            + "enclosed within the `<answer></answer>` tags.\n"
            # + "- You may perform reasoning and analysis **outside** the `<answer></answer>` tags.\n"
            # + "- The answer cannot be directly found in the input, you must infer the best possible estimate.\n"
            + "- The answer **must not** include explanations, qualifiers, or any extraneous text.\n"
            # + "- The enclosed answer **must** be valid for direct use in subsequent calculations of machine learning metrics such as accuracy, RMSE, ROUGE, etc.\n"
            # + "- Responses like 'unable to determine', 'cannot be inferred', or 'None' or any other ambiguous statements are strictly prohibited.\n"
        )
        alpaca_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n### Instruction:\n{query}\n\n### Response:"
        initial_prompt = alpaca_template.format(query=initial_prompt)
        prompts.append(initial_prompt)
    
    # 2. Call the pipeline in batches
    model_responses = query_llm_batch(
        pipe,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size
    )

    # 3. Parse responses and compute predictions
    preds, labels = [], []
    for entry, prediction in zip(samples, model_responses):
        entry["prediction"] = prediction  # Full prediction (with potential reasoning)
        entry["predicted_answer"] = extract_answer(prediction)  # Extracted direct answer
        # "answer" is the ground truth label; we parse it with your custom logic
        entry["answer"] = extract_info(entry.get("dataset", ""), entry["output"])
        
        preds.append(entry["predicted_answer"])
        labels.append(entry["answer"])
        
        print(f"Ground Truth: {entry['output']}")
        print(f"answer: {entry['answer']}")
        print("\n")
        print(f"prediction: {entry['prediction']}")
        print(f"predicted_answer: {entry['predicted_answer']}")
        print("-" * 50)
    
    # 4. Save predictions
    output_dir = os.path.join(
        os.path.dirname(file_path), 
        "ckpts",
        "baselines",
        model_name.split("/")[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/{os.path.basename(file_path).split('.')[0]}_with_prediction.json"
    save_results(output_file, samples)

    # 5. Compute metrics
    metric = compute_metric(samples[0].get("dataset", ""), preds, labels)
    save_results(f"{output_dir}/metric.json", metric)

def save_results(file_path: str, data):
    """
    Save the updated data (including predictions and verification results) to the specified JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(run_inference)
