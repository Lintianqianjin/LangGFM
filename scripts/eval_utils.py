import re
import os
import json
import numpy as np

from bert_score import score as bertscore
from openai import OpenAI
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score


def safe_string(value, default=""):
    if isinstance(value, str):
        return value
    return default

def safe_float(value, default=3.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
    
def init_client(api_key="12345", url="http://localhost:8016/v1"):
    return OpenAI(api_key=api_key, base_url=url)

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
        # E.g., The author with id 5 would likely describe research interests as Biomechanics, Tissue Engineering, and Acoustics.
        match = re.search(r'The author with id \d+ would likely describe research interests as (.*).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() == "fb15k237":
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The most likely relation between the entity with node id {target_src_node_idx} and entity with node id {target_dst_node_idx} is {ground_truth}.
        match = re.search(r'The most likely relation between the entity with node id \d+ and entity with node id \d+ is (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() == "twitch":
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The user with node id 6 is likely a {ground_truth}.
        match = re.search(r'The user with node id \d+ is likely a (.+).', text)
        if match:
            # Return the extracted research interests
            return 0. if match.group(1).strip() == "gaming content streamer" else 1.
    
    elif dataset.lower() in {'node_counting', 'edge_counting', 'node_attribute_retrieval', 'edge_attribute_retrieval',
        'degree_counting', 'edge_existence', 'connectivity', 'shortest_path', 'hamilton_path', 'cycle_checking',
        'graph_structure_detection', 'graph_automorphic'}:
        
        return text
    
    else:
        # Additional matching rules can be added for other datasets
        return None

def extract_answer(text, dataset=None, logprobs=None):
    """
    Extract the answer from the text.
    
    Parameters:
        text (str): The text to search for the answer.
        
    Returns:
        str: The extracted answer text.
    """
    if dataset in {"twitch"}:
        key_token_logprobs = logprobs[10].top_logprobs
        for token in key_token_logprobs:
            # print(token)
            if token.token == ' mature':
                prob = np.exp(token.logprob)
                break
        return prob
    
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

        predictions = [safe_float(x, default=3.0) for x in predictions]
        labels = list(map(float, labels))
        error = rmse(np.array(list(predictions)), np.array(list(labels)))
        return {"rmse": error}
    
    elif dataset == "oag_scholar_interest":
        # 使用 bert-score 计算文本相似度
        # 注意：确保 predictions 和 labels 都是字符串列表
        # 默认语言设为英语（lang="en"），可以根据需要调整
        predictions = [safe_string(x) for x in predictions]
        
        P, R, F1 = bertscore(predictions, labels, lang="en", verbose=True)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        for pred, label in zip(predictions, labels):
            print(f"\n{pred=}")
            print(f"{label=}\n")
            scores = scorer.score(label, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        avg_rouge1 = np.mean(rouge1_scores)
        avg_rouge2 = np.mean(rouge2_scores)
        avg_rougeL = np.mean(rougeL_scores)

        # 返回一个字典，其中包含 bert 的 F1 和三个 rouge 指标的平均值
        return {
            "bert_f1": F1.mean().item(),
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL
        }
    
    elif dataset in {"fb15k237"}:
        # computer exact match
        # safe conversion to string
        
        predictions = [safe_string(x) for x in predictions]
        
        correct = 0
        for pred, label in zip(predictions, labels):
            if pred == label:
                correct += 1
                # return accuracy
        return {"accuracy": correct / len(labels)}
        
    elif dataset in {"twitch"}:
        # compute roc auc
        auc = roc_auc_score(labels, predictions)
        return {"roc_auc": auc}
    
    # 其他数据集的处理逻辑可以在这里添加
    elif dataset in {'node_counting', 'edge_counting', 'node_attribute_retrieval', 'edge_attribute_retrieval',
        'degree_counting', 'edge_existence', 'connectivity', 'shortest_path', 'hamilton_path', 'cycle_checking',
        'graph_structure_detection', 'graph_automorphic'}:
        # 直接返回预测和标签的匹配率
        correct_count = sum(pred == label for pred, label in zip(predictions, labels))
        accuracy = correct_count / len(labels)
        return {"accuracy": accuracy}

    return 0.0


def get_best_trained_model(dataset, model_name):
    """read trainer_state.json and get the best trained model path"""
    trainer_state_path = os.path.expanduser(
        # "~/pyan1/projects/LangGFM/experiments/langgfm_i/"
        "experiments/langgfm_i/"
        f"{dataset}"
        "/train/ckpts/"
        f"{model_name}"
        "/lora_rank=64/lora_alpha=256/lora_dropout=0.0/learning_rate=2e-05/num_train_epochs=20/warmup_ratio=0.5/batch_size=64/"
        "trainer_state.json")
    trainer_state_path_2 = os.path.expanduser(
        # "~/pyan1/projects/LangGFM/experiments/langgfm_i/"
        "experiments/langgfm_i/"
        f"{dataset}"
        "/train/ckpts/"
        f"{model_name}"
        "/lora_rank=64/lora_alpha=256/lora_dropout=0.0/learning_rate=2e-05/num_train_epochs=50/warmup_ratio=0.2/batch_size=64/"
        "trainer_state.json")
    # print(trainer_state_path)
    if os.path.exists(trainer_state_path):
        trainer_state_path = trainer_state_path
    elif os.path.exists(trainer_state_path_2):
        trainer_state_path = trainer_state_path_2
    try:
        # return trainer_state_path
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        best_model_path = trainer_state["best_model_checkpoint"]
        return best_model_path
    except Exception as e:
        print(f"!!! Error reading trainer_state.json: {e}")
        return None
    

if __name__ == "__main__":
    # Test the functions with sample data
    # import json
    
    dataset = "oag_scholar_interest"
    # DeepSeek-R1-Distill-Llama-70B
    # Qwen2.5-72B-Instruct
    # Llama-3.3-70B-Instruct
    data = json.load(open("experiments/langgfm_i/oag_scholar_interest/test_200/ckpts/langgfm-i/meta-llama/Llama-3.1-8B-Instruct-LangGFM-I/instruction_dataset_with_prediction.json","r"))
    # data = json.load(open("experiments/langgfm_i/oag_scholar_interest/test_200/ckpts/openllm/Llama-3.3-70B-Instruct/instruction_dataset_with_prediction.json","r"))
    predictions = []
    labels = []
    for entry in data:
        predictions.append(entry["predicted_answer"])
        labels.append(entry["answer"])
    metrics = compute_metric(dataset, predictions, labels)
    
    # print(metrics)
    for model in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]:
        for dataset in ['node_counting', 'edge_counting', 'node_attribute_retrieval', 'edge_attribute_retrieval',
        'degree_counting', 'edge_existence', 'connectivity', 'shortest_path', 'hamilton_path', 'cycle_checking',
        'graph_structure_detection', 'graph_automorphic']:
            best_model = get_best_trained_model(dataset, model)
            # data
            # test_path = f"~/projects/LangGFM/experiments/langgfm_i/{dataset}/test/instruction_dataset.json"
            # os.system(f"bash ~/utils/rsync_auto.sh {test_path} jyu7")
            os.system(f"bash ~/utils/rsync_auto.sh {best_model} jyu7")
