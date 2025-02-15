import re
import os
import json
import numpy as np

from bert_score import score as bertscore
from openai import OpenAI
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def safe_string(value, default=""):
    if isinstance(value, str):
        return value
    return default

def safe_float(value, default=3.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def softmax(x):
    x_exp = np.exp(x - np.max(x))  # 防止溢出
    return x_exp / np.sum(x_exp)

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

def locate_target_token_for_openllm(logprobs, dataset, model_name):
    if dataset == "twitch":
        if "qwen2.5-72b-instruct" in model_name.lower():
            # first token '<', second token 'answer', third token '>'
            # 4th token is the target token, mature or gamining
            # print(f"{logprobs=}")
            print(f"key token prediction: {logprobs[3].token}")
            target_token_top_logprobs = logprobs[3].top_logprobs
            # logprobs, yes_token, no_token
            return target_token_top_logprobs, " mature", " gaming"
        elif "llama-3.3-70b-instruct" in model_name.lower():
            # first token '<', second token 'answer', third token '>'
            # 4th token is the target token, mature or gamining
            # print(f"{logprobs[:10]=}")
            # exit()
            print(f"key token prediction: {logprobs[4].token}")
            target_token_top_logprobs = logprobs[4].top_logprobs
            # logprobs, yes_token, no_token
            return target_token_top_logprobs, " mature", " gaming"
        elif "langgfm" in model_name.lower():
            print(f"key token prediction: {logprobs[10].token}")
            target_token_top_logprobs = logprobs[10].top_logprobs
            return target_token_top_logprobs, " mature", " gaming"
    elif dataset == "ogbl_vessel":
        if "qwen2.5-72b-instruct" in model_name.lower():
            # first token '<', second token 'answer', third token '>'
            # 4th token is the target token, True or False
            print(f"key token prediction: {logprobs[3].token}")
            target_token_top_logprobs = logprobs[3].top_logprobs
            # logprobs, yes_token, no_token
            return target_token_top_logprobs, "True", "False"
        elif "llama-3.3-70b" in model_name.lower():
            print(f"key token prediction: {logprobs[3].token}")
            target_token_top_logprobs = logprobs[3].top_logprobs
            # logprobs, yes_token, no_token
            return target_token_top_logprobs, "True", "False"
        elif "langgfm" in model_name.lower():
            key_token_logprobs = logprobs[0].top_logprobs
            return target_token_top_logprobs, "Yes", "No"
    elif dataset == "stack_elec":
        if "qwen2.5-72b-instruct" in model_name.lower():
            # first token '<', second token 'answer', third token '>'
            # print(logprobs[:10])
            # exit()
            print(f"key token prediction: {logprobs[4].token}")
            target_token_top_logprobs = logprobs[4].top_logprobs
            return target_token_top_logprobs, " useful", " useless"
        elif "llama-3.3-70b-instruct" in model_name.lower():
            # print(logprobs[:10])
            # exit()
            print(f"key token prediction: {logprobs[4].token}")
            target_token_top_logprobs = logprobs[4].top_logprobs
            return target_token_top_logprobs, "ful", "less"
        if "langgfm" in model_name.lower() and "llama" in model_name.lower():
            # The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} is likely to be classified as 'Useful'.
            # -1 token "", -2 token "'.", -3 token "ful" or "less", -4 token "Use" -5 token " '"
            print(f"key token prediction: {logprobs[-3].token}")
            target_token_top_logprobs = logprobs[-3].top_logprobs
            return target_token_top_logprobs, "ful", "less"
        elif "langgfm" in model_name.lower() and "qwen" in model_name.lower():
            # The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} is likely to be classified as 'Useful'.
            # -1 token "", -2 token "'.", -3 token "ful" or "less", -4 token "Use" -5 token " '"
            print(f"key token prediction: {logprobs[-3].token}")
            target_token_top_logprobs = logprobs[-3].top_logprobs
            return target_token_top_logprobs, "ful", "less"
    elif dataset == "bace":
        if "langgfm" not in model_name.lower() and "qwen2.5" in model_name.lower():
            # first token '<', second token 'answer', third token '>'
            # 4th token is the target token, yes or no
            # print(f"{logprobs[3]=}")
            print(f"key token prediction: {logprobs[3].token}")
            target_token_top_logprobs = logprobs[3].top_logprobs
            # logprobs, yes_token, no_token
            return target_token_top_logprobs, "yes", "no"
        elif "langgfm" not in model_name.lower() and "llama-3.3-70b" in model_name.lower():
            # first token '<', second token 'answer', third token '>'
            # 4th token is the target token, yes or no
            # print(f"{logprobs[3]=}")
            print(f"key token prediction: {logprobs[3].token}")
            target_token_top_logprobs = logprobs[3].top_logprobs
            # logprobs, yes_token, no_token
            return target_token_top_logprobs, "yes", "no"
        elif "langgfm" in model_name.lower() and "llama" in model_name.lower():
            print(f"key token prediction: {logprobs[0].token}")
            target_token_top_logprobs = logprobs[0].top_logprobs
            return target_token_top_logprobs, "Yes", "No"
        elif "langgfm" in model_name.lower() and "qwen" in model_name.lower():
            print(f"key token prediction: {logprobs[0].token}")
            target_token_top_logprobs = logprobs[0].top_logprobs
            return target_token_top_logprobs, "Yes", "No"
            
def compute_prob_for_auc(key_token_logprobs, yes_token, no_token):
    prob = np.zeros(2) + key_token_logprobs[-1].logprob 
    for token in key_token_logprobs:
        if token.token == yes_token:
            yes_token_logit = token.logprob
            prob[1] = yes_token_logit
            print(f"Found {yes_token=} {yes_token_logit=}")
        elif token.token == no_token:
            no_token_logit = token.logprob
            prob[0] = no_token_logit
            print(f"Found {no_token=} {no_token_logit=}")
    # softmax
    prob = softmax(prob)
    prob = prob[1]
    return prob

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
    if dataset.lower() == "re_europe":
        # Use regex to search for {avg_load} in the format , avg_load is a float number
        # e.g., around {avg_load} MW.
        match = re.search(r'around (\d+\.\d+) MW', text)
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
        
    elif dataset.lower() == "yelp_review":
        # Use regex to search for {review} in the following format
        # E.g., f"User with node id {target_src_node_idx} may leave a review for Business with node id {target_dst_node_idx} as follows: {review}"
        # Note special chars like "\n" may be present in the review
        match = re.search(r'User with node id \d+ may leave a review for Business with node id \d+ as follows: (.+)', text, re.DOTALL)
        if match:
            # Return the extracted review
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
            return 0 if match.group(1).strip() == "gaming content streamer" else 1
    
    elif dataset.lower() == "stack_elec":
        # Use regex to search for {ground_truth} in the following text format:
        # f"The edge from user with node id {target_src_node_idx} to question with node id {target_dst_node_idx} is likely to be classified as '{ground_truth}'.
        match = re.search(r"The edge from user with node id \d+ to question with node id \d+ is likely to be classified as '(.+)'.", text)
        if match:
            # Return the extracted research interests
            return 1 if match.group(1).strip() == "Useful" else 0
    
    elif dataset.lower() == "ogbl_vessel":
        # Yes, a vessel likely exists between the bifurcation point with node id 4 and the bifurcation point with node id 10.
        # No, a vessel likely does not exist between the bifurcation point with node id 7 and the bifurcation point with node id 3.
        # Yes -> 1, No -> 0
        match = re.search(r'Yes', text)
        if match:
            return 1
        else:
            return 0
        
    elif dataset.lower() == "fingerprint":
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The given fingerprint is likely to belong to {ground_truth}.
        match = re.search(r'The given fingerprint is likely to belong to (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    
    elif dataset.lower() == "esol":
        # Use regex to search for {label} in the following text format:
        # E.g., The $\log S$ of this compound is likely {label}.""
        match = re.search(r'The \$\\log S\$ of this compound is likely (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() == "bace":
        # Use regex to search for {ground_truth} in the following text format:
        # if label == "No":
        #     answer = "No, the given molecule is unlikely to inhibit BACE-1."
        # elif label == "Yes":
        #     answer = "Yes, the given molecule is likely to inhibit BACE-1."
        match = re.search(r'Yes', text)
        if match:
            return 1
        else:
            return 0
        
    elif dataset.lower() == "chebi20":
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., f"The given molecule can be described as: \"{ground_truth}\""
        match = re.search(r'The given molecule can be described as: "(.*)"', text, re.DOTALL)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    
    elif dataset.lower() in {'node_counting'}:
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., There are {ground_truth} nodes in the graph.
        match = re.search(r'There are (\d+) nodes in the graph.', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() in {'edge_counting'}:
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., There are {ground_truth} edges in the graph.
        match = re.search(r'There are (\d+) edges in the graph.', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() in {'node_attribute_retrieval'}:
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The value of attribute [weight] for node [node] is {ground_truth}.
        match = re.search(r'The value of attribute \[weight\] for node \[\d+\] is (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() in {'edge_attribute_retrieval'}:
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The value of attribute [weight] for edge [{node},{node}] is {ground_truth}.
        # The value of attribute [weight] for edge [3,10] is 14
        match = re.search(r'The value of attribute \[weight\] for edge \[\d+,\d+\] is (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() in {'degree_counting'}:
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The degree of node [2] is {ground_truth}.
        match = re.search(r'The degree of node \[\d+\] is (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
    elif dataset.lower() in {'shortest_path'}:
        # Use regex to search for {ground_truth} in the following text format:
        # E.g., The shortest paths from node {node} to node {node} are as follows: {ground_truth}.
        match = re.search(r'The shortest paths from node \d+ to node \d+ are as follows: (.+).', text)
        if match:
            # Return the extracted research interests
            return match.group(1).strip()
        
    elif dataset.lower() in {
        "ogbn_arxiv", "wikics",
        'node_counting', 'edge_counting', 'node_attribute_retrieval', 'edge_attribute_retrieval',
        'degree_counting', 'edge_existence', 'connectivity', 'shortest_path', 'hamilton_path', 'cycle_checking',
        'graph_structure_detection', 'graph_automorphic'}:
        
        return text
    
    else:
        # Additional matching rules can be added for other datasets
        return None

def extract_answer(text, dataset=None, logprobs=None, model_name=None):
    """
    Extract the answer from the text.
    
    Parameters:
        text (str): The text to search for the answer.
        
    Returns:
        str: The extracted answer text.
    """
    text_related_semantic_tasks = {"ogbn_arxiv", "re_europe", "fingerprint", "chebi20", "esol", "ogbn_arxiv", "wikics", "oag_scholar_interest", "yelp_review"}
    structure_tasks = {"node_counting", "edge_counting", "node_attribute_retrieval", "edge_attribute_retrieval","degree_counting", "edge_existence", "connectivity", "shortest_path", "hamilton_path", "cycle_checking","graph_structure_detection","graph_automorphic"}
    # AUC
    if dataset in {"twitch", "ogbl_vessel", "stack_elec","bace"}:
        key_token_logprobs, yes_token, no_token = locate_target_token_for_openllm(logprobs, dataset, model_name)
        prob = compute_prob_for_auc(key_token_logprobs, yes_token, no_token)
        return prob

    # elif dataset in {"ogbl_vessel"}:
    #     key_token_logprobs, yes_token, no_token = locate_target_token_for_openllm(logprobs, "ogbl_vessel", model_name)
    #     prob = compute_prob_for_auc(key_token_logprobs, yes_token, no_token)
    #     return prob

    # elif dataset in {"stack_elec"}:
    #     key_token_logprobs, yes_token, no_token = locate_target_token_for_openllm(logprobs, "stack_elec", model_name)
    #     prob = compute_prob_for_auc(key_token_logprobs, yes_token, no_token)
    #     return prob

    # elif dataset in {"bace"}:
    #     key_token_logprobs, yes_token, no_token = locate_target_token_for_openllm(logprobs, "bace", model_name)
    #     prob = compute_prob_for_auc(key_token_logprobs, yes_token, no_token)
    #     return prob
    
    # no special processing, model after sft should return the answer in the as same format as the label
    elif "langgfm" in model_name.lower() and dataset in text_related_semantic_tasks | structure_tasks: 
        return extract_info(dataset, text)
    
    else: # open llm & no need to process logit
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

    if dataset in {"esol",}:
        # Convert predictions and labels to floats and compute RMSE
        print(f"{predictions=}")
        print(f"{labels=}")
        if dataset == "movielens1m":
            default = 3.0
        elif dataset == "esol":
            default = 0.0
        predictions = [safe_float(x, default=default) for x in predictions]
        labels = list(map(float, labels))
        error = rmse(np.array(list(predictions)), np.array(list(labels)))
        return {"rmse": error}
    
    elif dataset in {"movielens1m", "re_europe"}:
        if dataset == "re_europe":
            default = 0.0
        predictions = [safe_float(x, default=default) for x in predictions]
        labels = list(map(float, labels))
        # compute spearman correlation
        corr, p_value = spearmanr(predictions, labels)
        return {"spearmanr": corr}
    
    elif dataset in {"oag_scholar_interest", "yelp_review", "chebi20"}:
        # 使用 bert-score 计算文本相似度
        # 注意：确保 predictions 和 labels 都是字符串列表
        # 默认语言设为英语（lang="en"），可以根据需要调整
        predictions = [safe_string(x) for x in predictions]
        
        # P, R, F1 = bertscore(predictions, labels, lang="en", verbose=True)
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
        metrics = {
            # "bert_f1": F1.mean().item(),
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL
        }
        print(metrics)
        # 返回一个字典，其中包含 bert 的 F1 和三个 rouge 指标的平均值
        return metrics
    
    elif dataset in {"ogbn_arxiv", "fb15k237", "fingerprint"}:
        # computer exact match
        # safe conversion to string
        
        predictions = [safe_string(x) for x in predictions]
        
        correct = 0
        for pred, label in zip(predictions, labels):
            if pred == label:
                correct += 1
                # return accuracy
        metrics = {"accuracy": correct / len(labels)}
        print(metrics)
        return metrics
        
    elif dataset in {"twitch", "ogbl_vessel", "stack_elec","bace"}:
        # compute roc auc
        auc = roc_auc_score(labels, predictions)
        return {"roc_auc": auc}
    
    # 其他数据集的处理逻辑可以在这里添加
    elif dataset in {"bace",'node_counting', 'edge_counting', 'node_attribute_retrieval', 'edge_attribute_retrieval',
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
    
    dataset = "re_europe"
    # DeepSeek-R1-Distill-Llama-70B
    # Qwen2.5-72B-Instruct
    # Llama-3.3-70B-Instruct
    data = json.load(open("experiments/langgfm_i/movielens1m/test_200/ckpts/langgfm-i/Qwen/Qwen2.5-7B-Instruct-LangGFM/instruction_dataset_with_prediction.json","r"))
    # data = json.load(open("experiments/langgfm_i/oag_scholar_interest/test_200/ckpts/openllm/Llama-3.3-70B-Instruct/instruction_dataset_with_prediction.json","r"))
    predictions = []
    labels = []
    for entry in data:
        predictions.append(entry["predicted_answer"])
        labels.append(entry["answer"])
    metrics = compute_metric(dataset, predictions, labels)
    print(metrics)
