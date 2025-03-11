from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor, ComputeExactMatch, ComputeRegressionMetrics, ComputeAucMetrics, BinaryClassificationProbabilityCalculator


def contains_any(string, substrings):
    """
    Check if the input string contains any of the substrings from the list.
    
    :param string: The main string to check.
    :param substrings: A list of substrings to search for.
    :return: True if any substring is found in the main string, otherwise False.
    """
    return any(sub in string for sub in substrings)


def get_evaluation_settings(dataset_name, tokenizer):
    """
    Returns evaluation-related settings as a dictionary based on the dataset name.

    Parameters:
        dataset_name (str): The name of the dataset (e.g. "shortest_path", "movielens", "bace").
        tokenizer: A tokenizer object used to construct metric computation objects.

    Returns:
        dict: A dictionary containing the evaluation settings.
    """
    evaluation_settings = {}
    
    if "shortest_path" in dataset_name:
        return {
            "compute_metrics": ComputeExactMatch(tokenizer=tokenizer),
            "gen_kwargs": {
                "do_sample": False,
                "max_new_tokens": 64,
            },
        }
    elif contains_any(dataset_name, ["movielens1m", "zinc"]):
        return {
            "compute_metrics": ComputeRegressionMetrics(tokenizer=tokenizer),
            "gen_kwargs": {
                "do_sample": False,
                "max_new_tokens": 64,
            },
        }
    elif "bace" in dataset_name:
        return {
            "compute_metrics": ComputeAucMetrics(tokenizer=tokenizer),
            "preprocess_generated_output_logits_for_metrics": BinaryClassificationProbabilityCalculator(
                tokenizer=tokenizer,
                positive_token_text=" Yes",  # Note: a leading space is required
                negative_token_text=" No"     # Note: a leading space is required
            ),
            "gen_kwargs": {
                "do_sample": False,
                "max_new_tokens": 32,
                "output_logits": True,
                "return_dict_in_generate": True,
            },
        }
    
    return evaluation_settings
