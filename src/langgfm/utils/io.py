import os
import yaml
import json
import pandas as pd


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)
    


def load_jsonl(file_path, return_type="list"):
    """
    Load a JSONL (JSON Lines) file.

    Parameters:
        file_path (str): Path to the JSONL file.
        return_type (str): "list" for a list of dictionaries, "dataframe" for a pandas DataFrame.

    Returns:
        list or pandas.DataFrame: The loaded data in the specified format.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if return_type == "dataframe":
        return pd.DataFrame(data)
    return data  # Default to list

# 示例用法：
# jsonl_data = load_jsonl("data.jsonl", return_type="list")  # 以列表形式返回
# df_data = load_jsonl("data.jsonl", return_type="dataframe")  # 以DataFrame形式返回
