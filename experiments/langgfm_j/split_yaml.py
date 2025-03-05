import os
import json
import yaml

def split_data_generation_and_indices(yaml_path="data_generation.yaml", indices_path="indices.json"):
    """
    将 data_generation.yaml 拆分到多个 train_<dataset> 文件夹，并将 indices.json 同步拆分。
    每个 train_<dataset> 文件夹下会产生:
        data_generation.yaml  (只包含公共配置 + 当前数据集配置)
        indices.json          (只包含当前数据集自己的索引列表)
    """

    # ---------- 1. 读取 data_generation.yaml ----------
    with open(yaml_path, 'r', encoding='utf-8') as f_yaml:
        data_yaml = yaml.safe_load(f_yaml)

    # ---------- 2. 读取 indices.json ----------
    with open(indices_path, 'r', encoding='utf-8') as f_json:
        data_indices = json.load(f_json)  # 例如 {datasetA: [...], datasetB: [...], ...}

    # ---------- 3. 区分公共锚点和具体数据集 ----------
    anchors = {}
    datasets = {}

    for key, value in data_yaml.items():
        # 约定：以 "common_" 开头的顶层 key 作为公共锚点
        if key.startswith("common_"):
            anchors[key] = value
        else:
            datasets[key] = value

    # ---------- 4. 为每个数据集创建 train_XXX 文件夹并写出对应的 data_generation.yaml 和 indices.json ----------
    for ds_name, ds_config in datasets.items():
        folder_name = f"{ds_name}"
        os.makedirs(folder_name, exist_ok=True)

        # 4.1 生成每个数据集自己的 data_generation.yaml
        out_dict = {}
        # 先写入公共 anchors
        for anchor_key, anchor_value in anchors.items():
            out_dict[anchor_key] = anchor_value
        # 再写入该数据集/任务的配置
        out_dict[ds_name] = ds_config

        output_yaml_path = os.path.join(folder_name, "data_generation.yaml")
        with open(output_yaml_path, 'w', encoding='utf-8') as out_f:
            yaml.dump(out_dict, out_f, allow_unicode=True, sort_keys=False)

        # 4.2 生成对应的 indices.json（只包含当前 ds_name 的键）
        ds_indices_dict = {}
        # if ds_name in data_indices:
        ds_indices_dict[ds_name] = data_indices[ds_name]
        # else:
        #     # 如果 indices.json 中没有该数据集，可以按需写空 {}
        #     ds_indices_dict = {}

        output_json_path = os.path.join(folder_name, "indices.json")
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(ds_indices_dict, out_f, ensure_ascii=False, indent=2)

    print("数据集配置和索引拆分完成！")

if __name__ == "__main__":
    split_data_generation_and_indices(
        yaml_path="data_generation.yaml",
        indices_path="indices.json"
    )
