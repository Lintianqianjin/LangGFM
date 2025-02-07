#!/usr/bin/env python3
import os
import shutil

def copy_files(dataset):
    # 定义源目录和目标目录
    source_train_dir = os.path.join(dataset, "train_800")
    source_test_dir = os.path.join(dataset, "test_200")
    dest_train_dir = os.path.join("langgfm_i", dataset, "train")
    dest_test_dir = os.path.join("langgfm_i", dataset, "test")

    # 需要复制的文件列表
    files_to_copy = ["data_generation.yaml", "indices.json"]

    # 如果目标目录不存在，则创建之
    os.makedirs(dest_train_dir, exist_ok=True)
    os.makedirs(dest_test_dir, exist_ok=True)

    # 从 train_800 复制文件到 langgfm_i/[dataset]/train
    for file_name in files_to_copy:
        src_file = os.path.join(source_train_dir, file_name)
        dst_file = os.path.join(dest_train_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} -> {dst_file}")
        else:
            print(f"Source file does not exist: {src_file}")

    # 从 test_200 复制文件到 langgfm_i/[dataset]/test
    for file_name in files_to_copy:
        src_file = os.path.join(source_test_dir, file_name)
        dst_file = os.path.join(dest_test_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} -> {dst_file}")
        else:
            print(f"Source file does not exist: {src_file}")

if __name__ == '__main__':
    os.chdir("./../experiments")
    # 修改此处的 dataset 名称为你的实际数据集目录名称
    exps = ['node_counting', 'edge_counting', 'node_attribute_retrieval', 'edge_attribute_retrieval',
        'degree_counting', 'edge_existence', 'connectivity', 'shortest_path', 'hamilton_path', 'cycle_checking',
        'graph_structure_detection', 'graph_automorphic']

    for dataset in exps:
        copy_files(dataset)

