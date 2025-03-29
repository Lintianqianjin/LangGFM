# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

# python scripts/rl/data_preprocess/gen_shortest_path.py --train_dir experiments/langgfm_i/shortest_path/train/instruction_dataset.json --test_dir experiments/langgfm_i/shortest_path/test/instruction_dataset.json --output_dir data/verl/shortest_path
import re
import os
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    # solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    # assert solution is not None
    # final_solution = solution.group(0)
    # final_solution = final_solution.split('#### ')[1].replace(',', '')
    # return final_solution
    
    # extract [21, 10] in "The shortest paths from node 21 to node 10 are as follows: [21, 10]."
    # note the list can be any length, for example [21, 10, 11]
    solution = re.search(r'\[(.*?)\]', solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    # final_solution = final_solution.split('[')[1].split(']')[0].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=None)
    parser.add_argument('--test_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'langgfm/shortest_path'

    # load from sft dataset (json file), e.g., 
    # train set 
    train_data = json.load(open(args.train_dir, 'r'))
    # test set
    test_data = json.load(open(args.test_dir, 'r'))
    
    # convert to datasets.Dataset
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)
    
    
    # instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            _inst = example.pop('instruction')
            _input = example.pop('input')
            
            _inst = _inst + "- You should first thinks about the reasoning process in the mind" + \
            " and then provides the user with the answer. The reasoning process and answer are enclosed " + \
            "within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \n"
            
            example.pop("dataset")
            example.pop("task_type")

            user_content = _inst + '\n' + _input

            answer_raw = example.pop('output')
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": user_content,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": example['metadata']['main_task']['query'],
                    "graph_format": example.pop('graph_format'),
                    "metadata": example.pop('metadata'),
                    "tokens": example.pop('#tokens'),
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # output_dir = args.output_dir
    # hdfs_dir = args.hdfs_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.output_dir
    hdfs_dir = args.hdfs_dir

    
    # if output_dir not exist create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # save to parquet
    train_dataset.to_parquet(os.path.join(output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(output_dir, 'test.parquet'))

    # save to json too
    train_dataset.to_json(os.path.join(output_dir, 'train.json'), indent=4)
    test_dataset.to_json(os.path.join(output_dir, 'test.json'), indent=4)
    
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=output_dir, dst=hdfs_dir)
