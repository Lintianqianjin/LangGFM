import os
import psutil
import sys
import json
import yaml
from tqdm import tqdm  # tqdm
from typing import Dict, Any, List
import pandas as pd
import networkx as nx

from ..data.graph_generator._base_generator import InputGraphGenerator
from ..data.ssl_tasks.base_ssl import SelfSupervisedGraphTask
from ..data.ssl_tasks.tae_ssl import TopologyAutoencoder
from ..data.ssl_tasks.fmae_ssl import NodeFeatureMaskedAutoencoder, EdgeFeatureMaskedAutoencoder
from ..data.graph_text_transformation.nxg_to_text import GraphTextualizer
from ..configs.instruction_template import SYSTEM, INSTRUCTION, INPUT

from multiprocessing import Pool
from functools import partial
from ..utils.language_model import count_tokens_batch
from transformers import AutoTokenizer

from datasets import Dataset

import logging
logger = logging.getLogger("root")


class DatasetGenerationCoordinator:
    """
    Coordinator class to manage the generation of graphs and their textual representations.
    the recommended template for the prompt, from llamafactory, is as follows:
    [
        {
            "instruction": "human instruction (required)",
            "input": "human input (optional)",
            "output": "model response (required)",
            "system": "system prompt (optional)",
            "history": ["history of conversation (optional)"],
        }
    ]
    """
    def __init__(self, job_path: str = "./experiments/default_job", is_continue: bool = False, return_token_length: bool = False, tokenizer_name_or_path: str = None):
        self.root = job_path
        self.data_filepath = os.path.join(self.root, "instruction_dataset.json")
        self.textualizer = GraphTextualizer()   # Suppose this is imported
        self._load_config()
        self._load_indices()
        self.return_token_length = return_token_length
        if self.return_token_length:
            assert tokenizer_name_or_path is not None, "Please provide a tokenizer name or path."
            self.tokenizer_name_or_path = tokenizer_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)

        self.num_proc = psutil.cpu_count(logical=True)
        
        if not is_continue:
            with open(self.data_filepath, "w") as f:
                f.write("[]")

    def _load_config(self):
        "load generation needed config, from {self.root}/data_generation.yaml"
        with open(os.path.join(self.root, "data_generation.yaml"), "r") as file:
            self.config = yaml.safe_load(file)
        
        self.job_tasks = set(self.config.keys()) - set([
            "common_format", 
            "common_node_feature_masked_autoencoder", 
            "common_edge_feature_masked_autoencoder"
        ])

    def _load_indices(self):
        "load indices for each dataset, from {self.root}/indices.json"
        with open(os.path.join(self.root, "indices.json"), "r") as file:
            self.indices = json.load(file)
        assert set(self.indices.keys()) >= self.job_tasks, f"Indices for {(self.job_tasks - set(self.indices.keys()))} are missing."

    def _append_dataset_samples(self, dataset_samples: list):
        """
        Append the dataset samples to instruction_dataset.json in self.root.
        If instruction_dataset.json doesn't exist, create it with an empty list first.
        """
        # Load existing data if present
        try:
            with open(self.data_filepath, "r") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []

        existing_data.extend(dataset_samples)
        with open(self.data_filepath, "w") as f:
            json.dump(existing_data, f, indent=4)

    def _main_task_graph_generator(self, generator, sample_id):
        """
        Generate the main-task graph (and associated query/answer).
        """
        graph, metadata = generator.generate_graph(sample_id)   
        query = metadata["main_task"]["query"]
        answer = metadata["main_task"]["answer"]
        return graph, query, answer, metadata

    def _ssl_task_graph_generator(self, graph, ssl_task_name, ssl_task_config):
        """
        Generate self-supervised samples based on the original graph.
        Returns a list of (modified_graph, query, answer) tuples.
        """
        generated_samples = []
        ssl_generator_config = ssl_task_config.get("generator", {})
        ssl_ratio = ssl_task_config["augment_ratio"]

        ssl_task_generator = SelfSupervisedGraphTask.create(ssl_task_name, **ssl_generator_config)
        
        for _ in range(ssl_ratio):
            # logger.debug(f"{self.textualizer.export(graph, format='table')=}")
            try:
                ssl_sample = ssl_task_generator.generate_sample(graph)
                generated_samples.append(
                    (ssl_sample["modified_graph"], ssl_sample["query"], ssl_sample["answer"])
                )
            except:
                continue

        return generated_samples

    def format_sample(self, graph, query, answer, fmt, directed, graph_description, **kwargs):
        """
        Create a final instruction-training record including token count.
        """
        graph_text = self.textualizer.export(graph, fmt, simplify_if_no_multi=True, directed=directed)
        input_text = INPUT.format(
            graph_description=graph_description,
            format=fmt,
            graph_text=graph_text,
            query=query
        )
        
        return {
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": answer,
            "system": SYSTEM,
            "dataset": kwargs.get("dataset", "unknown"),
            "task_type": kwargs.get("task_type", "main"),
            "graph_format": fmt,
            "metadata": kwargs.get("metadata", None),
        }

    
    def _generate_samples_for_one_index(self, generator, sample_id, output_formats, ssl_settings):
        """
        Generate main-task samples + optional SSL-task samples for a single sample_id.
        Returns a list of formatted samples.
        """
        all_formatted = []

        # 1) Main task
        main_graph, main_query, main_answer, metadata = self._main_task_graph_generator(generator, sample_id)
        for fmt in output_formats:
            formatted_sample = self.format_sample(
                graph=main_graph,
                query=main_query,
                answer=main_answer,
                fmt=fmt,
                directed=generator.directed,
                graph_description=generator.graph_description,
                dataset=generator.dataset_name,
                task_type="main",
                metadata=metadata,
            )
            all_formatted.append(formatted_sample)

        # 2) SSL tasks
        if ssl_settings:  # If no ssl_settings, skip
            for ssl_task_name, ssl_config in ssl_settings.items():
                ssl_results = self._ssl_task_graph_generator(main_graph, ssl_task_name, ssl_config)
                for (ssl_graph, ssl_query, ssl_answer) in ssl_results:
                    for fmt in output_formats:
                        formatted_ssl_sample = self.format_sample(
                            graph=ssl_graph,
                            query=ssl_query,
                            answer=ssl_answer,
                            fmt=fmt,
                            directed=generator.directed,
                            graph_description=generator.graph_description,
                            dataset=generator.dataset_name,
                            task_type=ssl_task_name,
                            metadata=metadata,
                        )
                        all_formatted.append(formatted_ssl_sample)

        return all_formatted

    def generate_by_dataset(self, dataset_name: str) -> list:
        """
        Synchronously generate data for a single dataset:
        - Creates the generator
        - Loops over sample indices sequentially
        - Returns a list of final (formatted) samples
        """
        task_config = self.config[dataset_name]

        sample_indices = self.indices[dataset_name]
        logger.info(f"Generating {len(sample_indices)} samples for {dataset_name}.")
        output_formats = task_config.get("format", ["json"])
        ssl_settings = task_config.get("ssl_setting", None)

        # Create the main generator
        generator_config = task_config.get("generator", {})
        generator = InputGraphGenerator.create(dataset_name, **generator_config)
        generator.dataset_name = dataset_name

        results = []
        for sample_id in tqdm(sample_indices, desc=f"Processing {dataset_name}"):
            try:
                result = self._generate_samples_for_one_index(
                    generator=generator,
                    sample_id=sample_id,
                    output_formats=output_formats,
                    ssl_settings=ssl_settings,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample_id={sample_id} in {dataset_name}: {e}")
                continue

        # Flatten the list of lists
        dataset_samples = [sample for task_results in results for sample in task_results]
        return dataset_samples
    
        # def process_sample(example):
        #     # 提取 sample_id
        #     sample_id = example["sample_id"]
        #     # 调用原有同步方法生成样本列表
        #     # 注意：如果 generator 或其他参数不可 pickle，则建议不要启用并行（num_proc=1）
        #     return {"samples": self._generate_samples_for_one_index(
        #         generator=generator,
        #         sample_id=sample_id,
        #         output_formats=output_formats,
        #         ssl_settings=ssl_settings,
        #     )}

        # # 假设 sample_indices 是一个列表
        # ds = Dataset.from_dict({"sample_id": sample_indices})

        # # 使用 map 操作处理所有样本
        # # 如果 generator 等对象不支持 pickle，请设置 num_proc=1；如果支持，则可以适当启用多进程加速
        # # current_load = psutil.cpu_percent(percpu=False) / 100
        # # _num_proc = max(1, int(self.num_proc * (1 - current_load)))
        # results_dataset = ds.map(process_sample, batched=False, num_proc=1)
        
        # # 从结果中提取每个样本生成的列表，并扁平化为一个总的样本列表
        # results_nested = results_dataset["samples"]
        # # results_nested 的每个元素本身是一个列表（对应一个 sample_id 生成的多个样本）
        # all_samples = [sample for sublist in results_nested for sample in sublist]
        
        # return all_samples

    def batch_tokenize_and_append(self, samples):
        """
        Tokenize instructions in batches using multiprocessing, then append to the dataset file.
        """
        instructions = [
            f"{sample['instruction']}\n{sample['input']}\n{sample['output']}"
            for sample in samples
        ]
        token_counts = []

        # Load to Hugging Face Dataset
        tmp_dataset = Dataset.from_dict({"text": instructions})
        
        # Get the current CPU usage as a percentage and convert it to a ratio (0 to 1)
        current_load = psutil.cpu_percent(percpu=False) / 100.0

        # Adjust the number of processes based on the CPU idle percentage
        cpu_based_proc = int(self.num_proc * (1 - current_load))

        # Define the minimum required memory per process (e.g., 1GB) in bytes
        required_mem_per_proc = 4 * 1024 * 1024  # 1GB

        # Retrieve the available system memory in bytes
        available_mem = psutil.virtual_memory().available

        # Calculate the maximum number of processes that can be supported by the available memory
        max_proc_by_mem = available_mem // required_mem_per_proc

        # Determine the final number of processes by taking the minimum between the CPU-based and memory-based limits,
        # ensuring that at least one process is used (even if available memory is below 1GB)
        _num_proc = max(1, int(min(cpu_based_proc, max_proc_by_mem)))
        _num_proc = min(_num_proc, 32)
        
        logger.info(f"Tokenizing instructions with {_num_proc} processes.")
        
        tmp_dataset = tmp_dataset.map(lambda sample: {"#tokens": len(self.tokenizer(sample['text'], return_attention_mask=False)['input_ids'])}, batched=False, num_proc=_num_proc)
        # logger.info(f"{tmp_dataset.column_names=}")
        # tmp_dataset = tmp_dataset.map(lambda sample: {"#tokens": len(sample['input_ids'])}, remove_columns=["input_ids"])
        logger.info(f"{tmp_dataset.column_names=}")
        # Add token counts to samples
        token_counts = tmp_dataset["#tokens"]
        for sample, count in zip(samples, token_counts):
            sample["#tokens"] = count

        # 直接追加到文件（取消了异步锁）
        self._append_dataset_samples(samples)

    def _parallel_dataset_task(self, dataset_name: str):
        """
        A single task that generates samples for a dataset and processes tokens.
        """
        logger.info(f"Starting dataset: {dataset_name}")
        dataset_samples = self.generate_by_dataset(dataset_name)

        # Tokenize and append
        self.batch_tokenize_and_append(dataset_samples)
        
        logger.info(f"Done dataset: {dataset_name} with {len(dataset_samples)} samples.")

    def _pipeline(self):
        """
        Orchestrates dataset generation and tokenization sequentially.
        """
        for dataset_name in tqdm(self.job_tasks, desc="Processing datasets"):
            self._parallel_dataset_task(dataset_name)
        logger.info("--- All datasets processed and appended into data.json ---")

    def check_token_length(self):
        def analyze_tokens(df, group_columns):
            # Group by the specified columns and calculate stats for #tokens
            result = df.groupby(group_columns)['#tokens'].agg(
                count='count',
                min_tokens='min',
                quantile_25_tokens=lambda x: x.quantile(0.25),
                median_tokens=lambda x: x.median(),
                quantile_75_tokens=lambda x: x.quantile(0.75),
                max_tokens='max',
                mean_tokens='mean',
                std_dev_tokens='std',
                less_than_15000=lambda x: (x < 15000).sum()  # Count of #tokens < 15000
            ).reset_index()

            # Sort the result for better readability
            return result.sort_values(by=group_columns)
        
        df = pd.read_json(self.data_filepath, orient='records')
        stats = analyze_tokens(df, ['dataset', 'graph_format'])
        # save stats to a csv file
        stats.to_csv(os.path.join(self.root, "token_stats.csv"), index=False)
        logger.info(f"Token stats saved to {os.path.join(self.root, 'token_stats.csv')}")

    def pipeline(self):
        """
        Entry point for running the coordinator in a synchronous manner.
        """
        self._pipeline()
        logger.info("--- Pipeline finished ---")
        self.check_token_length() 

