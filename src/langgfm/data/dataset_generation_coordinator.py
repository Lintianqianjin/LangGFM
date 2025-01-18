import os
import sys
import json
import asyncio
import yaml
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Any, List

from ..data.graph_generator._base_generator import InputGraphGenerator
from ..data.ssl_tasks.base_ssl import SelfSupervisedGraphTask
from ..data.ssl_tasks.tae_ssl import TopologyAutoencoder
from ..data.ssl_tasks.fmae_ssl import NodeFeatureMaskedAutoencoder, EdgeFeatureMaskedAutoencoder
from ..data.graph_text_transformation.nxg_to_text import GraphTextualizer
from ..configs.instruction_template import SYSTEM, INSTRUCTION, INPUT
from ..utils.logger import logger
import logging
logger.set_level(logging.INFO)

from multiprocessing import Pool
import asyncio
from ..utils.language_model import count_tokens_batch


class DatasetGenerationCoordinator:
    """
    Coordinator class to manage the generation of graphs and their textual representations (asynchronously).
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
    TODO: add pbar for progress tracking.
    """
    def __init__(self, job_path: str = "./experiments/default_job", _continue: bool = False):
        self.root = job_path
        self.data_filepath = os.path.join(self.root, "instruction_dataset.json")
        self.textualizer = GraphTextualizer()   # Suppose this is imported
        self._load_config()
        self._load_indices()
        self.data_file_lock = asyncio.Lock()

        if not _continue:
            with open(self.data_filepath, "w") as f:
                f.write("[]")

    def _load_config(self):
        "load generation needed config, from {self.root}/data_generation.yaml"
        with open(os.path.join(self.root, "data_generation.yaml"), "r") as file:
            self.config = yaml.safe_load(file)
        
        self.job_tasks = set(self.config.keys()) \
            - set([
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

    # -------------------------------------------------------------------
    # Asynchronous methods for generating data
    # -------------------------------------------------------------------

    async def _main_task_graph_generator(self, generator, sample_id):
        """
        Generate the main-task graph (and associated query/answer).
        """
        graph, metadata = generator.generate_graph(sample_id)
        query = metadata["main_task"]["query"]
        answer = metadata["main_task"]["answer"]
        return graph, query, answer, metadata

    async def _ssl_task_graph_generator(self, graph, ssl_task_name, ssl_task_config):
        """
        Generate self-supervised samples based on the original graph.
        Returns a list of (modified_graph, query, answer) tuples.
        """
        generated_samples = []
        ssl_generator_config = ssl_task_config.get("generator", {})
        ssl_ratio = ssl_task_config["augment_ratio"]

        ssl_task_generator = SelfSupervisedGraphTask.create(ssl_task_name, **ssl_generator_config)
        
        # try:
        for _ in range(ssl_ratio):
            ssl_sample = ssl_task_generator.generate_sample(graph)
            generated_samples.append(
                (ssl_sample["modified_graph"], ssl_sample["query"], ssl_sample["answer"])
            )
        # except:
        #     # 输出图的全部信息
        #     print("Graph Info:")
        #     print(f"{graph.nodes(data=True)=}")  # 节点及属性
        #     print(f"{graph.edges(data=True)=}")  # 边及属性
        #     print(f"{graph.graph=}", )  # 图的全局属性

        return generated_samples

    def format_sample(self, graph, query, answer, fmt, directed, graph_description, **kwargs):
        """
        Create a final instruction-training record including token count.
        """
        graph_text = self.textualizer.export(graph, fmt, directed=directed)
        input = INPUT.format(
            graph_description=graph_description,
            graph_text=graph_text,
            query=query
        )
        
        return {
            "instruction": INSTRUCTION,
            "input": input,
            "output": answer,
            "system": SYSTEM,
            "dataset": kwargs.get("dataset", "unknown"),
            "task_type": kwargs.get("task_type", "main"),
            "graph_format": fmt,
            "metadata": kwargs.get("metadata", None),
        }

    async def _generate_samples_for_one_index(
        self,
        generator,
        sample_id,
        output_formats,
        ssl_settings
    ):
        """
        Generate main-task samples + optional SSL-task samples for a single sample_id.
        Returns a list of formatted samples.
        """
        all_formatted = []

        # 1) Main task
        main_graph, main_query, main_answer, metadata = await self._main_task_graph_generator(generator, sample_id)
        try:
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
                    ssl_results = await self._ssl_task_graph_generator(main_graph, ssl_task_name, ssl_config)
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
        except Exception as e:
            logger.info(f"Error in main task: {e}. {sample_id=}")
            return []

        return all_formatted

    async def generate_by_dataset(self, dataset_name: str) -> list:
        """
        Asynchronously generate data for a single dataset:
        - Creates the generator
        - Loops over sample indices, potentially in parallel
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

        # Collect tasks for concurrency at the sample level, Wrap tasks in tqdm
        tasks = [
            asyncio.create_task(
                self._generate_samples_for_one_index(
                    generator=generator,
                    sample_id=sample_id,
                    output_formats=output_formats,
                    ssl_settings=ssl_settings,
                )
            )
            for sample_id in sample_indices
        ]

        # Run them concurrently, Track progress with tqdm
        results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {dataset_name}", total=len(tasks))

        # Flatten
        dataset_samples = [sample for task_results in results for sample in task_results]
        return dataset_samples
    
    # -------------------------------------------------------------------
    # Parallel dataset tasks & pipeline orchestration
    # -------------------------------------------------------------------

    async def batch_tokenize_and_append(self, samples, batch_size=1000, num_workers=4):
        """
        Tokenize instructions in batches using multiprocessing, then append to the dataset file.
        """
        instructions = [sample["instruction"] + sample['input'] + sample["output"] + sample["system"]
            for sample in samples]
        batches = [instructions[i:i + batch_size] for i in range(0, len(instructions), batch_size)]
        token_counts = []

        # Parallel token counting
        with Pool(num_workers) as pool:
            for result in pool.imap(count_tokens_batch, batches):
                token_counts.extend(result)

        # Add token counts to samples
        for sample, count in zip(samples, token_counts):
            sample["#tokens"] = count

        # Append to file
        async with self.data_file_lock:
            self._append_dataset_samples(samples)

    async def _parallel_dataset_task(self, dataset_name: str):
        """
        A single 'task' that generates samples for a dataset and processes tokens in parallel.
        """
        print(f"Starting dataset: {dataset_name}")
        dataset_samples = await self.generate_by_dataset(dataset_name)

        # Tokenize and append
        await self.batch_tokenize_and_append(dataset_samples, batch_size=1000, num_workers=8)
        print(f"Done dataset: {dataset_name} with {len(dataset_samples)} samples.")

    async def _async_pipeline(self):
        """
        Orchestrates dataset generation and tokenization in parallel.
        """
        tasks = [
            asyncio.create_task(self._parallel_dataset_task(dataset_name))
            for dataset_name in self.job_tasks
        ]
        await tqdm_asyncio.gather(*tasks, desc="Processing datasets", total=len(tasks))
        print("--- All datasets processed and appended into data.json ---")

    def pipeline(self):
        """
        Entry point for running the coordinator. 
        Uses asyncio.run(...) to kick off asynchronous tasks.
        """
        asyncio.run(self._async_pipeline())
        print("--- Pipeline finished ---")