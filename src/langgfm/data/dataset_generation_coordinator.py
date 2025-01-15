import os
import sys
import json
import asyncio
import yaml

from typing import Dict, Any, List

from ..data.graph_generator._base_generator import InputGraphGenerator
from ..data.ssl_tasks.base_ssl import SelfSupervisedGraphTask
from ..data.ssl_tasks.tae_ssl import TopologyAutoencoder
from ..data.ssl_tasks.fmae_ssl import NodeFeatureMaskedAutoencoder, EdgeFeatureMaskedAutoencoder
from ..data.graph_text_transformation.nxg_to_text import GraphTextualizer
from ..configs.instruction_template import PROMPT_TEMPLATE



class DatasetGenerationCoordinator:
    """
    Coordinator class to manage the generation of graphs and their textual representations (asynchronously).
    This version parallelizes among datasets, writes everything into a single `data.json`,
    and uses an asyncio.Lock to safely append results.
    """

    def __init__(self, job_path: str = "./experiments/default_job"):
        self.root = job_path
        self.prompt_template = PROMPT_TEMPLATE  # Suppose this is imported
        self.textualizer = GraphTextualizer()   # Suppose this is imported

        # load the config and indices
        self._load_config()
        self._load_indices()
        
        # Root path for saving outputs
        # self.root = os.path.join("./experiments/", job_name)
        # self._mkdir()
        
        # This lock ensures only one task at a time appends to data.json
        self.data_file_lock = asyncio.Lock()

    def _load_config(self):
        "load generation needed config, from {self.root}/data_generation.yaml"
        with open(os.path.join(self.root, "data_generation.yaml"), "r") as file:
            self.config = yaml.safe_load(file)
        
    def _load_indices(self):
        "load indices for each dataset, from {self.root}/indices.json"
        with open(os.path.join(self.root, "indices.json"), "r") as file:
            self.indices = json.load(file)

    def _mkdir(self):
        """Create the job root directory if it doesn't exist."""
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    # def _save_configs(self):
    #     """Save the coordinator config to disk for reproducibility."""
    #     config_path = os.path.join(self.root, "config.json")
    #     with open(config_path, "w") as f:
    #         json.dump(self.config, f, indent=4)

    def _append_dataset_samples(self, dataset_samples: list):
        """
        Append the dataset samples to instruction_dataset.json in self.root.
        If instruction_dataset.json doesn't exist, create it with an empty list first.
        """
        data_file = os.path.join(self.root, "instruction_dataset.json")

        # Load existing data if present
        if os.path.exists(data_file):
            try:
                with open(data_file, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        else:
            existing_data = []

        # Append new samples
        existing_data.extend(dataset_samples)

        # Write them back
        with open(data_file, "w") as f:
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
        return graph, query, answer

    async def _ssl_task_graph_generator(self, graph, ssl_task_name, ssl_task_config):
        """
        Generate self-supervised samples based on the original graph.
        Returns a list of (modified_graph, query, answer) tuples.
        """
        generated_samples = []
        ssl_generator_config = ssl_task_config.get("generator", {})
        ssl_ratio = ssl_task_config["augment_ratio"]

        ssl_task_generator = SelfSupervisedGraphTask.create(ssl_task_name, **ssl_generator_config)
        
        for _ in range(ssl_ratio):
            ssl_sample = ssl_task_generator.generate_sample(graph)
            generated_samples.append(
                (ssl_sample["modified_graph"], ssl_sample["query"], ssl_sample["answer"])
            )
        return generated_samples

    def format_sample(self, graph, query, answer, fmt, directed, graph_description):
        """
        Create a final instruction-training record in the format:
            {
               "instruction": <prompt>,
               "input": "",
               "output": <answer>
            }
        """
        graph_text = self.textualizer.export(graph, fmt,directed=directed)
        instruction = self.prompt_template.format(
            graph_description=graph_description,
            graph_text=graph_text,
            query=query,
        )
        return {
            "instruction": instruction,
            "input": "",
            "output": answer
        }

    async def _generate_samples_for_one_index(
        self,
        generator,
        sample_id,
        output_formats,
        directed,
        graph_description,
        ssl_settings
    ):
        """
        Generate main-task samples + optional SSL-task samples for a single sample_id.
        Returns a list of formatted samples.
        """
        all_formatted = []

        # 1) Main task
        main_graph, main_query, main_answer = await self._main_task_graph_generator(generator, sample_id)
        for fmt in output_formats:
            formatted_sample = self.format_sample(
                graph=main_graph,
                query=main_query,
                answer=main_answer,
                fmt=fmt,
                directed=directed,
                graph_description=graph_description
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
                            directed=directed,
                            graph_description=graph_description
                        )
                        all_formatted.append(formatted_ssl_sample)

        return all_formatted

    async def generate_by_dataset(self, dataset_name: str) -> list:
        """
        Asynchronously generate data for a single dataset:
        - Creates the generator
        - Loops over sample indices, potentially in parallel
        - Returns a list of final (formatted) samples
        """
        task_config = self.config[dataset_name]

        # sample_indices = task_config.get("index", [])
        sample_indices = self.indices[dataset_name]
        output_formats = task_config.get("format", ["json"])
        ssl_settings = task_config.get("ssl_setting", None)

        # Create the main generator
        generator_config = task_config.get("generator", {})
        generator = InputGraphGenerator.create(dataset_name, **generator_config)
        graph_description = generator.graph_description
        directed = generator.directed

        # Collect tasks for concurrency at the sample level
        tasks = []
        for sample_id in sample_indices:
            tasks.append(
                asyncio.create_task(
                    self._generate_samples_for_one_index(
                        generator=generator,
                        sample_id=sample_id,
                        output_formats=output_formats,
                        directed=directed,
                        graph_description=graph_description,
                        ssl_settings=ssl_settings
                    )
                )
            )
        
        # Run them concurrently
        results = await asyncio.gather(*tasks)

        # Flatten
        dataset_samples = []
        for res in results:
            dataset_samples.extend(res)

        return dataset_samples

    # -------------------------------------------------------------------
    # Parallel dataset tasks & pipeline orchestration
    # -------------------------------------------------------------------

    async def _parallel_dataset_task(self, dataset_name: str):
        """
        A single 'task' that generates samples for a dataset in parallel
        (sample-level concurrency), then appends them to data.json 
        using a lock to avoid conflicts.
        """
        print(f"Starting dataset: {dataset_name}")
        dataset_samples = await self.generate_by_dataset(dataset_name)
        
        # Safely append to the shared data.json file
        async with self.data_file_lock:
            self._append_dataset_samples(dataset_samples)

        print(f"Done dataset: {dataset_name} with {len(dataset_samples)} samples.")

    async def _async_pipeline(self):
        """
        Orchestrates dataset generation in parallel: 
         - create a task per dataset
         - gather them 
        """
        # self._save_configs()

        # Create parallel tasks for each dataset
        tasks = []
        for dataset_name in self.config:
            tasks.append(
                asyncio.create_task(self._parallel_dataset_task(dataset_name))
            )

        # Wait for all to finish
        await asyncio.gather(*tasks)

        print("--- All datasets processed and appended into data.json ---")

    def pipeline(self):
        """
        Entry point for running the coordinator. 
        Uses asyncio.run(...) to kick off asynchronous tasks.
        """
        asyncio.run(self._async_pipeline())
        print("--- Pipeline finished ---")
