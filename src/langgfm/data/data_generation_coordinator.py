# run at LangGFM.
import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from langgfm.data.graph_generator._base_generator import InputGraphGenerator
from langgfm.data.ssl_tasks.base_ssl import SelfSupervisedGraphTask
from langgfm.data.ssl_tasks.tae_ssl import TopologyAutoencoder
from langgfm.data.ssl_tasks.fmae_ssl import NodeFeatureMaskedAutoencoder, EdgeFeatureMaskedAutoencoder
from langgfm.data.graph_text_transformation.nxg_to_text import GraphTextualizer
from langgfm.configs.llama_factory_template import PROMPT_TEMPLATE


class DataGenerationCoordinator:
    """
    Coordinator class to manage the generation of graphs and their textual representations.
    TODO:
     - asynchoronous generation.
     - argument seperation, part affliating to generator, part affliating to coordinator.
     - read coordinator argument from yaml.
    """
    def __init__(self, config: Dict[str, Any], job_name: str = "./../"):
        
        self.job_name = job_name
        self.config = config
        self.prompt_template = PROMPT_TEMPLATE
        self.textualizer = GraphTextualizer()
        
        self.root = os.path.join('./data/instruction_data/',job_name)
        self.all_samples = []
        
        
    def _mkdir(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def _save_configs(self):
        with open(f"{self.root}/config.json", "w") as f:
            json.dump(self.config, f, indent=4)
    
    def _save_generated_samples(self):
        with open(f"{self.root}/data.json", "w") as f:
            json.dump(self.all_samples, f, indent=4)

    def _main_task_graph_generator(self, generator, sample_id):
        graph, metadata = generator.generate_graph(sample_id)
        query = metadata['main_task']["query"]
        answer = metadata['main_task']["answer"]
        return graph, query, answer

    def _ssl_task_graph_generator(self, graph, ssl_task_name, ssl_task_config):
        "return a list of samples"
        generated_samples = []
        # print(ssl_task_config)
        # exit()
        ssl_generator_config, ssl_ratio = ssl_task_config['generator'], ssl_task_config['augment_ratio']
        ssl_task_generator = SelfSupervisedGraphTask.create(ssl_task_name, **ssl_generator_config)
        for _ in range(ssl_ratio):
            # print(graph)
            # print(type(graph))
            # exit()
            ssl_sample = ssl_task_generator.generate_sample(graph)
            generated_samples.append((ssl_sample['modified_graph'], ssl_sample['query'], ssl_sample['answer']))
        return generated_samples
    
    def format_sample(self, graph, query, answer, fmt, datatype, graph_description):
        graph_text = self.textualizer.export(graph,fmt)
        instruction = self.prompt_template.format(
            graph_description=graph_description,
            graph_text=graph_text,
            query=query
        )
        return {"instruction": instruction,"input": "","output": answer}

    def generate_by_dataset(self, dataset_name: str):
        task_config = self.config[dataset_name]
        # TODO. para datatype for export
        sample_indices = task_config.get("index", [])
        output_formats = task_config.get("format", ["json"])
        datatype = task_config.get("datatype", {})
        generator = InputGraphGenerator.create(dataset_name, **task_config['generator'])
        graph_description = generator.graph_description

        for sample_id in sample_indices:
            # try:
            for fmt in output_formats:
                # main task
                graph, query, answer = self._main_task_graph_generator(generator, sample_id)
                self.all_samples.append(self.format_sample(graph, query, answer, fmt, datatype, graph_description))

                # ssl tasks
                if 'ssl_setting' not in task_config:
                    continue
                for ssl_task, ssl_config in task_config['ssl_setting'].items():
                    for graph, query, answer in self._ssl_task_graph_generator(graph, ssl_task, ssl_config):
                        self.all_samples.append(self.format_sample(graph, query, answer, fmt, datatype, graph_description))
            # except Exception as e:
            #     print(f"Error generating sample {sample_id} for dataset {dataset_name}: {e}")
            #     continue

    def pipeline(self):
        self._mkdir()
        self._save_configs()

        for dataset_name in self.config.keys():
            print(f"--- Creating generator for dataset: {dataset_name} ---")
            self.generate_by_dataset(dataset_name)
            print(f"--- Finish generating samples for dataset: {dataset_name} ---")

        self._save_generated_samples()


import os
import json
import asyncio
from typing import Dict, Any

# Hypothetical imports:
# from your_module import InputGraphGenerator, GraphTextualizer, SelfSupervisedGraphTask, PROMPT_TEMPLATE

class AsyncDataGenerationCoordinator:
    """
    Coordinator class to manage the generation of graphs and their textual representations (asynchronously).
    This version parallelizes among datasets, writes everything into a single `data.json`,
    and uses an asyncio.Lock to safely append results.
    """

    def __init__(self, config: Dict[str, Any], job_name: str = "default_job"):
        self.job_name = job_name
        self.config = config
        self.prompt_template = PROMPT_TEMPLATE  # Suppose this is imported
        self.textualizer = GraphTextualizer()   # Suppose this is imported

        # Root path for saving outputs
        self.root = os.path.join("./data/instruction_data/", job_name)
        self._mkdir()
        
        # This lock ensures only one task at a time appends to data.json
        self.data_file_lock = asyncio.Lock()

    def _mkdir(self):
        """Create the job root directory if it doesn't exist."""
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def _save_configs(self):
        """Save the coordinator config to disk for reproducibility."""
        config_path = os.path.join(self.root, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def _append_dataset_samples(self, dataset_samples: list):
        """
        Append the dataset samples to data.json in self.root.
        If data.json doesn't exist, create it with an empty list first.
        """
        data_file = os.path.join(self.root, "data.json")

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
        ssl_generator_config = ssl_task_config["generator"]
        ssl_ratio = ssl_task_config["augment_ratio"]

        ssl_task_generator = SelfSupervisedGraphTask.create(ssl_task_name, **ssl_generator_config)
        
        for _ in range(ssl_ratio):
            ssl_sample = ssl_task_generator.generate_sample(graph)
            generated_samples.append(
                (ssl_sample["modified_graph"], ssl_sample["query"], ssl_sample["answer"])
            )
        return generated_samples

    def format_sample(self, graph, query, answer, fmt, datatype, graph_description):
        """
        Create a final instruction-training record in the format:
            {
               "instruction": <prompt>,
               "input": "",
               "output": <answer>
            }
        """
        graph_text = self.textualizer.export(graph, fmt)
        instruction = self.prompt_template.format(
            graph_description=graph_description,
            graph_text=graph_text,
            query=query
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
        datatype,
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
                datatype=datatype,
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
                            datatype=datatype,
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

        sample_indices = task_config.get("index", [])
        output_formats = task_config.get("format", ["json"])
        datatype = task_config.get("datatype", {})
        ssl_settings = task_config.get("ssl_setting", None)

        # Create the main generator
        generator_config = task_config["generator"]
        generator = InputGraphGenerator.create(dataset_name, **generator_config)
        graph_description = generator.graph_description

        # Collect tasks for concurrency at the sample level
        tasks = []
        for sample_id in sample_indices:
            tasks.append(
                asyncio.create_task(
                    self._generate_samples_for_one_index(
                        generator=generator,
                        sample_id=sample_id,
                        output_formats=output_formats,
                        datatype=datatype,
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
        self._save_configs()

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




if __name__ == "__main__":
    # Example input config (could be read from anywhere, or just declared)
    job = "struc_bace"
    job_config = {
        "graph_structure_detection": {
            "generator": {},
            "index": [101, 102, 103],
            "format": ["gml", "json"]
       },
        "bace": {
            "generator": {"task_level": "graph"},
            "index": [1, 2],
            "datatype": {"directed": False},
            "ssl_setting": {
               "node_feature_masked_autoencoder": {
                   "generator": {
                        "mask_node_ratio": 0.2,
                        "mask_edge_ratio": 0.2,
                        "mask_reverse_edges": True, 
                    },
                    "augment_ratio":1
                },
                # "topology_autoencoder": {
                #    "generator": {
                #         "distinguish_directions": False
                #     },
                #     "augment_ratio":2
                # }
            },
            "format": ["gml", "json","table","graphml"]
       },
        "movielens1m": {
            "generator": {
                "task_level": "edge",
                "num_hops": 1,
                "sampling": True,
                "neighbor_size": [50],
                "random_seed": 42
            },
            "index": [(4027, 1931), (751, 558), (186, 793)],
            "datatype": {"directed": True},
            "ssl_setting": {
               "node_feature_masked_autoencoder": {
                   "generator": {
                        "mask_node_ratio": 0.2,
                        "mask_edge_ratio": 0.2,
                        "mask_reverse_edges": True, 
                    },
                    "augment_ratio":1
                },
        #         "topology_autoencoder": {
        #            "generator": {
        #                 "distinguish_directions": False
        #             },
        #             "augment_ratio":2
        #         }
            },
        #     # "format": ["gml", "json","table","graphml"]
            "format": ["json","table","graphml"]
        }
    }
    
    # Run the pipeline
    # outputs = coordinator_pipeline(job_config,job_name=job)
    
    coordinator = AsyncDataGenerationCoordinator(job_config, job_name=job)
    coordinator.pipeline()