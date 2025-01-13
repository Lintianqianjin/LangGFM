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
        graph_text = self.textualizer.export(graph,fmt, **datatype)
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


if __name__ == "__main__":
    # Example input config (could be read from anywhere, or just declared)
    job = "struc_bace"
    job_config = {
    #     "graph_structure_detection": {
    #         "generator": {},
    #         "index": [101, 102, 103],
    #         "format": ["gml", "json"]
    #    },
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
                "edge_feature_masked_autoencoder": {
                   "generator": {
                        "mask_node_ratio": 0.2,
                        "mask_edge_ratio": 0.2,
                        "mask_reverse_edges": True, 
                    },
                    "augment_ratio":1
                },
                "topology_autoencoder": {
                   "generator": {
                        "distinguish_directions": False
                    },
                    "augment_ratio":2
                }
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
                "topology_autoencoder": {
                   "generator": {
                        "distinguish_directions": False
                    },
                    "augment_ratio":2
                }
            },
            "format": ["gml", "json","table","graphml"]
        # #     "format": ["json","table","graphml"] "gml", 
        }
    }
    
    # Run the pipeline
    # outputs = coordinator_pipeline(job_config,job_name=job)
    
    coordinator = DataGenerationCoordinator(job_config, job_name=job)
    coordinator.pipeline()