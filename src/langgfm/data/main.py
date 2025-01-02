# import OgbnArxivGraphGenerator and GraphTextualizer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from langgfm.data.graph_generator.base_generator import InputGraphGenerator
from langgfm.data.graph_text_transformation.nxg_to_text import GraphTextualizer

import json

from tqdm import tqdm

# all GraphGenerator should be imported here for InputGraphGenerator.create()
from langgfm.data.graph_generator.ogbn_arxiv_generator import OgbnArxivGraphGenerator

# write a pipeline function to use OgbnArxivGraphGenerator to generate a graph and then use nxg_to_text to convert the graph to a GraphML format string.
def pipeline(datasets = ['ogbn_arxiv'], formats = ['json','graphml','gml','table']) -> tuple:
    """
    Generate graph texts and save into files.

    :return: None.
    """
    
    # load the generator configuration file
    config_path = os.path.join(os.path.dirname(__file__), '../configs/graph_generator.json')
    with open(config_path, 'r') as f:
        graph_generator_configs = json.load(f)
        
    generators = {}
    for dataset in datasets:
        print(f"Creating generator for {dataset}...")
        # get the configuration for the dataset
        config = graph_generator_configs.get(dataset, {})
        # create the generator instance
        generators[dataset] = InputGraphGenerator.create(dataset, **config)
    
    # load the dataset splits configuration file
    dataset_splits_path = os.path.join(os.path.dirname(__file__), '../configs/dataset_splits.json')
    with open(dataset_splits_path, 'r') as f:
        dataset_splits = json.load(f)

    # create the GraphTextualizer instance
    converter = GraphTextualizer()
    dataset_type_path = os.path.join(os.path.dirname(__file__), '../configs/dataset_type.json')
    with open(dataset_type_path, "r") as f:
        dataset_type = json.load(f)
            
    # iter over datasets and formats to generate graph_text
    for dataset in datasets:
        for format in formats:
            for split, sample_id_list in dataset_splits[dataset].items():
                data_list = []
                for sample_id in tqdm(sample_id_list, desc=f"Generating {dataset} {split} {format}"):
                    graph, metadata = generators[dataset].generate_graph(sample_id=sample_id)
                    graph_text = converter.export(graph, format=format, **dataset_type[dataset])
                    data_list.append({"graph_text": graph_text, "metadata": metadata})
                
                # save the data_list to a file with the path {dataset}/{format}/{split}.json
                # if the directory does not exist, create it
                if not os.path.exists(os.path.join(os.path.dirname(__file__), f'./outputs/{dataset}/{format}')):
                    os.makedirs(os.path.join(os.path.dirname(__file__), f'./outputs/{dataset}/{format}'))
                # create the filename
                filename = os.path.join(os.path.dirname(__file__), f'./outputs/{dataset}/{format}/{split}.json')
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=4)

# write main function to run the pipeline function and print the result
def main():
    pipeline()


# run the main function
if __name__ == "__main__":
    main(dataset=['edge_existence'], formats=['graphml'])