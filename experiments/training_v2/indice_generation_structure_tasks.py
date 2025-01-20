

import json
import os

from langgfm.utils.io import save_beautiful_json
# from langgfm.data.build_synthetic_graph import *

from langgfm.data.graph_generator._base_generator import InputGraphGenerator
from langgfm.data.dataset_generation_coordinator import DatasetGenerationCoordinator






structure_splits = json.load(open('src/langgfm/configs/data_splits_structure.json'))
train_structure_splits = {task: structure_splits[task]['train'] for task in structure_splits}
save_beautiful_json(train_structure_splits, 'experiments/training_v2/indices_structure.json')
