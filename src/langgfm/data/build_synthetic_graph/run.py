import os

from structural_tasks_graph_builder import StructuralTaskDatasetBuilder
from langgfm.utils.io import load_yaml


job_config = os.path.join(
    os.path.dirname(__file__), 
    '../../configs/structural_task_generation.yaml'
)

tasks = load_yaml(job_config).keys()

for task in tasks:
    # if task == 'shortest_path':
    builder = StructuralTaskDatasetBuilder.create(task)
    builder.build_dataset()
    


