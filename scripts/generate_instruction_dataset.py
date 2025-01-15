from langgfm.data.dataset_generation_coordinator import DatasetGenerationCoordinator

job_path = "./experiments/training_v1_mini"
coordinator = DatasetGenerationCoordinator(job_path)
coordinator.pipeline()