import fire
from langgfm.data.dataset_generation_coordinator import DatasetGenerationCoordinator

def main(job_path: str, continue_flag: bool = False, return_token_length: bool = True, tokenizer_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Run the dataset generation pipeline.

    Args:
        job_path (str): Path to the dataset generation job directory.
        continue_flag (bool): Continue from the last checkpoint if set to True.
    """
    coordinator = DatasetGenerationCoordinator(
        job_path=job_path,
        is_continue=continue_flag,
        return_token_length=return_token_length,
        tokenizer_name_or_path=tokenizer_name_or_path
    )
    coordinator.pipeline()
    print("Dataset generation pipeline completed successfully.")

if __name__ == "__main__":
    fire.Fire(main)
