import argparse
from langgfm.data.dataset_generation_coordinator import DatasetGenerationCoordinator
import logging
from langgfm.utils.logger import logger
logger.set_level(logging.WARNING)

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Run the dataset generation pipeline.")
    # 添加参数
    parser.add_argument(
        "--job_path",
        type=str,
        required=True,
        help="Path to the dataset generation job directory."
    )
    parser.add_argument(
        "--continue_flag",
        action="store_true",
        help="Continue from the last checkpoint."
    )

    args = parser.parse_args()
    job_path = args.job_path
    _continue = args.continue_flag

    coordinator = DatasetGenerationCoordinator(job_path,_continue)
    coordinator.pipeline()
    print("Dataset generation pipeline completed successfully.")
    # except Exception as e:
    #     print(f"An error occurred during dataset generation: {e}")

if __name__ == "__main__":
    main()
