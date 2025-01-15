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
    # 解析参数
    args = parser.parse_args()

    # 获取 job_path 参数
    job_path = args.job_path

    # try:
    # 创建并运行 DatasetGenerationCoordinator
    coordinator = DatasetGenerationCoordinator(job_path)
    coordinator.pipeline()
    print("Dataset generation pipeline completed successfully.")
    # except Exception as e:
    #     print(f"An error occurred during dataset generation: {e}")

if __name__ == "__main__":
    main()
