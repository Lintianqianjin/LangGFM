import unittest
import os
import sys
import time
import json


from langgfm.data.dataset_generation_coordinator import DatasetGenerationCoordinator


class TestDataGeneration(unittest.TestCase):
    """Tests for the AsyncDataGenerationCoordinator pipeline."""

    def setUp(self):
        """
        Setup runs before each test method. We'll define a config
        and a unique job name so we don't collide with other runs.
        """
        self.job_path = "./experiments/training_v1_mini_debug"
        # self.job_path = "./experiments/training_v1_mini"
        self.root = self.job_path
        # self.root_path = os.path.join("./data/instruction_data/", self.job_name)
        # Example input config (mirroring your example)

        # Clean up any prior run data if it exists
        # if os.path.exists(self.root_path):
        #     # Caution: This removes the entire directory contents
        #     # in a real test environment you might use a safer approach
        #     import shutil
        #     shutil.rmtree(self.root_path)

    def test_data_generation_pipeline(self):
        """
        Test that the pipeline runs without errors and generates data.json,
        then check that the file is not empty.
        """
        # Instantiate the coordinator with our config
        coordinator = DatasetGenerationCoordinator(
            self.job_path
        )

        start_time = time.time()
        coordinator.pipeline()  # Run the async pipeline
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"Pipeline finished in {elapsed:.2f} seconds.")

        # Check if the job root path was created
        self.assertTrue(os.path.isdir(self.root), "Job root directory should exist.")

        # Check if data.json was created
        data_file = os.path.join(self.root, "instruction_dataset.json")
        self.assertTrue(os.path.isfile(data_file), "data.json should be created after pipeline runs.")

        # Check if data.json is non-empty
        file_size = os.path.getsize(data_file)
        self.assertGreater(file_size, 0, "data.json file should not be empty.")

        # (Optional) Parse data.json to check JSON structure
        with open(data_file, "r") as f:
            data_content = json.load(f)
        
        # We expect at least some entries from each dataset
        # For a minimal check: ensure data_content is a list and non-empty
        self.assertIsInstance(data_content, list, "data.json content should be a list.")
        self.assertTrue(len(data_content) > 0, "Generated samples list should not be empty.")

        # (Optional) More detailed checks can go here, e.g. verifying schema of first item:
        first_item = data_content[0]
        self.assertIn("instruction", first_item, "Each sample dict should have 'instruction' field.")
        self.assertIn("output", first_item, "Each sample dict should have 'output' field.")

    def tearDown(self):
        """
        Runs after each test. Optionally, clean up the generated data.
        Uncomment to delete generated files after the test.
        """
        # import shutil
        # if os.path.exists(self.root):
        #     shutil.rmtree(self.root)
        pass

if __name__ == "__main__":
    unittest.main()
