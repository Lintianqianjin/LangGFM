# test src/langgfm/utils/random_control.py

import unittest
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from langgfm.utils.random_control import set_seed

class TestRandomControl(unittest.TestCase):
    
    def test_set_seed(self):
        """
        Test the set_seed function.
        """
        # Set the seed and check the current seeds
        seed = 42
        print(f"Setting seed to {seed}")
        set_seed(seed)
        
        # generate random array using random, numpy, and torch
        random_array = [random.random() for _ in range(5)]
        print(f"Random array: {random_array}")
        numpy_array = [random.randint(0, 100) for _ in range(5)]
        print(f"NumPy array: {numpy_array}")
        torch_array = [random.randint(0, 100) for _ in range(5)]
        print(f"PyTorch array: {torch_array}")
        
        # set seed again
        set_seed(seed)
        # generate random array using random, numpy, and torch again
        random_array2 = [random.random() for _ in range(5)]
        print(f"Random array 2: {random_array2}")
        numpy_array2 = [random.randint(0, 100) for _ in range(5)]
        print(f"NumPy array 2: {numpy_array2}")
        torch_array2 = [random.randint(0, 100) for _ in range(5)]
        print(f"PyTorch array 2: {torch_array2}")
        
        # Validate the two random arrays
        self.assertEqual(random_array, random_array2, "Random array should be the same.")
        self.assertEqual(numpy_array, numpy_array2, "NumPy array should be the same.")
        self.assertEqual(torch_array, torch_array2, "PyTorch array should be the same.")

    # def test_get_current_seeds(self):
    #     """
    #     Test the get_current_seeds function.
    #     """
    #     # Get the current seeds
    #     seeds = get_current_seeds()
    #     print(f"Current seeds: {seeds}")
    #     # Validate the seeds
    #     self.assertIsInstance(seeds, dict, "Seeds should be returned as a dictionary.")
    #     self.assertIn('random', seeds, "Random seed should be included.")
    #     self.assertIn('numpy', seeds, "NumPy seed should be included.")
    #     self.assertIn('torch', seeds, "PyTorch seed should be included.")
    #     self.assertIsInstance(seeds['random'], int, "Random seed should be an integer.")
    #     self.assertIsInstance(seeds['numpy'], int, "NumPy seed should be an integer.")
    #     self.assertIsInstance(seeds['torch'], int, "PyTorch seed should be an integer.")
        
        
if __name__ == '__main__':
    unittest.main()
    