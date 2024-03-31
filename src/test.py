# This file is used to test the code in the src folder

import unittest
import utils
import torch
import pandas as pd
import numpy as np

class TestSrc(unittest.TestCase):
    def test_dotbracket_to_basepairs(self):
        # Test the dotbracket to base pairs conversion
        dotbracket = "(((...)))"
        base_pairs = utils.dotbracket_to_basepairs(dotbracket)
        self.assertEqual(base_pairs, [(1, 9), (2, 8), (3, 7)])
    
    def test_dot_to_tensor(self):
        # Test the dotbracket to tensor conversion
        dotbracket = "(((...)))"
        tensor = utils.dot_to_tensor(dotbracket)
        self.assertEqual(tensor.shape, (9, 9))
        # Corrected checks for paired and unpaired bases
        paired_indices = [0, 1, 2, 6, 7, 8]  # Based on the dotbracket structure
        unpaired_indices = [3, 4, 5]
        for i in paired_indices:
            self.assertEqual(tensor[:, i].sum(), 1)
            self.assertEqual(tensor[i, :].sum(), 1)
        for i in unpaired_indices:
            self.assertEqual(tensor[:, i].sum(), 0)
            self.assertEqual(tensor[i, :].sum(), 0)
        self.assertTrue(torch.allclose(tensor, tensor.t()))
        
    def test_basepairs_to_tensor(self):
        # Test the base pairs to tensor conversion
        base_pairs = [(1, 9), (2, 8), (3, 7)]  # Assuming 1-based indexing
        tensor = utils.basepairs_to_tensor(base_pairs, 9)
        self.assertEqual(tensor.shape, (9, 9))
        
        # Adjust expectations for sums based on pairing
        expected_sums = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype = torch.float)  # Paired bases have sum 1, unpaired bases have sum 0
        self.assertTrue(torch.allclose(tensor.sum(dim=1), expected_sums))
        self.assertTrue(torch.allclose(tensor.sum(dim=0), expected_sums))
        self.assertTrue(torch.allclose(tensor, tensor.t()))

    def test_pair_strength(self):
        # Test the pair strength calculation
        self.assertEqual(utils.pair_strength(("G", "C")), 3)
        self.assertEqual(utils.pair_strength(("A", "U")), 2)
        self.assertEqual(utils.pair_strength(("G", "U")), 1)
        self.assertEqual(utils.pair_strength(("A", "C")), 0)

    def test_probability_matrix(self):
        sequence = "AGCU"
        expected_output = np.zeros((4, 4))
        
        # Adjust your expected output based on corrected understanding of how coefficients are calculated
        
        prob_matrix = utils.probability_matrix(sequence)
        
        # Ensure correct dtype and device as needed
        expected_tensor = torch.tensor(expected_output, dtype=torch.float32)
        
        self.assertEqual(prob_matrix.shape, (4, 4))
        self.assertTrue(torch.allclose(prob_matrix, expected_tensor, atol=1e-5, rtol=1e-3))

if __name__ == "__main__":
    unittest.main()