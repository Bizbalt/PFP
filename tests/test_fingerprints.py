import unittest
import numpy as np
from rdkit import Chem


class TestFingerprintFunctions(unittest.TestCase):
    def test_create_RDKFingerprint(self):
        from polyfingerprints.fingerprints import create_RDKFingerprint

        smiles = ["CC", "CCC"]
        result = create_RDKFingerprint(smiles)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].size, 2048)

        result_with_complement = create_RDKFingerprint(
            ["[CH2][CH](C)"], complement=True
        )
        self.assertTrue(
            np.equal(
                result_with_complement[0], create_RDKFingerprint(["CC(C)"])[0]
            ).all()
        )

    def test_merge_bit_fingerprints(self):
        from polyfingerprints.fingerprints import merge_bit_fingerprints

        fps = [np.array([True, False, True]), np.array([False, True, True])]
        result = merge_bit_fingerprints(fps)
        self.assertEqual(result.tolist(), [True, True, True])

        with self.assertRaises(ValueError):
            merge_bit_fingerprints(
                [np.array([True, False, True]), np.array([False, True])]
            )

    def test_weight_sum_fingerprints(self):
        from polyfingerprints.fingerprints import weight_sum_fingerprints

        fps = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        weights = [0.5, 0.5]
        result = weight_sum_fingerprints(fps, weights)
        np.testing.assert_array_equal(result, np.array([2.5, 3.5, 4.5]))

        with self.assertRaises(ValueError):
            weight_sum_fingerprints(fps, [0.5])

        with self.assertRaises(ValueError):
            weight_sum_fingerprints(
                [np.array([1.0, 2.0]), np.array([4.0, 5.0, 6.0])], weights
            )


if __name__ == "__main__":
    unittest.main()
