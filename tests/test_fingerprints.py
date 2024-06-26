import unittest
import numpy as np
from rdkit import Chem
from unittest.mock import patch


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


class TestReduceFPSet(unittest.TestCase):
    @patch("polyfingerprints.PFPLOGGER.info")  # Mock the "info" method of the logger
    def test_reduce_fp_set(self, mock_logger):
        from polyfingerprints.fingerprints import reduce_fp_set

        # Sample fingerprints for testing
        fp1 = np.array([0.2, 0.5, 0.1])
        fp2 = np.array([0.2, 0.6, 0.1])
        fp3 = np.array([0.2, 0.7, 0.2])

        reduced_fps, mask, reference_fp = reduce_fp_set([fp1, fp2, fp3])

        # Check that the reduced fingerprints have the expected values
        np.testing.assert_array_equal(
            reduced_fps,
            [np.array([0.5, 0.1]), np.array([0.6, 0.1]), np.array([0.7, 0.2])],
        )

        # Check that the mask is correct
        np.testing.assert_array_equal(mask, np.array([True, False, False]))

        # Check that the reference fingerprint is correct
        np.testing.assert_array_equal(reference_fp, fp1)

        # Check the logging output
        mock_logger.assert_called_once_with("reduced size by 33%")


class TestReduceAnotherFPSet(unittest.TestCase):
    # catch the logging output
    @patch("polyfingerprints.PFPLOGGER.info")  # Mock the "info" method of the logger
    def test_reduce_another_fp_set(self, mock_logger):
        from polyfingerprints.fingerprints import apply_reduction_fp_set

        # Sample fingerprints, mask, and reference_fp for testing
        fp4 = np.array([0.2, 0.6, 0.2])
        fp5 = np.array([0.2, 0.7, 0.1])
        mask = np.array([True, False, True])
        reference_fp = np.array([0.2, 0.5, 0.1])
        reduced_fps = apply_reduction_fp_set(
            fingerprints=[fp4, fp5], mask=mask, reference_fp=reference_fp
        )

        # Check that the reduced fingerprints have the expected values
        np.testing.assert_array_equal(reduced_fps, [np.array([0.6]), np.array([0.7])])

        # Check the logging output
        mock_logger.assert_called_once_with(
            "mean reduction loss is %.0f%% with the highest loss per fingerprint beeing %.0f%%",
            25.0,
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
