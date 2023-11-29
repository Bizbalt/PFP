import unittest
import numpy as np


class TestCreatePfp(unittest.TestCase):
    def test_create_pfp_default(self):
        from polyfingerprints import create_pfp

        repeating_units = {"C=C": 0.5, "C#C": 0.5}
        mol_weight = 150.0

        pfp = create_pfp(repeating_units, mol_weight)
        # Default settings should produce a Polyfingerprint with a length of:
        # 2048 + (2 * 2048) * 2 = 10240
        self.assertIsInstance(pfp, np.ndarray)
        self.assertEqual(len(pfp), 10240)

    def test_create_pfp_with_enhanced_functions(self):
        from polyfingerprints.fingerprints import create_RDKFingerprint
        from polyfingerprints import create_pfp

        repeating_units = {"C=C": 0.5, "C#C": 0.5}
        mol_weight = 150.0
        enhanced_fp_functions = [
            create_RDKFingerprint,
            create_RDKFingerprint,
        ]  # Using the same function twice for simplicity

        pfp = create_pfp(
            repeating_units, mol_weight, enhanced_fp_functions=enhanced_fp_functions
        )
        # With the above settings, the Polyfingerprint should have a length of:
        # 2048 + (2 * 2048) * 2 = 10240
        self.assertIsInstance(pfp, np.ndarray)
        self.assertEqual(len(pfp), 10240)


if __name__ == "__main__":
    unittest.main()
