import unittest
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


class TestmolarWtfromSmiles(unittest.TestCase):
    def test_molar_wt_fromSmiles_1(self):
        from polyfingerprints.utils import molar_wt_fromSmiles

        smiles_1 = "C"  # Methane
        smiles_2 = "CC"  # Water
        mol = Chem.MolFromSmiles(smiles_1)
        expected_mol_weight = Descriptors.MolWt(mol)
        calculated_mol_weight = molar_wt_fromSmiles(smiles_1)
        self.assertAlmostEqual(expected_mol_weight, calculated_mol_weight, places=5)

        mol = Chem.MolFromSmiles(smiles_2)
        expected_mol_weight = Descriptors.MolWt(mol)
        calculated_mol_weight = molar_wt_fromSmiles(smiles_2)
        self.assertAlmostEqual(expected_mol_weight, calculated_mol_weight, places=5)

    def test_molar_wt_fromSmiles_invalid_smiles(self):
        from polyfingerprints.utils import molar_wt_fromSmiles

        # This is to test if the function behaves well with an invalid smiles string
        invalid_smiles = "invalid_smiles"
        with self.assertRaises(Exception):
            molar_wt_fromSmiles(invalid_smiles)


class TestCalcPolymerShares(unittest.TestCase):
    def setUp(self):
        # Sample repeating units and end groups
        self.rep_units = {"C": 0.6, "O": 0.4}
        self.ends = ["F", "Cl"]
        self.total_weight = 100

    def test_polymer_shares_give_100perc(self):
        from polyfingerprints.utils import calc_polymer_shares

        ru_mole_fractions, ends_mole_fraction = calc_polymer_shares(
            self.rep_units, self.ends, self.total_weight
        )

        self.assertTrue(sum(ru_mole_fractions.values()) + sum(ends_mole_fraction) == 1)

    def test_total_weight_less_than_ends(self):
        from polyfingerprints.utils import calc_polymer_shares

        with self.assertRaises(ValueError):
            calc_polymer_shares(self.rep_units, ["F", "CCCCC"], 50)

    def test_normalize_rep_units(self):
        from polyfingerprints.utils import calc_polymer_shares

        rep_units = {"C": 60, "O": 40}
        ru_mole_fractions, _ = calc_polymer_shares(
            rep_units, self.ends, self.total_weight
        )
        self.assertTrue(sum(ru_mole_fractions.values()) <= 1)

    def test_resulting_mole_fractions_sum_to_1(self):
        from polyfingerprints.utils import calc_polymer_shares

        ru_mole_fractions, ends_mole_fraction = calc_polymer_shares(
            self.rep_units, self.ends, self.total_weight
        )

        self.assertAlmostEqual(
            sum(ru_mole_fractions.values()) + sum(ends_mole_fraction), 1, places=5
        )


class TestRepeatingUnitCombinations(unittest.TestCase):
    def test_combinations(self):
        from polyfingerprints.utils import repeating_unit_combinations

        repeating_units = ["[CH2]C[CH2]", "[CH2][CH](-C)"]
        start_unit = "[H]"
        end_unit = "[CH3]"
        expected_combinations = [
            "[H][CH2]C[CH2][CH2]C[CH2][CH3]",
            "[H][CH2]C[CH2][CH2][CH](-C)[CH3]",
            "[H][CH2][CH](-C)[CH2]C[CH2][CH3]",
            "[H][CH2][CH](-C)[CH2][CH](-C)[CH3]",
        ]

        result = repeating_unit_combinations(
            repeating_units, start=start_unit, end=end_unit
        )
        self.assertListEqual(result, expected_combinations)

    def test_combinations_length(self):
        from polyfingerprints.utils import repeating_unit_combinations

        for i in range(1, 5):
            units = [f"U{j+1}" for j in range(i)]
            combinations = repeating_unit_combinations(units)
            self.assertEqual(len(combinations), i**i)


class TestTestPolymerSmiles(unittest.TestCase):
    def test_valid_polymer_smiles(self):
        from polyfingerprints.utils import test_polymer_smiles

        # Provided examples
        self.assertFalse(test_polymer_smiles("CCC(=O)OC"))  # has no open ends
        self.assertFalse(
            test_polymer_smiles("[CH2][CH]C(=O)OC")
        )  # has both open ends, but not at the terminal atoms
        self.assertTrue(
            test_polymer_smiles("[CH2][CH](C(=O)(OC))")
        )  # has both open ends at the terminal atoms

        # Additional test cases
        self.assertFalse(test_polymer_smiles(""))  # empty string
        self.assertTrue(
            test_polymer_smiles("[CH2][CH](C(=O)(OC(C)(C)OCC))")
        )  # a more complex structure with open ends

        # Test cases that can potentially raise exceptions
        self.assertFalse(
            test_polymer_smiles("[CH2][CH]C(=O)(OC")
        )  # unbalanced parentheses
        self.assertFalse(
            test_polymer_smiles("invalidSmiles")
        )  # not a valid SMILES string

    def test_valid_startgroup(self):
        from polyfingerprints.utils import test_startgroup

        # Provided examples
        self.assertFalse(test_startgroup("CCC"))  # has no open ends
        self.assertFalse(
            test_startgroup("C[CH]C")
        )  # has a radical but not at the terminal atom
        self.assertTrue(
            test_startgroup("C[CH](C)")
        )  # has a radical at the terminal atom

        # Additional test cases
        self.assertFalse(test_startgroup(""))  # empty string
        self.assertFalse(
            test_startgroup("[CH2]CC")
        )  # radical at the start of the molecule
        self.assertTrue(
            test_startgroup("[CH2][CH2]")
        )  # radicals at the end start is optional
        self.assertTrue(test_startgroup("CC[CH2]"))  # radical at the end

        # Test cases that can potentially raise exceptions
        self.assertFalse(test_startgroup("C[CH2]C("))  # unbalanced parentheses
        self.assertFalse(test_startgroup("invalidSmiles"))  # not a valid SMILES string

    def test_valid_endgroup(self):
        from polyfingerprints.utils import test_endgroup

        # Provided examples
        self.assertFalse(test_endgroup("CCC"))  # has no open ends
        self.assertFalse(
            test_endgroup("C[CH]C")
        )  # has a radical but not at the first atom
        self.assertTrue(test_endgroup("[CH]C(C)"))  # has a radical at the terminal atom

        # Additional test cases
        self.assertFalse(test_endgroup(""))  # empty string
        self.assertFalse(test_endgroup("CC[CH2]"))  # radical at the end of the molecule
        self.assertTrue(
            test_endgroup("[CH2][CH2]")
        )  # radicals at the end start is optional
        self.assertTrue(test_endgroup("[CH2]CC"))  # radical at the start

        # Test cases that can potentially raise exceptions
        self.assertFalse(test_endgroup("[CH2]CC("))  # unbalanced parentheses
        self.assertFalse(test_endgroup("invalidSmiles"))  # not a valid SMILES string


class TestCategoricalFunction(unittest.TestCase):
    def test_with_categorical_data(self):
        from polyfingerprints.utils import test_categorical

        self.assertTrue(test_categorical(["apple", "banana", "cherry"]))
        self.assertTrue(
            test_categorical(np.array(["apple", "banana", "cherry"], dtype=object))
        )
        self.assertTrue(test_categorical(np.array(["1", "2", "3"], dtype=str)))

    def test_with_numerical_data(self):
        from polyfingerprints.utils import test_categorical

        self.assertFalse(test_categorical([1, 2, 3]))
        self.assertFalse(test_categorical(np.array([1.5, 2.5, 3.5])))
        self.assertFalse(test_categorical(np.array([1, 2, 3], dtype=np.int64)))
        self.assertFalse(test_categorical(np.array([1, 2, 3], dtype=np.uint8)))
        self.assertFalse(test_categorical(np.array([True, False, True], dtype=bool)))

    def test_with_empty_data(self):
        from polyfingerprints.utils import test_categorical

        with self.assertRaises(ValueError):
            test_categorical([])

    def test_with_higher_dimensional_data(self):
        from polyfingerprints.utils import test_categorical

        with self.assertRaises(ValueError):
            test_categorical([[1, 2, 3], [4, 5, 6]])

    def test_with_pandas_series(self):
        from polyfingerprints.utils import test_categorical
        import pandas as pd

        self.assertTrue(test_categorical(pd.Series(["apple", "banana", "cherry"])))
        self.assertFalse(test_categorical(pd.Series([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
