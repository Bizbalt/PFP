from typing import Tuple, List, Optional, Dict, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, MolFromSmiles
import numpy as np
from rdkit import RDLogger


def molar_wt_fromSmiles(smiles: str) -> float:
    return Descriptors.MolWt(polymol_fom_smiles(smiles))


def calc_polymer_shares(
    rep_units: dict[str, float],
    start_end_smiles: List[str],
    total_weight: float,
) -> Tuple[Dict[str, float], List[float]]:
    """calculates the mole fraction of the polymer repeating units and the start end groups

    Args:
        rep_units (Dict[str:float]): Dictionary containing the SMILES representation of each repeating unit and the
            corresponding relative amount.
        ends (List[str]): list of SMILES strings for the start and end groups

    Returns:
        Tuple[Dict[str:float], List[float]]: Tuple containing the mole fraction of the repeating units and the start end groups

    """

    total_weight_minus = total_weight - sum(
        molar_wt_fromSmiles(x) for x in start_end_smiles
    )
    if total_weight_minus < 0:
        raise ValueError(
            "Total weight of the polymer is smaller than the weight of the start and end groups."
        )

    # normalize rep_units, normally this should be done in a previous step, but just in case
    total_amount = sum([ru for ru in rep_units.values()])
    rep_units = {k: v / total_amount for k, v in rep_units.items()}

    abs_amounts_start_end = np.ones(len(start_end_smiles))
    abs_amounts_ru = {k: 1.0 for k in rep_units.keys()}
    for rep_w, chie in rep_units.items():
        rep_m = total_weight_minus * chie
        rep_n = rep_m / molar_wt_fromSmiles(rep_w)
        abs_amounts_ru[rep_w] = rep_n

    total_amounts = sum(abs_amounts_ru.values()) + sum(abs_amounts_start_end)

    return {
        k: v / total_amounts for k, v in abs_amounts_ru.items()
    }, abs_amounts_start_end / total_amounts


def repeating_unit_combinations(
    frag_smiles: List[str], start: Optional[str] = None, end: Optional[str] = None
) -> List[str]:
    """create all possible combinations of repeating units

    Args:
        frag_smiles (List[str]): list of smiles strings for each repeating unit
        start (Optional[str], optional): start of the polymer. Defaults to None.
        end (Optional[str], optional): end of the polymer. Defaults to None.

    Returns:
        List[str]: list of all possible combinations of repeating units as smiles strings

    Example:
        >>> repeating_unnit_combinations(["A", "B"], start="I", end="E")
        ["IAAE", "IBBE", "IABE", "IBAE"]
    """
    frag_smiles_index = np.arange(len(frag_smiles))
    recomb = np.array(
        np.meshgrid(*([frag_smiles_index] * len(frag_smiles_index)))
    ).T.reshape(-1, len(frag_smiles))
    # remove combinations with no -1

    smiles = []
    for r in recomb:
        ruse = r[r >= 0]
        if len(ruse) == 0:
            continue
        smiles.append([frag_smiles[i] for i in r if i != -1])

    recombinations_smiles = ["".join(s) for s in smiles]
    if start:
        recombinations_smiles = [start + s for s in recombinations_smiles]
    if end:
        recombinations_smiles = [s + end for s in recombinations_smiles]
    return recombinations_smiles


def polymol_fom_smiles(smiles: str) -> Mol:
    RDLogger.DisableLog("rdApp.warning")
    mol = MolFromSmiles(smiles)
    RDLogger.EnableLog("rdApp.warning")
    return mol


def test_polymer_smiles(smiles: str) -> bool:
    """Test if a smiles string is a valid monomer representation.
    For this the first and last atom of the molecule must have a a missing bond (radical) like [CH2][CH2](C(=O)(OC))

    Args:
        smiles (str): smiles string of the monomer

    Returns:
        bool: True if the smiles string is a valid monomer representation

    Example:
        >>> test_polymer_smiles("CCC(=O)OC") # has no open ends
        False
        >>> test_polymer_smiles("[CH2][CH]C(=O)OC") # has both open ends, but not at the terminal atoms
        False
        >>> test_polymer_smiles("[CH2][CH](C(=O)(OC))") # has both open ends at the terminal atoms
        True
    """

    try:
        basemol = polymol_fom_smiles(smiles)
        # minimum two radicals
        ori_n_radicals = Descriptors.NumRadicalElectrons(basemol)
        if not ori_n_radicals >= 2:
            return False
        if not basemol:
            return False
        if not polymol_fom_smiles(smiles * 3):
            return False

        hmol = polymol_fom_smiles("[H]" + smiles + "[H]")
        if not hmol:
            return False
        if not Descriptors.NumRadicalElectrons(hmol) == ori_n_radicals - 2:
            return False
        return True
    except Exception:
        return False
