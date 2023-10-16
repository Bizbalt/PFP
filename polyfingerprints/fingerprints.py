from typing import Callable, List
import numpy as np
from rdkit import Chem
from ._types import FingerprintFunction


def create_RDKFingerprint(
    smiles_s: list[str], fp_size: int = 2048, complement: bool = False
) -> list[np.ndarray]:
    """
    Creates a list of RDKFingerprints from a list of SMILES strings.

    Args:
        smiles_s (list[str]): List of SMILES strings.
        fp_size (int, optional): Size of the fingerprint. Defaults to 2048.
        complement (bool, optional): If True, the SMILES strings are patched
            with [H][H] at the beginning and end, should be true if working
            with repeating unit smiles to hide the radicals. Defaults to False.
    Returns:
        list[np.array]: List of RDKFingerprints.
    """
    if complement:
        smiles_list = [("[H]{}[H]".format(smiles)) for smiles in smiles_s]
    else:
        smiles_list = smiles_s

    fingerprint_s = [
        np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=fp_size))
        for smiles in smiles_list
    ]
    return fingerprint_s


def merge_bit_fingerprints(
    fingerprints: List[np.ndarray[[-1], bool]]
) -> np.ndarray[[-1], bool]:
    """merges an arbitrary number of bit fingerprints into one by using the or operator

    Args:
        fingerprints (List[np.ndarray[[-1], bool]]): arbitrary number of bit fingerprints with the same length L

    Returns:
        np.ndarray[[-1], bool]: merged fingerprint with length L
    """

    # flatten
    fingerprints = [fp.flatten() for fp in fingerprints]
    # make sure all fingerprints have the same length
    if len(set([len(fp) for fp in fingerprints])) > 1:
        raise ValueError("All fingerprints must have the same length.")

    # make sure all fingerprints are bit fingerprints
    fingerprints = [fp.astype(bool) for fp in fingerprints]

    # merge fingerprints
    merged_fp = np.stack(fingerprints)
    merged_fp = np.any(merged_fp, axis=0)
    return merged_fp


def weight_sum_fingerprints(
    fingerprints: list[np.ndarray[[-1], float]], weights: list[float]
) -> np.ndarray[[-1], float]:
    """sums up a list of fingerprints with weights and returns the weighted sum fingerprint

    Args:
        fingerprints (list[np.ndarray[[-1], float]]): list of fingerprints
        weights (list[float]): list of weights

    Returns:
        np.ndarray[[-1], float]: weighted sum fingerprint

    Raises:
        ValueError: if the number of weights is not the same as the number of fingerprints
        ValueError: if the fingerprints do not have the same length

    Example:
        >>> weight_sum_fingerprints([np.array([1,2,3]), np.array([4,5,6])], [0.5, 0.5])
        np.array([2.5, 3.5, 4.5])
    """

    # flatten
    fingerprints = [fp.flatten() for fp in fingerprints]
    # make sure all fingerprints have the same length L
    if len(set([len(fp) for fp in fingerprints])) > 1:
        raise ValueError("All fingerprints must have the same length.")

    # make sure all fingerprints are float fingerprints
    fingerprints = [fp.astype(float) for fp in fingerprints]

    # make sure all weights have the same length as the number of fingerprints
    if len(weights) != len(fingerprints):
        raise ValueError(
            "Number of weights must be the same as the number of fingerprints."
        )

    fingerprints = [fp * weights[i] for i, fp in enumerate(fingerprints)]

    return np.sum(fingerprints, axis=0)
