from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
from .utils import polymol_fom_smiles
from .logger import PFPLOGGER


def create_AtomicPairFingerprint(
    smiles_s: List[str], fp_size: int = 2048, complement: bool = False
) -> List[np.ndarray]:
    """
    Creates a list of AtomicPairFingerprints from a list of SMILES strings.

    Args:
        smiles_s (List[str]): List of SMILES strings.
        fp_size (int, optional): Size of the fingerprint. Defaults to 2048.
        complement (bool, optional): If True, the SMILES strings are patched
            with [H][H] at the beginning and end, should be true if working
            with repeating unit smiles to hide the radicals. Defaults to False.
    Returns:
        List[np.array]: List of AtomicPairFingerprints.
    """
    if complement:
        smiles_list = [("[H]{}[H]".format(smiles)) for smiles in smiles_s]
    else:
        smiles_list = smiles_s

    fingerprint_s = [
        np.array(
            list(
                Pairs.GetHashedAtomPairFingerprint(
                    polymol_fom_smiles(smiles), nBits=fp_size
                )
            )
        )
        for smiles in smiles_list
    ]
    return fingerprint_s


def create_RDKFingerprint(
    smiles_s: List[str], fp_size: int = 2048, complement: bool = False
) -> List[np.ndarray]:
    """
    Creates a list of RDKFingerprints from a list of SMILES strings.

    Args:
        smiles_s (List[str]): List of SMILES strings.
        fp_size (int, optional): Size of the fingerprint. Defaults to 2048.
        complement (bool, optional): If True, the SMILES strings are patched
            with [H][H] at the beginning and end, should be true if working
            with repeating unit smiles to hide the radicals. Defaults to False.
    Returns:
        List[np.array]: List of RDKFingerprints.
    """
    if complement:
        smiles_list = [("[H]{}[H]".format(smiles)) for smiles in smiles_s]
    else:
        smiles_list = smiles_s

    fingerprint_s = [
        np.array(Chem.RDKFingerprint(polymol_fom_smiles(smiles), fpSize=fp_size))
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
    fingerprints: List[np.ndarray[[-1], float]], weights: List[float]
) -> np.ndarray[[-1], float]:
    """sums up a list of fingerprints with weights and returns the weighted sum fingerprint

    Args:
        fingerprints (List[np.ndarray[[-1], float]]): list of fingerprints
        weights (List[float]): list of weights

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

    # make sure all weights have the same length as the number of fingerprints
    if len(weights) != len(fingerprints):
        raise ValueError(
            "Number of weights must be the same as the number of fingerprints."
        )

    fingerprints = [fp * weights[i] for i, fp in enumerate(fingerprints)]

    return np.sum(fingerprints, axis=0)


def reduce_fp_set(
    fingerprints: List[np.ndarray[[-1], float]]
) -> Tuple[
    List[np.ndarray[[-1], float]], np.ndarray[[-1], bool], np.ndarray[[-1], float]
]:
    """
    Reduces a set of fingerprints by removing positions that have identical values across all provided fingerprints.

    Given multiple fingerprints, this function identifies and discards the positions (features)
    that are consistent among all the fingerprints. Such consistent positions might be deemed as less informative
    for certain analyses since they don't contribute to the distinction between fingerprints.

    Args:
        fingerprints (List[np.ndarray]): A list of 1D numpy arrays representing the fingerprints.

    Returns:
        Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
            - List[np.ndarray]: A list of reduced 1D numpy arrays where identical positions across all fingerprints have been removed.
            - np.ndarray: A boolean mask indicating the positions that were kept (False) or removed (True).
            - np.ndarray: The first fingerprint from the input, prior to any reductions.

    Note:
        This function assumes that all input fingerprints are of the same length. It also logs the percentage
        reduction in fingerprint size, which might be useful for understanding the impact of the reduction on the data.

    Examples:
        >>> fp1 = np.array([0.2, 0.5, 0.1])
        >>> fp2 = np.array([0.2, 0.6, 0.1])
        >>> fp3 = np.array([0.2, 0.7, 0.1])
        >>> reduced_fps, mask, reference_fp = reduce_fp_set(fp1, fp2, fp3)
        >>> print(reduced_fps)  # Lists of reduced fingerprints
        np.array([[0.5], [0.6], [0.7]])
        >>> print(mask)         # Mask used for reduction
        np.array([True, False, True])
        >>> print(reference_fp) # Reference fingerprint
        np.array([0.2, 0.5, 0.1])
    """

    # Stack the fingerprints to identify common positions
    stacked_fps = np.stack(fingerprints)

    # Identify positions that are the same across all fingerprints
    same_positions = np.all(stacked_fps == stacked_fps[0, :], axis=0)

    # Create a mask to keep positions that are not the same
    mask = same_positions

    # Reduce the fingerprints using the mask
    reduced_fps = [fp[~mask] for fp in fingerprints]

    PFPLOGGER.info(
        "reduced size by {0:.0f}%".format(
            (1 - (len(reduced_fps[0]) / len(fingerprints[0]))) * 100
        )
    )

    # Return the reduced fingerprints, the mask, and a reference fingerprint
    return reduced_fps, mask, fingerprints[0].copy()


def apply_reduction_fp_set(
    fingerprints: List[np.ndarray[[-1], float]],
    mask: np.ndarray[[-1], bool],
    reference_fp: np.ndarray[[-1], float],
) -> List[np.ndarray[[-1], float]]:
    """
    Given multiple fingerprints, this function discards the same positions (features)
    like the reduce_fp_set function did with a prior set, thus rendering the fingerprints comparable.

    Args:
        fingerprints (List[np.ndarray]): A list of 1D numpy arrays representing the fingerprints.
        mask (np.ndarray): A boolean mask indicating the positions that were kept (False) or removed (True).
        reference_fp (np.ndarray): The first fingerprint from the input, prior to any reductions (needed for the information loss calculation).

    Returns:
        Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
            - List[np.ndarray]: A list of 1D numpy arrays reduced by the mask.

    Note:
        This function assumes that all input fingerprints are of the same length and of the same length as the mask (prior set).
        It also logs the loss which depicts the percentage of information lost by the reduction, respectively positions
         that were new but removed for comparability.

    Examples:
        >>> fp4 = np.array([0.2, 0.6, 0.2])
        >>> fp5 = np.array([0.2, 0.7, 0.1])
        >>> mask = np.array([True, False, True])
        >>> reference_fp = np.array([0.2, 0.5, 0.1])
        >>> reduced_fps= apply_reduction_fp_set([fp4, fp5], mask, reference_fp)
        >>> print(reduced_fps)  # Lists of reduced fingerprints
        np.array([[[0.6], [0.7]])
    """
    is_out_count = mask.sum()
    if is_out_count == 0:
        return fingerprints

    stacked_fps = np.stack(fingerprints)
    should_out = (stacked_fps == reference_fp) & mask
    should_out_count = should_out.sum(1)
    min_should_out = should_out_count.min()
    mean_should_out = should_out_count.mean()
    print(
        (1 - mean_should_out / is_out_count) * 100,
        (1 - min_should_out / is_out_count) * 100,
    )

    PFPLOGGER.info(
        "mean reduction loss is {0:.0f}% with the highest loss per fingerprint beeing {1:.0f}%",
        (1 - mean_should_out / is_out_count) * 100,
        (1 - min_should_out / is_out_count) * 100,
    )

    return [new_fp[~mask] for new_fp in fingerprints]
