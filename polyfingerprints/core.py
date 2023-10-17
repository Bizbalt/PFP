from typing import Optional, List, Tuple
import numpy as np
from .fingerprints import (
    create_RDKFingerprint,
    merge_bit_fingerprints,
    weight_sum_fingerprints,
    reduce_fp_set,
)
from ._types import FingerprintFunction, PfpData
from .utils import repeating_unit_combinations, calc_polymer_shares


def create_pfp(
    repeating_units: dict[str, float],
    mol_weight: float,
    start: Optional[str] = None,
    end: Optional[str] = None,
    intersection_fp_size: Optional[int] = 2048,
    enhanced_sum_fp_size: Optional[int] = 2048,
    enhanced_fp_functions: Optional[List[FingerprintFunction]] = None,
) -> np.ndarray:
    """Creates a Polyfingerprint

    Args:
        repeating_units (Dict[str: float]): Dictionary containing the
            SMILES representation of each repeating unit and the
            corresponding relative amount.
        mol_weight (float): Molecular weight of the polymer.
        start (Optional[str], optional): start of the polymer. Defaults to hydrogen.
        end (Optional[str], optional): end of the polymer. Defaults to hydrogen.
        intersection_fp_size (Optional[int], optional): Size of the fingerprint
            used to create the intersection fingerprint. Defaults to 2048.
        enhanced_sum_fp_size (Optional[int], optional): Size of the fingerprint
            used to create the enhanced sum fingerprint. Defaults to 2048.
        enhanced_fp_functions (Optional[Callable[[List[str], int], List[np.ndarray]]], optional):
            List of functions used to create the enhanced fingerprints.
            Defaults to [create_RDKFingerprint].

    Returns:
        np.ndarray[[-1], float]:
            Polyfingerprint with length intersection_fp_size + (2 * enhanced_sum_fp_size)*len(enhanced_fp_functions)

    """

    if not start:
        start = "[H]"
    if not end:
        end = "[H]"

    if not enhanced_fp_functions:
        enhanced_fp_functions: List[FingerprintFunction] = [create_RDKFingerprint]

    # create intersection fingerprint
    # create all possible combinations of repeating units
    fragment_recomb = repeating_unit_combinations(
        list(repeating_units.keys()), start=start, end=end
    )

    intersection_fp = merge_bit_fingerprints(
        create_RDKFingerprint(fragment_recomb, fp_size=intersection_fp_size)
    )

    ru_fractions, start_end_share = calc_polymer_shares(
        repeating_units, [start, end], mol_weight
    )

    enhanced_sum_fp = np.concatenate(
        [
            np.concatenate(
                [
                    weight_sum_fingerprints(
                        efpf(
                            list(repeating_units.keys()),
                            fp_size=enhanced_sum_fp_size,
                        ),
                        weights=list(ru_fractions.values()),
                    ),
                    weight_sum_fingerprints(
                        efpf(
                            [
                                start,
                                end,
                            ],
                            fp_size=enhanced_sum_fp_size,
                        ),
                        weights=start_end_share,
                    ),
                ]
            )
            for efpf in enhanced_fp_functions
        ]
    )

    return np.concatenate([intersection_fp, enhanced_sum_fp])


def reduce_pfp_in_dataset(pfp_data: List[PfpData]) -> Tuple[List[PfpData], dict]:
    for d in pfp_data:
        if "pfp" not in d or d["pfp"] is None:
            raise ValueError("Missing Polyfingerprint in dataset.")

    fingerprints: List[np.ndarray] = [d["pfp"] for d in pfp_data]
    reduced_fps, mask, reference_fp = reduce_fp_set(fingerprints)
    for d, rfp in zip(pfp_data, reduced_fps):
        d["pfp"] = rfp

    return pfp_data, {
        "mask": mask,
        "reference_fp": reference_fp,
    }
