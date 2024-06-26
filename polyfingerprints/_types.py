from typing import List, Tuple, TypedDict, Optional, Union, Callable, Dict
import numpy as np


class PfpData(TypedDict):
    """TypedDict containing the data for a single Polyfingerprint.

    Args:
        repeating_units (Dict[str: float]): Dictionary containing the
            SMILES representation of each repeating unit and the
            corresponding relative amount.
    """

    repeating_units: Dict[str, float]
    startgroup: Optional[str]
    endgroup: Optional[str]
    y: Optional[Union[float, int]]
    mw: float
    pfp: Optional[np.ndarray]


FingerprintFunction = Callable[[List[str], int, bool], List[np.ndarray]]
"""
Type hint for the fingerprint creation functions.

The function expects the following parameters:
    - A list of strings (SMILES strings).
    - An integer (the fingerprint size), which defaults to 2048.
    - A boolean (whether to complement the SMILES strings or not), which defaults to False.

The function returns:
    - A list of numpy arrays (the fingerprints).
"""
