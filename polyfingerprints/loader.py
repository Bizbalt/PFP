from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

from ._types import PfpData, FingerprintFunction
from .core import create_pfp
from .utils import test_polymer_smiles, test_endgroup, test_startgroup
from .logger import PFPLOGGER


def data_encoder_picker(
    data: pd.Series,
):
    """Picks the right encoder for the given data."""
    strtype = str(data.dtype)
    if strtype == "object":
        # one hot encoding
        uniques = data.unique().tolist()
        length = len(uniques)

        def _oh_ecoder(x):
            arr = np.zeros(length + 1)
            try:
                index = uniques.index(x)
                arr[index] = 1
            except ValueError:
                arr[-1] = 1
            return arr

        return _oh_ecoder

    elif (
        strtype.startswith("float")
        or strtype.startswith("int")
        or strtype.startswith("bool")
    ):
        return lambda x: np.array([x], dtype=np.float32)

    raise ValueError(f"Data type {strtype} not supported.")


def df_loader(
    df: pd.DataFrame,
    repeating_unit_columns: List[Tuple[str, str]],
    mw_column: str,
    start_group_column: Optional[str] = None,
    end_group_column: Optional[str] = None,
    y: Optional[Union[str, List[str]]] = None,
    intersection_fp_size: int | None = 2048,
    enhanced_sum_fp_size: int | None = 2048,
    enhanced_fp_functions: List[FingerprintFunction] | None = None,
    additional_columns: Optional[List[str]] = None,
):
    """Loads the data to create a Polyfingerprint from a pandas dataframe.

    Args:
        df (str): Path to the csv file.
        repeating_unit_columns (Tuple(Tuple[str,str])): Tuple of tuples
            containing the column names of the SMILES representation for
            each repeating unit and the corresponding relative amount.
        mw_column (str): Name of the column containing the molecular weight.
        y (Optional[str]): Name of the column containing the target values.
    """
    if additional_columns is None:
        additional_columns = []

    # check df:
    colstofind = [mw_column] + additional_columns
    for smiles, amount in repeating_unit_columns:
        colstofind.append(smiles)
        colstofind.append(amount)

    if y:
        if isinstance(y, str):
            y = [y]
        colstofind.extend(y)
    else:
        y = []
    for _y in y:
        if _y in additional_columns:
            additional_columns.remove(_y)

    if start_group_column:
        colstofind.append(start_group_column)
    if end_group_column:
        colstofind.append(end_group_column)

    # check if all columns are in df

    missing_cols = list(set(colstofind) - set(df.columns))
    if len(missing_cols) > 0:
        raise ValueError(
            f"Column(s) '{', '.join(missing_cols)}' not found in csv file."
        )

    # print all other columns
    for col in df.columns:
        if col not in colstofind:
            print(f"Warning: Column '{col}' not used.")

    # additional_data_types
    additional_data_encoder = {
        col: data_encoder_picker(df[col]) for col in additional_columns
    }

    alldata: List[PfpData] = []
    additional_data_indices = None
    for _, rowdata in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
        repeatingunits: Dict[str, float] = {}
        for smiles, amount in repeating_unit_columns:
            # skip if no smiles or amount are None or NaN
            if (
                not rowdata[smiles]
                or not rowdata[amount]
                or np.isnan(rowdata[amount])
                or np.isnan(rowdata[mw_column])
            ):
                continue
            if rowdata[smiles] in repeatingunits:
                repeatingunits[rowdata[smiles]] += rowdata[amount]
            else:
                repeatingunits[rowdata[smiles]] = rowdata[amount]

        # skip if no repeating units were found
        if not repeatingunits:
            continue

        # check repeating units
        invalid_ru = False
        for ru in repeatingunits.keys():
            if not test_polymer_smiles(ru):
                PFPLOGGER.warning(f"Invalid SMILES string for repeating unit: {ru}")
                invalid_ru = True
        if invalid_ru:
            continue

        # normalize amounts
        total_amount = sum([ru for ru in repeatingunits.values()])
        for ru in repeatingunits.keys():
            repeatingunits[ru] = repeatingunits[ru] / total_amount

        dy = None
        if y:
            dy = rowdata[y].values.astype(np.float32)
            if any(np.isnan(dy)):
                dy = None

        start_group = None
        if start_group_column:
            start_group = rowdata[start_group_column]

        if start_group and not test_startgroup(start_group):
            PFPLOGGER.warning(f"Invalid SMILES string for start group: {start_group}")
            continue

        end_group = None
        if end_group_column:
            end_group = rowdata[end_group_column]

        if end_group and not test_endgroup(end_group):
            PFPLOGGER.warning(f"Invalid SMILES string for end group: {end_group}")
            continue

        _additional_data = [
            additional_data_encoder[col](rowdata[col])
            for col in additional_data_encoder
        ]
        if additional_data_indices is None:
            additional_data_indices = np.array(
                [0] + [len(x) for x in _additional_data]
            ).cumsum()[:-1]
        pfpdat = PfpData(
            repeating_units=repeatingunits,
            y=dy,
            mw=rowdata[mw_column],
            startgroup=start_group,
            endgroup=end_group,
            additional_data_keys=additional_columns,
            additional_data_indices=additional_data_indices,
            additional_data=np.concatenate(_additional_data),
        )
        alldata.append(pfpdat)

    for d in tqdm(alldata, total=len(alldata), desc="Creating Polyfingerprints"):
        d["pfp"] = create_pfp(
            repeating_units=d["repeating_units"],
            mol_weight=d["mw"],
            start=d["startgroup"],
            end=d["endgroup"],
            intersection_fp_size=intersection_fp_size,
            enhanced_sum_fp_size=enhanced_sum_fp_size,
            enhanced_fp_functions=enhanced_fp_functions,
        )
    return alldata


def csv_loader(
    csv: str,
    repeating_unit_columns: List[Tuple[str, str]],
    mw_column: str,
    start_group_column: Optional[str] = None,
    end_group_column: Optional[str] = None,
    y: Optional[Union[str, List[str]]] = None,
    intersection_fp_size: int | None = 2048,
    enhanced_sum_fp_size: int | None = 2048,
    enhanced_fp_functions: List[FingerprintFunction] | None = None,
    additional_columns: Optional[List[str]] = None,
    **kwargs,
):
    """Loads the data to create a Polyfingerprint from a csv file.

    Args:
        csv (str): Path to the csv file.
        repeating_unit_columns (List(Tuple[str,str])): List of tuples
            containing the column names of the SMILES representation for
            each repeating unit and the corresponding relativ amount.
        mw_column (str): Name of the column containing the molecular weight.
        y (Optional[str]): Name of the column containing the target values.

        kwargs: Keyword arguments passed to pandas.read_csv to load
            the csv file, for more information see:
            https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    """
    df = pd.read_csv(csv, **kwargs)
    return df_loader(
        df=df,
        repeating_unit_columns=repeating_unit_columns,
        mw_column=mw_column,
        start_group_column=start_group_column,
        end_group_column=end_group_column,
        y=y,
        intersection_fp_size=intersection_fp_size,
        enhanced_sum_fp_size=enhanced_sum_fp_size,
        enhanced_fp_functions=enhanced_fp_functions,
        additional_columns=additional_columns,
    )


def to_input_output_data(fpdata: List[PfpData]) -> Tuple[np.ndarray, np.ndarray]:
    input_data = np.array(
        [np.concatenate([d["pfp"], d["additional_data"]]) for d in fpdata]
    )
    output_data = np.array([d["y"] for d in fpdata])

    return input_data, output_data
