from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

from ._types import PfpData, FingerprintFunction
from .core import create_pfp


def csv_loader(
    csv: str,
    repeating_unit_columns: List[Tuple[str, str]],
    mw_column: str,
    start_group_column: Optional[str] = None,
    end_group_column: Optional[str] = None,
    y: Optional[str] = None,
    intersection_fp_size: int | None = 2048,
    enhanced_sum_fp_size: int | None = 2048,
    enhanced_fp_functions: List[FingerprintFunction] | None = None,
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

    # check df:
    colstofind = [smiles for (smiles, amount) in repeating_unit_columns] + [mw_column]

    if y:
        colstofind.append(y)
    if start_group_column:
        colstofind.append(start_group_column)
    if end_group_column:
        colstofind.append(end_group_column)

    # check if all columns are in df
    for col in colstofind:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in csv file.")

    # print all other columns
    for col in df.columns:
        if col not in colstofind:
            print(f"Warning: Column '{col}' not used.")

    alldata: List[PfpData] = []
    for _, rowdata in df.iterrows():
        repeatingunits: Dict[str, float] = {}
        for smiles, amount in repeating_unit_columns:
            # skip if no smiles or amount are None or NaN
            if not rowdata[smiles] or not rowdata[amount] or np.isnan(rowdata[amount]):
                continue
            if rowdata[smiles] in repeatingunits:
                repeatingunits[rowdata[smiles]] += rowdata[amount]
            else:
                repeatingunits[rowdata[smiles]] = rowdata[amount]

        # skip if no repeating units were found
        if not repeatingunits:
            continue

        # normalize amounts
        total_amount = sum([ru for ru in repeatingunits.values()])
        for ru in repeatingunits.keys():
            repeatingunits[ru] = repeatingunits[ru] / total_amount

        dy = None
        if y:
            dy = rowdata[y]
            if np.isnan(dy):
                dy = None
        start_group = None
        if start_group_column:
            start_group = rowdata[start_group_column]
        end_group = None
        if end_group_column:
            end_group = rowdata[end_group_column]

        pfpdat = PfpData(
            repeating_units=repeatingunits,
            y=dy,
            mw=rowdata[mw_column],
            startgroup=start_group,
            endgroup=end_group,
        )
        alldata.append(pfpdat)

    for d in alldata:
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
