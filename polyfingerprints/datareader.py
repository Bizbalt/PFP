from typing import List, Optional
import pandas as pd
import numpy as np

from rdkit import Chem

import polyfingerprints as pfp

REPEATING_UNIT_COLUMN_PREFIX = "SMILES_repeating_unit"
MOLPERCENT_RU_COLUMN_PREFIX = "molpercent_repeating_unit"

ADDITIVE_COLUMN_PREFIX = "additive"
ADDITIVE_WP_COLUMN_POSTFIX = "_weight_percent"
ADDITIVE_CONC_COLUMN_POSTFIX = "_concentration_molar"

SMILES_START_GROUP_COL = "SMILES_start_group"
SMILES_END_GROUP_COL = "SMILES_end_group"
MN_COL = "Mn"
import pytorch_lightning


class InvalidSmilesError(Exception):
    pass


def check_additive(additive, weight_percent=None, concentration=None, density=1):
    if pd.isnull(additive):
        return {}

    if pd.isnull(weight_percent):
        weight_percent = None
    if pd.isnull(concentration):
        concentration = None

    mol = Chem.MolFromSmiles(additive)
    if mol is None:
        raise InvalidSmilesError(f"additive {additive} is not a valid SMILES")

    if weight_percent is not None:
        conc = weight_percent / Chem.Descriptors.MolWt(mol) * 1000 * density
    if concentration is not None:
        if weight_percent is not None:
            if not np.isclose(concentration, conc, rtol=1e-2, atol=1e-3):
                raise ValueError(
                    f"concentration {concentration} and weight_percent {weight_percent} (conc={conc}) do not match"
                )

        conc = concentration

    # seperate cations and anions

    frags = list(Chem.GetMolFrags(mol, asMols=True))
    smiles = [Chem.MolToSmiles(frag) for frag in frags]
    unique_smiles = {}
    for s in smiles:
        if s not in unique_smiles:
            unique_smiles[s] = 1
        else:
            unique_smiles[s] += 1

    return {k: v * conc for k, v in unique_smiles.items()}


def expand_data(
    sampledf,
    ignored_columns: Optional[List[str]] = None,
    density=1,
    repeating_unit_column_prefix=REPEATING_UNIT_COLUMN_PREFIX,
    molpercent_ru_column_prefix=MOLPERCENT_RU_COLUMN_PREFIX,
    additive_column_prefix=ADDITIVE_COLUMN_PREFIX,
    additive_wp_column_postfix=ADDITIVE_WP_COLUMN_POSTFIX,
    additive_conc_column_postfix=ADDITIVE_CONC_COLUMN_POSTFIX,
    smiles_start_group_col=SMILES_START_GROUP_COL,
    smiles_end_group_col=SMILES_END_GROUP_COL,
    mn_col=MN_COL,
):
    if mn_col not in sampledf.columns:
        raise ValueError(f"Column '{mn_col}' not found in excel file.")
    if ignored_columns is None:
        ignored_columns = []

    # drop ingored columns
    for c in ignored_columns:
        if c in sampledf.columns:
            sampledf = sampledf.drop(columns=[c])

    unused_columns = list(sampledf.columns)
    unused_columns.remove(mn_col)

    # remove samples with out mn
    index_with_nan_mn = sampledf.index[sampledf[mn_col].isna()].tolist()
    sampledf = sampledf.drop(index=index_with_nan_mn)

    # extract repeating units
    repeating_unit_columns = []
    for col in sampledf.columns:
        if not col.startswith(repeating_unit_column_prefix):
            continue
        c = col[len(repeating_unit_column_prefix) :]
        if molpercent_ru_column_prefix + c not in sampledf.columns:
            pfp.PFPLOGGER.warning(
                f"Missing molpercent column for repeating unit {c}. Skipping this repeating unit and using its columns as datapoint"
            )
            continue
        unused_columns.remove(repeating_unit_column_prefix + c)
        unused_columns.remove(molpercent_ru_column_prefix + c)
        repeating_unit_columns.append(c)

    unique_rus = set()
    for c in repeating_unit_columns:
        notnanru = sampledf[repeating_unit_column_prefix + c][
            ~sampledf[repeating_unit_column_prefix + c].isna()
        ]
        if len(notnanru) > 0:
            unique_rus.add(notnanru.unique()[0])
    # check if all repeating units valid
    invalid_rus = [s for s in unique_rus if not pfp.test_polymer_smiles(s)]
    invlaid_rus_indices = []
    if invalid_rus:
        for c in repeating_unit_columns:
            invlaid_rus_indices.extend(
                sampledf.index[
                    sampledf[repeating_unit_column_prefix + c].isin(invalid_rus)
                ].tolist()
            )
    sampledf = sampledf.drop(index=invlaid_rus_indices)

    if smiles_start_group_col not in sampledf.columns:
        sampledf[smiles_start_group_col] = "[H]"
    if smiles_start_group_col in unused_columns:
        unused_columns.remove(smiles_start_group_col)
    unique_start_groups = set(sampledf[smiles_start_group_col].unique())
    invalid_start_groups = [
        s for s in unique_start_groups if not pfp.test_startgroup(s)
    ]
    invalid_start_groups_indices = []
    if invalid_start_groups:
        invalid_start_groups_indices = sampledf.index[
            sampledf[smiles_start_group_col].isin(invalid_start_groups)
        ].tolist()
        sampledf = sampledf.drop(index=invalid_start_groups_indices)

    if smiles_end_group_col not in sampledf.columns:
        sampledf[smiles_end_group_col] = "[H]"
    if smiles_end_group_col in unused_columns:
        unused_columns.remove(smiles_end_group_col)
    unique_end_groups = set(sampledf[smiles_end_group_col].unique())
    invalid_end_groups = [s for s in unique_end_groups if not pfp.test_endgroup(s)]
    invalid_end_groups_indices = []
    if invalid_end_groups:
        invalid_end_groups_indices = sampledf.index[
            sampledf[smiles_end_group_col].isin(invalid_end_groups)
        ].tolist()
        sampledf = sampledf.drop(index=invalid_end_groups_indices)

    additive_columns = []
    for col in sampledf.columns:
        if not col.startswith(additive_column_prefix):
            # skip non-additive columns
            continue
        if col.endswith(additive_conc_column_postfix) or col.endswith(
            additive_wp_column_postfix
        ):
            # skip concentration and weight percent columns
            continue
        c = col[len(additive_column_prefix) :]
        if not c:
            # skip unnamed additive columns
            continue
        molar_c_col = f"{additive_column_prefix}{c}{additive_conc_column_postfix}"
        wp_col = f"{additive_column_prefix}{c}{additive_wp_column_postfix}"

        if wp_col in unused_columns:
            unused_columns.remove(wp_col)
        if wp_col not in sampledf.columns:
            sampledf[wp_col] = np.nan

        if molar_c_col in unused_columns:
            unused_columns.remove(molar_c_col)
        if molar_c_col not in sampledf.columns:
            sampledf[molar_c_col] = np.nan
        additive_columns.append(c)
        if col in unused_columns:
            unused_columns.remove(col)

    # check additive columns
    removed_because_no_additive_amount = []
    for ac in additive_columns:
        additive_not_nan = sampledf[~sampledf[additive_column_prefix + ac].isna()]
        additive_molpercent = additive_not_nan[
            additive_column_prefix + ac + additive_wp_column_postfix
        ]
        additive_weightpercent = additive_not_nan[
            additive_column_prefix + ac + additive_conc_column_postfix
        ]
        # inces of rows where no additive amount is given in additive_molpercent or additive_weightpercent
        removed_because_no_additive_amount = additive_not_nan.index[
            additive_molpercent.isna() & additive_weightpercent.isna()
        ].tolist()
    sampledf = sampledf.drop(index=set(removed_because_no_additive_amount))

    # split additives into
    additive_splits = {i: {} for i in sampledf.index}
    removed_because_invalid_additive_smiles = []
    removed_because_invalid_additive_amount = []
    for index, row in sampledf.iterrows():
        for ac in additive_columns:
            additive_smiles = row[additive_column_prefix + ac]
            additive_weightpercent = row[
                additive_column_prefix + ac + additive_wp_column_postfix
            ]
            additive_conc = row[
                additive_column_prefix + ac + additive_conc_column_postfix
            ]
            try:
                additive_split = check_additive(
                    additive_smiles,
                    weight_percent=additive_weightpercent,
                    concentration=additive_conc,
                    density=density,
                )
                for smiles, amount in additive_split.items():
                    if smiles in additive_splits[index]:
                        additive_splits[index][smiles] += amount
                    else:
                        additive_splits[index][smiles] = amount
            except InvalidSmilesError:
                pfp.PFPLOGGER.exception(
                    f"Invalid SMILES {additive_smiles} in row {index}. Skipping additive."
                )
                removed_because_invalid_additive_smiles.append(index)
            except ValueError as e:
                pfp.PFPLOGGER.warning(str(e) + f" in row {index}. Skipping additive.")
                removed_because_invalid_additive_amount.append(index)

    sampledf = sampledf.drop(
        index=set(
            removed_because_invalid_additive_smiles
            + removed_because_invalid_additive_amount
        )
    )

    for ac in additive_columns:
        sampledf = sampledf.drop(columns=[additive_column_prefix + ac])
        sampledf = sampledf.drop(
            columns=[additive_column_prefix + ac + additive_conc_column_postfix]
        )
        sampledf = sampledf.drop(
            columns=[additive_column_prefix + ac + additive_wp_column_postfix]
        )

    unique_additives = set()
    for split in additive_splits.values():
        unique_additives.update(split.keys())
    unique_additives = list(unique_additives)

    additive_cols = []
    for i, additive in enumerate(unique_additives):
        sampledf[f"additive_{i}"] = 0.0
        additive_cols.append(f"additive_{i}")

    for index, row in sampledf.iterrows():
        additive_split = additive_splits[index]
        for smiles, amount in additive_split.items():
            sampledf.loc[index, f"additive_{unique_additives.index(smiles)}"] = amount

    numerical_columns = []
    categorical_columns = []
    for col in unused_columns + additive_cols:
        if pfp.utils.test_categorical(sampledf[col]):
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)

    missing_numerical_indices = {}
    for col in numerical_columns:
        nan_indices = sampledf.index[sampledf[col].isna()].tolist()
        sampledf = sampledf.drop(index=nan_indices)
        if len(nan_indices) > 0:
            missing_numerical_indices[col] = nan_indices

    categorical_unique_values = {}
    for catcol in categorical_columns:
        unique_values = list(sampledf[catcol].unique())
        categorical_unique_values[catcol] = unique_values

    info = {
        "repeating_unit_columns": [
            f"{repeating_unit_column_prefix}{c}" for c in repeating_unit_columns
        ],
        "molpercent_repeating_unit_columns": [
            f"{molpercent_ru_column_prefix}{c}" for c in repeating_unit_columns
        ],
        "additive_columns": additive_columns,
        "missing_additive_amount": removed_because_no_additive_amount,
        "invalid_end_groups_indices": invalid_end_groups_indices,
        "invalid_start_groups_indices": invalid_start_groups_indices,
        "invalid_repeating_unit_indices": invlaid_rus_indices,
        "invalid_additive_smiles_indices": removed_because_invalid_additive_smiles,
        "invalid_additive_amount_indices": removed_because_invalid_additive_amount,
        "missing_mn_indices": index_with_nan_mn,
        "additives": unique_additives,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "categorical_unique_values": categorical_unique_values,
        "missing_numerical_indices": missing_numerical_indices,
    }

    return sampledf, info


def excel_to_data(path: str, ignored_columns: Optional[List[str]] = None):
    sampledf = pd.read_excel(path, sheet_name="samples")

    datadf = pd.read_excel(path, sheet_name="data")
    datadf = datadf.set_index("property")
    density = datadf.loc["density"]["value"]
    return expand_data(sampledf, ignored_columns=ignored_columns, density=density)
