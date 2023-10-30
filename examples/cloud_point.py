import pandas as pd
import sys
import os
import numpy as np
from warnings import warn
from rdkit import Chem
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polyfingerprints as pfp
from pprint import pprint

RAW_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cloud_points_data.csv"
)
PARSED_CSV_PATH = os.path.join(
    os.path.dirname(RAW_CSV_PATH), "cloud_points_data_parsed.csv"
)
PARSED_INFO_JSON_PATH = os.path.join(
    os.path.dirname(RAW_CSV_PATH), "cloud_points_data_info.json"
)


def check_additive(additive, weight_percent=None, concentration=None):
    if pd.isnull(additive):
        return {}

    mol = Chem.MolFromSmiles(additive)
    if mol is None:
        raise ValueError(f"additive {additive} is not a valid SMILES")

    if weight_percent is not None:
        conc = weight_percent / Chem.Descriptors.MolWt(mol) * 1000

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


def raw_data_to_dataset():
    df = pd.read_csv(RAW_CSV_PATH, sep=";", decimal=",")
    out_data = []
    unused_columns = list(df.columns)
    ignored_columns = [
        "reference",
        "polymer_type",
        "polymer_type_style",
        "polymer_architecture",
        "polymerisation_type",
        "Mw",
        "PDI",
        "mass_characterisation_method",
        "mass_characterisation_standart",
        "N/A",
        "identifier",
        "comment",
        "tacticity",
        "rating",
    ]
    for c in ignored_columns:
        if c in unused_columns:
            unused_columns.remove(c)

    repeating_unit_columns = []
    for c in "ABCDEFGHIJKL":
        if (
            "SMILES_repeating_unit" + c in df.columns
            and "molpercent_repeating_unit" + c in df.columns
        ):
            repeating_unit_columns.append(c)

            unused_columns.remove("SMILES_repeating_unit" + c)
            unused_columns.remove("molpercent_repeating_unit" + c)

    additive_columns = []
    for c in [str(x) for x in range(1, 20)]:
        if (
            "additive" + c in df.columns
            and f"additive{c}_concentration_weight_percent" in df.columns
        ):
            unused_columns.remove("additive" + c + "_concentration_molar")
            unused_columns.remove("additive" + c)
            if f"additive{c}_concentration_weight_percent" in unused_columns:
                unused_columns.remove(f"additive{c}_concentration_weight_percent")

            additive_columns.append(c)
    # replace nans in pH with 7
    df["pH"] = df["pH"].fillna(7)
    unused_columns.remove("pH")

    df["def_type"] = df["def_type"].fillna("0.1")

    DEF_TYPE_MAP = {
        "A": 0.1,
        "B": 0.2,
        "C": 0.5,
        "DSC": 0.01,
    }
    df["def_type"] = df["def_type"].apply(
        lambda x: float(
            (DEF_TYPE_MAP[x] if x in DEF_TYPE_MAP else x.replace(",", "."))
        ),
    )

    unused_columns.remove("def_type")

    df["SMILES_start_group"] = df["SMILES_start_group"].fillna("[H]")
    unused_columns.remove("SMILES_start_group")

    df["SMILES_end_group"] = df["SMILES_end_group"].fillna("[H]")
    unused_columns.remove("SMILES_end_group")

    unused_columns.remove("Mn")
    unused_columns.remove("cloud_point")

    # wt%|mass fraction and mass concentration in g/mL are approximately the same for water
    df["poly_conc"] = df["polymer_concentration_wpercent"]
    # set poly_conc to polymer_concentration_mass_conc where it is NaN
    df["poly_conc"] = df["poly_conc"].fillna(df["polymer_concentration_mass_conc"])
    unused_columns.remove("polymer_concentration_wpercent")
    unused_columns.remove("polymer_concentration_mass_conc")

    all_additives = []
    for rowindex, row in df.iterrows():
        rowdat = {}
        for c in repeating_unit_columns:
            rowdat["SMILES_repeating_unit" + c] = row["SMILES_repeating_unit" + c]
            rowdat["molpercent_repeating_unit" + c] = row[
                "molpercent_repeating_unit" + c
            ]

        rowdat["pH"] = row["pH"]
        rowdat["def_type"] = row["def_type"]
        rowdat["SMILES_start_group"] = row["SMILES_start_group"]
        rowdat["SMILES_end_group"] = row["SMILES_end_group"]
        rowdat["Mn"] = row["Mn"]
        rowdat["poly_conc"] = row["poly_conc"]
        rowdat["cloud_point"] = row["cloud_point"]

        additives = {}
        for c in additive_columns:
            conc = row["additive" + c + "_concentration_molar"]
            w_perc = row[f"additive{c}_concentration_weight_percent"]
            if np.isnan(conc) and np.isnan(w_perc):
                if not pd.isnull(row["additive" + c]):
                    warn(
                        f"additive {rowindex} {row['additive' + c]} has no concentration"
                    )
                continue

            for a, aconc in check_additive(
                row["additive" + c],
                weight_percent=w_perc if not np.isnan(w_perc) else None,
                concentration=conc if not np.isnan(conc) else None,
            ).items():
                if a in additives:
                    additives[a] += aconc
                else:
                    additives[a] = aconc
                if a not in all_additives:
                    all_additives.append(a)

        for k, v in additives.items():
            rowdat[f"additive_{all_additives.index(k)}"] = v
        out_data.append(rowdat)

    parseddf = pd.DataFrame(out_data)
    for i, _ in enumerate(all_additives):
        parseddf[f"additive_{i}"] = parseddf[f"additive_{i}"].fillna(0)

    parseddf["log_Mn"] = np.log10(parseddf["Mn"]) / 6
    parseddf.to_csv(
        PARSED_CSV_PATH,
        index=False,
    )

    # infos
    all_ru_mp = parseddf[
        [c for c in parseddf.columns if "molpercent_repeating_unit" in c]
    ].values.flatten()
    all_ru = parseddf[
        [c for c in parseddf.columns if "SMILES_repeating_unit" in c]
    ].values.flatten()[~np.isnan(all_ru_mp)]

    info = {
        "def_types": list(df["def_type"].unique()),
        "num_unique_ru": len(set(all_ru)),
        # "unique_ru": list(set(all_ru)),
        "num_unique_additives": len(all_additives),
        "additives": all_additives,
    }

    with open(PARSED_INFO_JSON_PATH, "w") as f:
        json.dump(info, f, indent=4)


def generate_data():
    with open(PARSED_INFO_JSON_PATH, "r") as f:
        info = json.load(f)

    data = pfp.loader.csv_loader(
        PARSED_CSV_PATH,
        repeating_unit_columns=[
            ("SMILES_repeating_unit" + a, "molpercent_repeating_unit" + a)
            for a in "ABCDE"
        ],
        y="cloud_point",
        sep=";",
        decimal=",",
        mw_column="Mn",
        start_group_column="SMILES_start_group",
        end_group_column="SMILES_end_group",
        additional_columns=[
            "pH",
            "additive1",
            "additive2",
            "additive1_concentration_molar",
            "additive2_concentration_molar",
            "def_type",
        ],
    )

    data, reduced_fp_data = pfp.reduce_pfp_in_dataset(data)

    pprint(data[-10:])
    print("infos")
    infos = {"fp_size": len(data[0]["pfp"])}
    pprint(infos)


if __name__ == "__main__":
    raw_data_to_dataset()
    # generate_data()
