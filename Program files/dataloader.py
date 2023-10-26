import pandas as pd
import polyfingerprints as pfp
import os
from tkinter import Tk, filedialog
import numpy as np
import datetime
import pickle
from math import isnan
from parameters import RunParameters as Pm


def default_export_path():
    date_and_time = str(datetime.date.today()) + " " + datetime.datetime.now().strftime("%H:%M:%S").replace(":", " ")

    Pm.export_path = os.path.join(
        (os.path.expanduser("~") + "\\Documents\\Jupyter Files\\Cloud Point determination"), date_and_time)
    os.makedirs(Pm.export_path, exist_ok=True)
    print("Files will be saved at {}".format(Pm.export_path))


def choose_dataset():
    root = Tk()

    root.fileName = filedialog.askopenfilename(
        filetypes=(("semicolon-separated values", "*.csv"), ("All files", "*.*")))
    Tk.destroy(self=root)
    if not root.fileName.endswith(".csv"):
        raise Exception("No CSV Found! \n"
                        "Enter proper Excel Sheet separated by semicolon and comma as decimal key \n"
                        "CSV UTF-8 (durch Trennzeichen getrennt)")
    try:
        df = pd.read_csv(root.fileName, sep=";", decimal=",", encoding="utf8")
        print("reading csv table...")
        return df

    except pd.errors.EmptyDataError:
        print("No file found!")


def create_pfp_dataset(reduce=True):
    df = choose_dataset()
    end_groups = df["SMILES_start_group"], df["SMILES_end_group"]
    structure_tuple = ({A1: A2, B1: B2, C1: C2, D1: D2, E1: E2} for A1, A2, B1, B2, C1, C2, D1, D2, E1, E2 in
                       zip(df["SMILES_repeating_unitA"], df["molpercent_repeating_unitA"],
                           df["SMILES_repeating_unitB"],
                           df["molpercent_repeating_unitB"], df["SMILES_repeating_unitC"],
                           df["molpercent_repeating_unitC"], df["SMILES_repeating_unitD"],
                           df["molpercent_repeating_unitD"], df["SMILES_repeating_unitE"],
                           df["molpercent_repeating_unitE"]))  # arbitrarily many repeating units
    molar_weights = df["Mn"]

    print("creating Polymer-Fingerprints from Dataframe...")
    fingerprints = [pfp.create_pfp(
        start=start, end=end,
        repeating_units={smiles: ratio for smiles, ratio in smiles_tuple.items() if not isnan(ratio)},
        mol_weight=weight, intersection_fp_size=Pm.FP_SIZE, enhanced_sum_fp_size=Pm.FP_SIZE,
        enhanced_fp_functions=Pm.pfp_const_type)
        for start, end, smiles_tuple, weight in
        zip(*end_groups, structure_tuple, molar_weights)]

    print("reducing Fingerprint-set...")
    reduced_fp, reduced_fp_mask, reduced_fp_mask2 = pfp.reduce_fp_set(*fingerprints)
    print("fingerprint size for first Layer is %s" % len(reduced_fp[0]))

    p_fp = [reduced_fp, reduced_fp_mask, reduced_fp_mask2]  # save pfp for later
    with open(os.path.join(Pm.export_path, "P_FP_of_model_{}.pickle".format(Pm.model_nr)), "wb") as f:
        pickle.dump(p_fp, f)

    if not reduce:
        df["fp"] = fingerprints  # without reduction
    else:
        df["fp"] = reduced_fp

    return df


def read_out_other_fp_set(df, fp_name):  # read in fp created in an WSL2 environment and saved into csv
    print("reading csv containing " + fp_name + "..." + "\n   search for \"cp_data_w_additional_fps.csv\"")
    df2 = choose_dataset()

    # reverting np's __string__ print-style: rid linebreaks/brackets, whitespace to list, to float-list and to array
    df["fp"] = df2[fp_name].apply(lambda x: np.array(
        [int(x) for x in (x.replace("\n", "").replace("[", "").replace("]", "")).split()]))
    return df
