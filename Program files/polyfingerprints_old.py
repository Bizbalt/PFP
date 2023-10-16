import numpy as np
import itertools
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Descriptors


def combinations(*smiles_frags, start, end):
    """returns all possible combinations of substructures
     takes in a number of smiles fragments. Returns a list of possible representations"""
    combination_fps = []
    for mutation in (itertools.permutations(smiles_frags, len(smiles_frags))):
        combination_fps.append(start + "".join(mutation) + end)
    return combination_fps


def concatenator(args):
    """concatenates arrays from a list to single 1D-array"""
    if not type(args) in (list, tuple):
        raise Exception("List, Tuple, or other sequence expected but got %s!" % (type(args)))
    return np.concatenate(args, axis=0)


def merge_subs_fp(*fingerprints):
    """takes in a number of numpy arrays as list and merges to one array containing all substructures"""
    complement_fp = np.zeros(len(fingerprints[0]))
    if len(fingerprints) == 1:
        return fingerprints[0]
    for pos, bit in enumerate(np.array(fingerprints).T):
        if 1 in bit:
            complement_fp[pos] = 1
    return complement_fp


def merge_hashed_ap_fp(*fingerprints):
    """takes in a number of numpy arrays as list and merges to one array containing all substructures
    deprived: because hashed fingerprint bits are ambiguous """
    complement_fp = np.zeros(len(fingerprints[1]))
    for pos, bit in enumerate(np.array(fingerprints).T):
        complement_fp[pos] = max(bit)
    return complement_fp


def create_hashed_ap_fp(*smiles_s, complement=False, fp_size=2048):
    """universe fingerprint creation method from (un)finished smiles codes and returns them as list"""
    if complement:
        smiles_list = [("[H]{}[H]".format(smiles)) for smiles in smiles_s]
    else:
        smiles_list = smiles_s

    fingerprint_s = [np.array(
        list(Pairs.GetHashedAtomPairFingerprint(Chem.MolFromSmiles(smiles), fp_size))) for smiles in smiles_list]
    return fingerprint_s


def create_subs_fp(*smiles_s, complement=False, fp_size=2048):
    """universe fingerprint creation method from (un)finished smiles codes and returns them as list"""
    if complement:
        smiles_list = [("[H]{}[H]".format(smiles)) for smiles in smiles_s]
    else:
        smiles_list = smiles_s

    fingerprint_s = [np.array(
        Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=fp_size)) for smiles in smiles_list]
    return fingerprint_s


def create_morgan_bit_fp(*smiles_s, radius=4, fp_size=2048):
    from rdkit.Chem import AllChem
    """"universe fingerprint creation method returning a list of morgan/ECFP-ish fingerprints"""
    return [np.array(Chem.AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), radius, nBits=fp_size))
        for smiles in smiles_s]


def significance_sigma(fp_rep_arrays, incidence_s):
    """multiplies fingerprint arrays accordingly with occurrence and returns a list of all arrays"""
    sigma = np.zeros(len(fp_rep_arrays[0]))
    for fp_arr, incidence in zip(fp_rep_arrays, incidence_s):
        sigma_part = np.array(fp_arr) * incidence
        sigma += sigma_part
    return [sigma]


def molar_wt_fromSmiles(smiles):
    try:
        wt = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
    except ValueError:
        print("%s is not a proper smiles code!" % smiles)
        raise
    return wt


def calc_ends_share(ends, rep_units, total_weight):
    """calculates the mole fraction of the polymer end groups"""
    total_moles_minus = 0
    total_weight_minus = total_weight - sum(molar_wt_fromSmiles(x) for x in ends.values())
    for rep_w, chie in zip(rep_units.values(), rep_units.keys()):
        rep_m = total_weight_minus * chie
        rep_n = rep_m / molar_wt_fromSmiles(rep_w)
        total_moles_minus += rep_n
    start_end_chie = 1 / total_moles_minus
    return start_end_chie


def subduct_subs_fp(*fingerprints):
    """subtracts bits which are not set as true in the first array, from all following
    subduct takes in multiple arrays as tuple gives out a list
    deprecated: useless for subduction in hashed fingerprints as single bits represent multiple pairs"""
    whet = fingerprints[0]
    fingerprint_s = []
    remnant = []
    for fp in fingerprints[1:]:
        current_s_fp = fp
        for pos, whetstone in enumerate(whet):
            if whetstone == 0:
                if current_s_fp[pos] != 0:
                    remnant.append(current_s_fp[pos])
                current_s_fp[pos] = 0
        fingerprint_s.append(current_s_fp)
        remnant.append("end")
    return fingerprint_s, remnant


def fp_investigator(fp_arr):
    """takes in one array and breaks down numbers into a more visible view"""
    one_s = 0
    zero_s = 0
    for numb in fp_arr:
        if numb == 1:
            one_s += 1
        elif numb == 0:
            zero_s += 1
        else:
            print(numb)
    print(one_s, " one\'s and ", zero_s, " zero\'s with a length of ", len(fp_arr))


def create_pfp(end_units: dict, repeating_units: dict, mol_weight: int, fp_size: int, fp_type="pfp"):
    """this function exactly takes in one dictionary with the start and end keys for the end-groups smiles code,
    a second dictionary containing smiles repetition units as values and respective key of their mole fraction,
    the number average molar mass and optionally the desired fingerprint bit size.

    A concatenated version of a representative and fingerprint containing all possible substructures and a weighting
    of the occurrence of the different components is returned as one array with five times the input fp_size."""
    fp_funcs = [create_subs_fp, create_hashed_ap_fp]
    if fp_type.startswith("morg"):  # using the morgan/ECFP-ish FP with radius of 4 instead of subs+AP
        fp_funcs = [create_morgan_bit_fp]
    end_share = calc_ends_share(end_units, repeating_units, mol_weight)
    intersection_fp = merge_subs_fp(*create_subs_fp(*(combinations(*(repeating_units.values()), **end_units))))
    enhanced_sum_fp = concatenator(
        [concatenator(  # both types of fp with significance_sigma for both start/end and rep units
            significance_sigma(fp_rep_arrays=fp_func(*list(end_units.values()), fp_size=fp_size),
                               incidence_s=2 * [end_share])
            +
            significance_sigma(fp_rep_arrays=fp_func(*repeating_units.values(), fp_size=fp_size),
                               incidence_s=list(repeating_units.keys()))
        ) for fp_func in fp_funcs])
    return concatenator([intersection_fp] + [enhanced_sum_fp])


def reduce_fp_set(*fingerprints):
    """takes in a number of arrays and gives back a list of arrays without the positions which never change
    and the two masks needed to add another fingerprint to that list (see reduce_fp)."""
    all_fps = np.concatenate(fingerprints)
    all_fps = all_fps.reshape(len(fingerprints), -1)
    first = all_fps[0]
    mask = np.all(all_fps == first, axis=0)
    all_fps = all_fps.transpose()
    reduced_fp = list(all_fps[~mask].transpose())
    print("reduced size by {0:.0f}%".format((1 - (len(reduced_fp[0]) / len(first))) * 100))
    return reduced_fp, mask, first


def reduce_fp(*fingerprint, mask, first):
    """takes in a number of arrays and two masks and returns the masked-out version and an idea of information lost."""
    reduced_fp = [new_fp[~mask] for new_fp in fingerprint]
    bit_loss = len([new_fp2[~(mask & (new_fp2 == first))] for new_fp2 in fingerprint][0])
    # only the info loss for the first fingerprint is given
    loss = 1 - (len(reduced_fp[0]) / bit_loss)
    return reduced_fp, loss
