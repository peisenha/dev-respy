import pandas as pd
import numpy as np

import string


def add_generic_occupations(params, NUM_OCCUPATIONS):
    letters = string.ascii_lowercase[:NUM_OCCUPATIONS]
    params_debug = params.copy()

    occ_base = params_debug.loc["wage_a", :].copy()

    params_debug.drop("shocks_sdcorr", level="category", inplace=True)
    params_debug.drop("wage_b", level="category", inplace=True)
    params_debug.drop("wage_a", level="category", inplace=True)

    # We now create a generic new occupation
    occ_base = pd.concat([occ_base], keys=["wage_a"], names=["category"])

    # We need to drop all experience terms that refer to other occupations
    for label in occ_base.index.get_level_values(level="name"):
        if "exp_" in label and "edu" not in label:
            occ_base.drop(label, level="name", inplace=True)

    # We now create new experience variables.
    for letter in letters:
        for ext_ in ["", "_square"]:
            name = f"exp_{letter}" + ext_
            info = {
                "category": ["wage_a"],
                "name": [name],
                "value": [0.0],
                "comment": ["comment"],
            }

            info = pd.DataFrame.from_dict(info).set_index(["category", "name"])
            occ_base = occ_base.append(info)

    # We now add it to the params dataframe
    for letter in letters:
        occ_params = occ_base.loc["wage_a", :].copy()
        occ_params = pd.concat([occ_params], keys=[f"wage_{letter}"], names=["category"])

        params_debug = params_debug.append(occ_params)

    params_occ = params_debug.copy()

    return params_occ


def construct_shocks_sdcorr():

    choices = ["a", "b", "c", "edu", "home"]
    num_choices = len(choices)
    cov = np.tile("a" * 20, (num_choices, num_choices))

    for i, row in enumerate(choices):
        for j, column in enumerate(choices):
            if row == column:
                cov[i, j] = f"sd_{row}"
            else:
                str_ = f"corr_{row}_{column}"

                cov[i, j] = str_

    index = list(np.diag(cov)) + list(cov[np.tril_indices(num_choices, -1)].flatten())

    indices = list()
    for name in index:
        indices += [("shocks_sdcorr", name)]

    index = pd.MultiIndex.from_tuples(indices, names=["category", "name"])
    shocks_sdcorr = pd.DataFrame(index=index, columns=["value", "comment"])

    for category, name in shocks_sdcorr.index:
        if "sd_" in name:
            shocks_sdcorr.loc[(category, name), "value"] = 1.0
        elif "corr_" in name:
            shocks_sdcorr.loc[(category, name), "value"] = 0.0
        else:
            raise AssertionError

    return shocks_sdcorr

