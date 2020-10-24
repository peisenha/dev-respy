import pandas as pd
import numpy as np

import string


def update_model_specification(params, options, num_occupations):
    params_occ = update_params(params, num_occupations)
    options_occ = update_options(options, params_occ)

    return params_occ, options_occ

def update_params(params, num_occupations):
    params_occ = add_generic_occupations(params, num_occupations)
    params_occ = construct_shocks_sdcorr(params_occ)

    return params_occ


def _get_choices_occupations(params):
    choices, occupations = list(), list()
    for candidate in params.index.get_level_values("category"):
        if "wage" not in candidate and "nonpec" not in candidate:
            continue

        choice = candidate.split("_")[1:][0]
        if "wage" in candidate and choice not in occupations:
            occupations.append(choice)

        if choice in choices:
            continue

        choices.append(choice)

    return choices, occupations


def update_options(options, params_occ):
    # TODO: Updating filters ...
    _, occupations = _get_choices_occupations(params_occ)
    substring = ""
    for occupation in occupations:
        substring += f"exp_{occupation} + "

    substring = substring[:substring.rfind("+") - 1]
    substring = f"period > 0 and {substring} + exp_edu == period "
    substring += "and lagged_choice_1 == '{choices_wo_exp}'"

    options["core_state_space_filters"][1] = substring

    # TODO: Updating covariates ...
    entries_to_remove = list()
    for key_ in options["covariates"].keys():
        if key_.startswith("exp_"):
            entries_to_remove.append(key_)

    for key_ in entries_to_remove:
        options["covariates"].pop(key_, None)

    choices, occupations = _get_choices_occupations(params_occ)

    for occupation in occupations:
        options["covariates"][f"exp_{occupation}_square"] = f"exp_{occupation} ** 2"

    return options


def add_generic_occupations(params, NUM_OCCUPATIONS):
    letters = string.ascii_lowercase[:NUM_OCCUPATIONS]
    params_debug = params.copy()

    occ_base = params_debug.loc["wage_a", :].copy()

    # We remove all existing occupations.
    _, occupations = _get_choices_occupations(params)
    for occupation in occupations:
        params_debug.drop(f"wage_{occupation}", level="category", inplace=True)

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


def construct_shocks_sdcorr(params_occ):

    params_occ.drop("shocks_sdcorr", level="category", inplace=True)


    choices, _ = _get_choices_occupations(params_occ)
    # TODO: I do not know if this is flexible enough. This ensures ordering in CORR matrix.
    choices.sort()

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

    params_occ = params_occ.append(shocks_sdcorr)
    params_occ["value"] = params_occ["value"].astype("float")

    return params_occ

