import pandas as pd
import numpy as np

import string


# TODO: Types as hard-coded number


def update_model_specification(params, options, num_occupations):
    params_occ = update_params(params, num_occupations)
    options_occ = update_options(options, params_occ)
    return params_occ, options_occ


def update_params(params, num_occupations):
    params_occ = add_generic_occupations(params, num_occupations)
    params_occ = construct_shocks_sdcorr(params_occ)
    params_occ = construct_meas_error(params_occ)
    return params_occ


def construct_meas_error(params_occ):
    _, occupations = _get_choices_occupations(params_occ)

    if "meas_error" in params_occ.index.get_level_values("category"):
        occupations = _get_choices_occupations(params_occ)[1]

        value = params_occ.loc["meas_error"]["value"].iloc[0]
        params_occ.drop("meas_error", level="category", inplace=True)

        info_base = {"category": ["meas_error"], "comment": ["..."], "value": value}
        for occupation in occupations:
            info_base["name"] = [f"sd_{occupation}"]
            meas_error = pd.DataFrame.from_dict(info_base).set_index(["category", "name"])
            params_occ = params_occ.append(meas_error)

    return params_occ


def _get_choices_occupations(params):
    choices, occupations = list(), list()
    for candidate in params.index.get_level_values("category"):
        if "wage" not in candidate and "nonpec" not in candidate:
            continue

        choice = candidate.split('_', 1)[1]
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

    # There is a different naming between the models.

    substring = substring[:substring.rfind("+") - 1]
    if "exp_edu" in params_occ.index.get_level_values("name"):
        substring = f"period > 0 and {substring} + exp_edu == period "
    else:
        substring = f"period > 0 and {substring} + exp_school == period "

    substring += "and lagged_choice_1 == '{choices_wo_exp}'"

    try:
        options["core_state_space_filters"][1] = substring
    except KeyError:
        pass

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


def add_generic_occupations(params, num_occupations):
    letters = string.ascii_lowercase[:num_occupations]
    params_debug = params.copy()

    _, occupations = _get_choices_occupations(params)

    model = "kw_94"
    if "military" in occupations:
        model = "kw_97"

    occ_grid = pd.read_pickle(f"occ_grid_{model}.pkl")

    # TODO: Need to add mode complex grid and nonceps.
    for letter in letters:
        occ_add = occ_grid.copy()
        occ_add.reset_index(inplace=True)
        occ_add.loc[:, "category"] = f"wage_{letter}{letter}"
        occ_add.set_index(["category", "name"], inplace=True)
        params_debug = params_debug.append(occ_add)

    _, occupations_update = _get_choices_occupations(params_debug)

    for letter in letters:
        for occupation in occupations_update:
            for ext_ in ["", "_square"]:

                name = f"exp_{letter}{letter}" + ext_
                info = {
                     "category": [f"wage_{occupation}"],
                     "name": [name],
                     "value": [0.0],
                     "comment": ["comment"],
                }
                info = pd.DataFrame.from_dict(info).set_index(["category", "name"])
                params_debug = params_debug.append(info)

    params_debug = params_debug.sort_index()

    return params_debug


def construct_shocks_sdcorr(params_occ):

    params_occ.drop("shocks_sdcorr", level="category", inplace=True)

    choices, occupations = _get_choices_occupations(params_occ)

    choices_order = occupations
    for choice in choices:
        if choice in choices_order:
            continue
        if choice in ["school", "home", "edu"]:
            continue

        choices_order.append(choice)

    if "military" in occupations:
        choices_order.append("school")
    else:
        choices_order.append("edu")

    choices_order.append("home")

    print(choices_order)
    num_choices = len(choices)
    cov = np.tile("a" * 30, (num_choices, num_choices))

    for i, row in enumerate(choices_order):
        for j, column in enumerate(choices_order):
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

