import pandas as pd
import numpy as np

import string


def update_model_specification(params, options, num_occupations):
    params_update, options_update = params.copy(), options.copy()
    params_update = update_params(params_update, num_occupations)
    options_update = update_options(options_update, params_update)
    return params_update, options_update


def update_params(params_update, num_occupations):
    params_update = add_generic_occupations(params_update, num_occupations)
    params_update = construct_shocks_sdcorr(params_update)
    params_update = construct_meas_error(params_update)
    return params_update


def update_options(options_update, params_update):
    _, occupations = _get_choices_occupations(params_update)
    options_update = update_core_state_space_filters(options_update, occupations)
    options_update = update_covariates(options_update, occupations)
    return options_update


def construct_meas_error(params_occ):

    if "meas_error" not in params_occ.index.get_level_values("category"):
        return params_occ

    occupations = _get_choices_occupations(params_occ)[1]

    default_value = params_occ.loc["meas_error"]["value"].iloc[0]
    meas_error_base = params_occ.loc["meas_error"].copy()

    params_occ.drop("meas_error", level="category", inplace=True)

    info_base = {"category": ["meas_error"], "comment": "..."}
    for occupation in occupations:
        name = f"sd_{occupation}"

        if name in meas_error_base.index:
            value = meas_error_base.loc[name, "value"]
        else:
            value = default_value
        info_base["value"] = value

        info_base["name"] = name

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


def update_core_state_space_filters(options, occupations):
    substring = ""
    for occupation in occupations:
        substring += f"exp_{occupation} + "

    # We need to update the state space filters.
    substring = substring[:substring.rfind("+") - 1]
    if check_is_kw_97(occupations=occupations):
        substring = f"period > 0 and {substring} + exp_school == period "
    else:
        substring = f"period > 0 and {substring} + exp_edu == period "
    substring += "and lagged_choice_1 == '{choices_wo_exp}'"

    try:
        options["core_state_space_filters"][1] = substring
    except KeyError:
        pass

    return options


def update_covariates(options, occupations):
    # TODO: Updating covariates ...
    entries_to_remove = list()
    for key_ in options["covariates"].keys():
        if key_.startswith("exp_"):
            entries_to_remove.append(key_)

    for key_ in entries_to_remove:
        options["covariates"].pop(key_, None)

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


def check_is_kw_97(**kwargs):
    if "params" in kwargs:
        _, occupations = _get_choices_occupations(kwargs["params"])
        if "military" in occupations:
            return True
        else:
            return False
    elif "occupations" in kwargs:
        if "military" in kwargs["occupations"]:
            return True
        else:
            return False


def _construct_sdcorr_indices(occupations):

    # We need to ensure a particular order in the entries to shock matrix.
    choices_order = occupations
    if check_is_kw_97(occupations=occupations):
        choices_order += ["school", "home"]
    else:
        choices_order += ["edu", "home"]

    # for choice in choices:
    num_choices = len(choices_order)
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
    return index


def construct_shocks_sdcorr(params_occ):

    sd_corr_base = params_occ.loc[("shocks_sdcorr", slice(None)), :].copy()
    default_value = sd_corr_base.loc["shocks_sdcorr"]["value"].iloc[0]

    params_occ.drop("shocks_sdcorr", level="category", inplace=True)

    choices, occupations = _get_choices_occupations(params_occ)

    indices = _construct_sdcorr_indices(occupations)
    shocks_sdcorr = pd.DataFrame(index=indices, columns=["value", "comment"])

    for category, name in shocks_sdcorr.index:
        index = (category, name)

        # If the required information is available, we just use that one.
        if index in sd_corr_base.index:
            access = index, "value"
            shocks_sdcorr.loc[access] = sd_corr_base.loc[access]
            continue

        # We set any remaining correlations to zero.
        if "corr_" in name:
            shocks_sdcorr.loc[index, "value"] = 0.0
            continue

        # We set any remaining standard deviations to the default value.
        if "sd_" in name:
            shocks_sdcorr.loc[index, "value"] = default_value
            continue

    params_occ = params_occ.append(shocks_sdcorr)
    params_occ["value"] = params_occ["value"].astype("float")

    return params_occ

