import pandas as pd
import numpy as np
import respy as rp

import string


def scaling_model_specification(base_model, num_periods=None, add_occ=None, add_types=None):
    params, options = rp.get_example_model(base_model, with_data=False)

    if num_periods is not None:
        options = _modify_periods(options, num_periods)

    if add_occ is not None:
        params, options = _add_occupations(params, options, add_occ)

    if add_types is not None:
        params = _add_types(params, add_types)

    return params, options


def _add_types(params_update, add_types):

    choices, occupations = _get_choices_occupations(params_update)
    non_occupations = list(set(choices).difference(occupations))

    for iter_ in range(add_types):
        for choice in choices:

            if choice in non_occupations:
                category = f"nonpec_{choice}"
            else:
                category = f"wage_{choice}"

            if _check_is_kw_97(occupations=occupations):
                if "military" in choice:
                    continue
                value = params_update.loc[(category, "type_1"), "value"]
                name = f"type_{4 + iter_}"
            else:
                name, value = f"type_{1 + iter_}", 0.0

            params_update.loc[(category, name), "value"] = value

    for iter_ in range(add_types):

        if _check_is_kw_97(occupations=occupations):
            if "military" in choice:
                continue
            type_add = params_update.loc[("type_1", slice(None)), :].copy()
            name = f"type_{4 + iter_}"
        else:
            base = {"category": ["type_1"], "name": "constant", "value": 1.0}
            type_add = pd.DataFrame(base).set_index(["category", "name"])
            name = f"type_{1 + iter_}"

        type_add.reset_index(inplace=True)
        type_add.loc[:, "category"] = name
        type_add.set_index(["category", "name"], inplace=True)

        params_update = params_update.append(type_add)

    return params_update


def _modify_periods(options, num_periods):
    options["n_periods"] = num_periods
    return options


def _add_occupations(params, options, add_occupations):
    params_update, options_update = params.copy(), options.copy()
    params_update = _update_params(params_update, add_occupations)
    options_update = _update_options(params_update, options_update)
    return params_update, options_update


def _update_params(params_update, add_occupations):
    params_update = _add_generic_occupations(params_update, add_occupations)
    params_update = _construct_shocks_sdcorr(params_update)
    params_update = _construct_meas_error(params_update)
    return params_update


def _update_options(params_update, options_update):
    _, occupations = _get_choices_occupations(params_update)
    options_update = _update_core_state_space_filters(options_update, occupations)
    options_update = _update_covariates(options_update, occupations)
    return options_update


def _construct_meas_error(params_update):

    if "meas_error" not in params_update.index.get_level_values("category"):
        return params_update

    occupations = _get_choices_occupations(params_update)[1]

    default_value = params_update.loc["meas_error"]["value"].iloc[0]
    meas_error_base = params_update.loc["meas_error"].copy()

    params_update.drop("meas_error", level="category", inplace=True)

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
        params_update = params_update.append(meas_error)

    return params_update


def _get_choices_occupations(params_update):
    choices, occupations = list(), list()
    for candidate in params_update.index.get_level_values("category"):
        if "wage" not in candidate and "nonpec" not in candidate:
            continue
        choice = candidate.split('_', 1)[1]
        if "wage" in candidate and choice not in occupations:
            occupations.append(choice)

        if choice in choices:
            continue

        choices.append(choice)

    return choices, occupations


def _update_core_state_space_filters(options_update, occupations):
    substring = ''.join([f"exp_{occupation} + " for occupation in occupations])
    substring = substring[:substring.rfind("+") - 1]
    substring = f"period > 0 and {substring}"
    if _check_is_kw_97(occupations=occupations):
        substring += " + exp_school == period "
    else:
        substring += f" + exp_edu == period "
    substring += "and lagged_choice_1 == '{choices_wo_exp}'"

    # TODO: For some reason there are no state space filters defined in the basic model.
    try:
        options_update["core_state_space_filters"][1] = substring
    except KeyError:
        pass

    return options_update


def _update_covariates(options_update, occupations):
    for occupation in occupations:
        key_ = f"exp_{occupation}_square"
        if key_ in options_update["covariates"].keys():
            continue
        options_update["covariates"][key_] = f"exp_{occupation} ** 2"

    return options_update


def _add_generic_occupations(params_update, add_occupations):
    letters = string.ascii_lowercase[:add_occupations]

    _, occupations = _get_choices_occupations(params_update)

    model = "kw_94"
    if "military" in occupations:
        model = "kw_97"

    occ_grid = pd.read_pickle(f"occ_grid_{model}.pkl")

    for letter in letters:
        occ_add = occ_grid.copy()
        occ_add.reset_index(inplace=True)
        occ_add.loc[:, "category"] = f"wage_{letter}{letter}"
        occ_add.set_index(["category", "name"], inplace=True)
        params_update = params_update.append(occ_add)

    _, occupations_update = _get_choices_occupations(params_update)

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
                params_update = params_update.append(info)

    params_update = params_update.sort_index()

    return params_update


def _check_is_kw_97(**kwargs):
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
    if _check_is_kw_97(occupations=occupations):
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


def _construct_shocks_sdcorr(params_update):

    sd_corr_base = params_update.loc[("shocks_sdcorr", slice(None)), :].copy()
    default_value = sd_corr_base.loc["shocks_sdcorr"]["value"].iloc[0]

    params_update.drop("shocks_sdcorr", level="category", inplace=True)

    choices, occupations = _get_choices_occupations(params_update)

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

    params_update = params_update.append(shocks_sdcorr)
    params_update["value"] = params_update["value"].astype("float")

    return params_update

