from itertools import product

import pandas as pd
import numpy as np
import respy as rp

import string

# TODO: nonpecs need to be dealt with ..


def construct_shocks_sdcorr():
    # non_occupation = ["edu", "home"]
    # occupation = list(letters)
    #
    # names = list()
    # labels = non_occupation + occupation
    #
    # for labels in product(labels, labels):
    #     left, right = labels
    #
    #     # We do not need correlation
    #     if left == right:
    #         names += [f"sd_{left}"]
    #         continue
    #
    #     # We check if
    #     if f"corr_{right}_{left}" in names:
    #         continue
    #
    #     names += [f"corr_{left}_{right}"]
    #

    index = [
        "sd_a",
        "sd_b",
        "sd_c",
        "sd_edu",
        "sd_home",
        "corr_b_a",
        "corr_c_a",
        "corr_c_b",
        "corr_edu_a",
        "corr_edu_b",
        "corr_edu_c",
        "corr_home_a",
        "corr_home_b",
        "corr_home_c",
        "corr_home_edu",
    ]

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


def construct_generic_occupation():
    """This funcions creates generic occupation by just taking one of te already specified ones
    and removing all exerience related terms, these then are added again later."""

    d


params, options, data = rp.get_example_model("kw_94_two")


NUM_OCCUPATIONS = 3

assert NUM_OCCUPATIONS > 0
letters = string.ascii_lowercase[:NUM_OCCUPATIONS]

# We first delete all other occupations.
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

print(occ_base)

    # We now add it to the params dataframe
for letter in letters:
    occ_params = occ_base.loc["wage_a", :].copy()
    occ_params = pd.concat([occ_params], keys=[f"wage_{letter}"], names=["category"])

    params_debug = params_debug.append(occ_params)

params_occ = params_debug.copy()

# # Add generic occupations.
# def add_generic_occupations(params, num_occupations):
#     params_occ = params.copy()
#     choices = params.index.unique("category")
#
#     occupations = list()
#     for choice in choices:
#         if "wage_" not in choice:
#             continue
#         occupations.append(choice.split("_")[1])
#
#     # Now we just pick the first one.
#     occ_generic = params.loc[f"wage_{occupations[0]}"]
#     occ_generic = pd.concat([occ_generic], keys=["wage_a"], names=["category"])
#
#     # Now we remove all experience terms.
#     for index in occ_generic.index.get_level_values(0):
#
#         if "exp_" not in index:
#             continue
#
#         if "edu" in index:
#             continue
#
#         occ_generic.drop(index, level="name", inplace=True)
#
#         # Now we create a new experience variables
#     for letter in letters:
#         for ext_ in ["", "_square"]:
#             name = f"exp_{letter}" + ext_
#             info = {
#                 "category": ["wage_a"],
#                 "name": [name],
#                 "value": [0.0],
#                 "comment": ["comment"],
#             }
#
#             info = pd.DataFrame.from_dict(info).set_index(["category", "name"])
#             occ_generic = occ_generic.append(info)
#
#     # Now we just put them all together.
#     for letter in letters:
#         occ_letter = occ_generic.loc["wage_a", :].copy()
#         occ_letter = pd.concat([occ_letter], keys=[f"wage_{letter}"], names=["category"])  ##
#         params_occ = params_occ.append(occ_letter)
#
#     return params_occ
#
#
# params_occ = add_generic_occupations(params_debug, NUM_OCCUPATIONS)

# Construct shock matrix under new conditions.
shocks_sdcorr = construct_shocks_sdcorr()
params_occ = params_occ.append(shocks_sdcorr)


options["core_state_space_filters"] = [
    "period > 0 and exp_{choices_w_exp} == period and lagged_choice_1 != '{choices_w_exp}'",
    "period > 0 and exp_a + exp_b + exp_c + exp_edu == period and lagged_choice_1 == '{choices_wo_exp}'",
    "period > 0 and lagged_choice_1 == 'edu' and exp_edu == 0",
    "lagged_choice_1 == '{choices_w_wage}' and exp_{choices_w_wage} == 0",
    "period == 0 and lagged_choice_1 == '{choices_w_wage}'",
]

options["n_periods"] = 10
options["covariates"]["exp_c_square"] = "exp_c ** 2"

params_occ["value"] = params_occ["value"].astype("float")


simulate = rp.get_simulate_func(params_occ, options)
df = simulate(params_occ)

np.testing.assert_equal(df["Choice"].nunique(), 5)
np.testing.assert_almost_equal(df.sum().sum(), 21461750223.46615)
