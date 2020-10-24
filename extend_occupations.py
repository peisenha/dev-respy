import numpy as np
import respy as rp

# TODO: nonpecs need to be dealt with ..
from auxiliary import add_generic_occupations
from auxiliary import construct_shocks_sdcorr
from auxiliary import _get_choices_occupations
params, options, data = rp.get_example_model("kw_94_two")


NUM_OCCUPATIONS = 3

# TODO: Needs to be larger than previous ones?
assert NUM_OCCUPATIONS > 0

params_occ = add_generic_occupations(params, NUM_OCCUPATIONS)
params_occ = construct_shocks_sdcorr(params_occ)

# TODO: We are now working on options part based on the revised params.
options["n_periods"] = 10 # debugging only.

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

_, occupations = _get_choices_occupations(params_occ)

for occupation in occupations:
    options["covariates"][f"exp_{occupation}_square"] = f"exp_{occupation} ** 2"

simulate = rp.get_simulate_func(params_occ, options)
df = simulate(params_occ)

np.testing.assert_equal(df["Choice"].nunique(), 5)
np.testing.assert_almost_equal(df.sum().sum(), 21461750223.46615)
