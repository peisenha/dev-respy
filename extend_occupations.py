import numpy as np
import respy as rp

# TODO: nonpecs need to be dealt with ..
from auxiliary import add_generic_occupations
from auxiliary import construct_shocks_sdcorr

params, options, data = rp.get_example_model("kw_94_two")


NUM_OCCUPATIONS = 3

# TODO: Needs to be larger than previous ones.
assert NUM_OCCUPATIONS > 0

params_occ = add_generic_occupations(params, NUM_OCCUPATIONS)

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
params_occ.to_pickle("params_occ.pkl")

simulate = rp.get_simulate_func(params_occ, options)
df = simulate(params_occ)

np.testing.assert_equal(df["Choice"].nunique(), 5)
np.testing.assert_almost_equal(df.sum().sum(), 21461750223.46615)
