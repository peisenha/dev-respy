import numpy as np
import respy as rp

# TODO: nonpecs need to be dealt with ..
# TODO: Types as hard-coded number
# TODO: Need to add mode complex grid and nonceps.

from auxiliary import _get_choices_occupations, update_model_specification

n_periods = 3
params, options, data = rp.get_example_model("kw_97_basic")






for num_occupations in range(1, 5):
    params_occ, options_occ = update_model_specification(params, options, num_occupations)

    choices = _get_choices_occupations(params_occ)[0]

    options["n_periods"] = n_periods

    simulate = rp.get_simulate_func(params_occ, options)
