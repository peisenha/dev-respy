import respy as rp
import numpy as np

from auxiliary import update_model_specification
from auxiliary import _get_choices_occupations

def test_1():

    params, options, data = rp.get_example_model("kw_94_two")

    num_occupations, n_periods = 3, 10
    params_occ, options_occ = update_model_specification(params, options, num_occupations)

    options["n_periods"] = n_periods

    simulate = rp.get_simulate_func(params_occ, options)
    df = simulate(params_occ)

    np.testing.assert_equal(df["Choice"].nunique(), 5)
    np.testing.assert_almost_equal(df.sum().sum(), 21461750223.46615)


def test_2():

    params, options, data = rp.get_example_model("kw_94_two")

    for num_occupations in range(1, 5):
        n_periods = 3
        params_occ, options_occ = update_model_specification(params, options, num_occupations)

        choices, _ = _get_choices_occupations(params_occ)

        options["n_periods"] = n_periods

        simulate = rp.get_simulate_func(params_occ, options)
        simulate(params_occ)

        np.testing.assert_equal(num_occupations + 2, len(choices))
