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

    np.testing.assert_almost_equal(df.sum().sum(), 14022027476.102118)


def test_2():

    n_periods = 3

    for model in ["kw_97_basic", "kw_94_one", "kw_97_extended"]:

        params, options, data = rp.get_example_model(model)

        for num_occupations in range(1, 3):
            params_occ, options_occ = update_model_specification(params, options, num_occupations)

            choices = _get_choices_occupations(params_occ)[0]

            options["n_periods"] = n_periods

            simulate = rp.get_simulate_func(params_occ, options)
            simulate(params_occ)
            if "kw_94" in model:
                np.testing.assert_equal(num_occupations + 4, len(choices))
            else:
                np.testing.assert_equal(num_occupations + 5, len(choices))


def test_3():
    """Test that renaming does not affect rsult ,so if the num_occupation is same as in original
    then state space stuff should be very similar."""
    pass