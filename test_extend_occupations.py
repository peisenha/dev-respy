from itertools import product

import respy as rp
import numpy as np

from auxiliary import scaling_model_specification


def test_1():
    params, options = scaling_model_specification("kw_94_two", 10, 3)
    df = rp.get_simulate_func(params, options)(params)
    np.testing.assert_almost_equal(df.sum().sum(), 14022027476.102118)


def test_2():
    combinations = product(["kw_97_basic", "kw_94_one", "kw_97_extended"], range(1, 3))
    for model, n_occupations in combinations:
        params, options = scaling_model_specification(model, 3, n_occupations)
        rp.get_simulate_func(params, options)(params)
