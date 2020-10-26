from itertools import product

import respy as rp
import numpy as np

from auxiliary import scaling_model_specification


def get_random_request():
    add_sample = [None, 1, 2, 3, 4]
    add_occ, add_types = np.random.choice(add_sample, size=2)
    num_periods = np.random.choice(range(1, 10))
    model = np.random.choice(["kw_97_basic", "kw_94_one", "kw_97_extended"])
    return model, num_periods, add_occ, add_types


def test_1():
    params, options = scaling_model_specification("kw_94_two", 10, 3)
    df = rp.get_simulate_func(params, options)(params)
    np.testing.assert_almost_equal(df.sum().sum(), 14022027476.102118)


def test_2():
    models, num_periods = ["kw_97_basic", "kw_94_one", "kw_97_extended"], 3
    add_occ, add_types = range(1, 3), range(1, 3)

    for model, add_occ, add_types in list(product(models, add_occ, add_types)):
        args = (model, num_periods, add_occ, add_types)
        params, options = scaling_model_specification(*args)
        rp.get_simulate_func(params, options)(params)


def test_3():
    params, options = scaling_model_specification(*get_random_request())
    rp.get_simulate_func(params, options)(params)
