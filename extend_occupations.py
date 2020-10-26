import respy as rp

#
from auxiliary import _add_occupations

models, num_periods = ["kw_97_basic", "kw_94_one", "kw_97_extended"], 3
add_occ, add_types = range(1, 3), range(1, 3)

for model, add_occ, add_types in list(product(models, add_occ, add_types)):
    args = (model, num_periods, add_occ, add_types)
    params, options = scaling_model_specification(*args)
    rp.get_simulate_func(params, options)(params)
