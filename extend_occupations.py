import respy as rp
from test_extend_occupations import get_random_request
from auxiliary import scaling_model_specification

while True:
    params, options = scaling_model_specification(*get_random_request())
    rp.get_simulate_func(params, options)(params)
