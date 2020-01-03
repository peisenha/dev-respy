import os

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

import respy as rp
import numpy as np

from respy.config import EXAMPLE_MODELS

for model in ["kw_97_basic"]:
    print(model)
    params, options, df = rp.get_example_model(model, with_data=True)

    simulate = rp.get_simulate_func(params, options)
    simulate(params)

    crit_func = rp.get_crit_func(params, options, df)
    crit_func(params)