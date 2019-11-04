import os
# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

import respy as rp
import numpy as np

params, options, df = rp.get_example_model("robinson", with_data=True)

crit_func = rp.get_crit_func(params, options, df)

for delta in [0.001, 0.5]:
    print("\n\n")
    params.loc[("shocks_sdcorr", "sd_hammock"), :] = delta
    fval = crit_func(params)
    print(delta, fval)