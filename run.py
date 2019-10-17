from pathlib import Path
import yaml

import pandas as pd
import respy as rp
import numpy as np
#params = pd.read_csv("robinson-crusoe.csv")
#options = yaml.safe_load(Path("robinson-crusoe.yaml").read_text())
#simulate = rp.get_simulate_func(params, options)
#df = simulate(params)
np.random.seed(123)
from respy.tests.random_model import generate_random_model
params, options = generate_random_model()

options["n_periods"] = 1
options["simulation_agents"] = 1000

params.loc["meas_error", :] = 0
params.loc["shocks_sdcorr", "sd_b"] = 1


params.to_pickle("params.respy.pkl")
with open('options.respy.yml', 'w') as outfile:
    yaml.dump(options, outfile, default_flow_style=False)

params = pd.read_pickle("params.respy.pkl")
options = yaml.load(open('options.respy.yml', 'r'))

simulate = rp.get_simulate_func(params, options)
df = simulate(params)


crit_func = rp.get_crit_func(params, options, df)
crit_func(params)

df.to_pickle("simulated.respy.pkl")