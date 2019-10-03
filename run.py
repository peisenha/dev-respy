import respy as rp
import yaml
import pandas as pd


# Import start values and model spec
options = yaml.safe_load(open('obs.options.yaml', 'r'))
params = pd.read_pickle('obs.params.pkl')

simulate = rp.get_simulate_func(params, options)


df = simulate(params)
