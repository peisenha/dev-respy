import respy as rp

params, options = rp.get_example_model("kw_94_one", with_data=False)


options["simulation_agents"] = 1

simulate = rp.get_simulate_func(params, options)
df = simulate(params)

