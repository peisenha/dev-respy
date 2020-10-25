import respy as rp

# We create a grid for the KW94 additions.
params, options, data = rp.get_example_model("kw_94_one")
occ_grid_kw94 = params.loc[(f"wage_a", slice(None)), :].copy()

occ_grid_kw94.reset_index(inplace=True)
occ_grid_kw94["category"] = "wage_aa"
occ_grid_kw94.set_index(["category", "name"], inplace=True)
occ_grid_kw94.to_pickle("occ_grid_kw_94.pkl")

# We create a grid for the KW97 additions.
params, options, data = rp.get_example_model("kw_97_extended")
labels = list()
labels += ["constant", "exp_school", "exp_white_collar", "exp_white_collar_square"]
labels += ["exp_blue_collar", "exp_blue_collar_squared", "exp_military"]
occ_grid_kw97 = params.loc[(f"wage_white_collar", labels), :].copy()

occ_grid_kw97.reset_index(inplace=True)
occ_grid_kw97.loc[:, "category"] = "wage_aa"
occ_grid_kw97.set_index(["category", "name"], inplace=True)
occ_grid_kw97.to_pickle("occ_grid_kw_97.pkl")
