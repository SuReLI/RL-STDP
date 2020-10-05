# %% imports

using JLD
using EvolutionaryStrategies


include("cartpole_fitness.jl")
#include("swingup_fitness.jl")
#include("mountcar_fitness.jl")

# %% sNES optimisation


cfg = get_config("Julia/Actors/cartpole_esconfig.yml")
es = sNES(cfg,fitness)
run!(es)


# %%


save("Julia/Actors/sNESelites_cartpole.jld", "elites", es.elites)
# %% save the results

elites = load("Julia/Actors/sNESelites_cartpole.jld", "elites")
