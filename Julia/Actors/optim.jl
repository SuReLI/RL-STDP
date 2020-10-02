# %% imports

using JLD
using EvolutionaryStrategies

include("gensaver.jl")

include("cartpole_fitness.jl")
#include("swingup_fitness.jl")
#include("mountcar_fitness.jl")

# %% sNES optimisation

results = Results(20,["fit","genes"])
@show results.values["genes"]
cfg = get_config("Julia/Actors/cartpole_esconfig.yml")
es = sNES(cfg,fitness)
run!(es)

# %% save the results

resul = load("Julia/Actors/results_sNES_cartpole.jld", "results")
@show resul.values["genes"]
