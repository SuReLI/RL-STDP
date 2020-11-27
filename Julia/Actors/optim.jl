# %% imports

using JLD
using EvolutionaryStrategies


include("cartpole_fitness.jl")
#include("swingup_fitness.jl")
#include("mountcar_fitness.jl")
include("cambrianrun.jl")

# %% create snestopgen
init = 0
save("Julia/Actors/topgensNES.jld","init", init)
# %% sNES optimisation

for opti in 11:100
    cfg = get_config("Julia/Actors/cartpole_esconfig.yml")
    es = sNES(cfg,fitness)
    a = run!(es, opti)
    println("optim $(opti)")
end


# %% plot the distribution of gen convergence

f = load("Julia/Actors/topgensNES.jld")
gens = Int64[]
for i in 1:length(f)-1
    push!(gens, f["$(i)_run"])
end

histogram(f, bins = 0:10:300,
            label = "First R>195 individual (mean = 87.58)",
            color = :blue,
            xlabel = "Generation",
            ylabel = "Number of optimisations")
