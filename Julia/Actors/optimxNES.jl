# %% imports

using JLD
using EvolutionaryStrategies


include("cartpole_fitness.jl")
#include("swingup_fitness.jl")
#include("mountcar_fitness.jl")
include("cambrianrun.jl")

# %% create snestopgen
init = 0
save("Julia/Actors/topgenxNES.jld","init", init)
# %% xNES optimisation

for opti in 1:50
    cfg = get_config("Julia/Actors/cartpole_esconfig.yml")
    es = xNES(cfg,fitness)
    a = run!(es, opti)
    println("optim $(opti)")
end


# %% plot the distribution of gen convergence

f = load("Julia/Actors/topgenxNES.jld")
gens = Int64[]
for i in 1:length(f)-1
    push!(gens, f["$(i)_run"])
end

histogram(f, bins = 0:10:300,
            label = "First R>195 individual (mean = 82.68)",
            color = :green,
            xlabel = "Generation",
            ylabel = "Number of optimisations")
