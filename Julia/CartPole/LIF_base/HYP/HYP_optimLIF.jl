# %% imports

include("../../../Modules/CMAES/cmaes.jl")
include("HYP_sim.jl")
include("../../../Modules/save_param.jl")


# %%


function optim()
    cfg = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes.yml"))
    tunekeys = cfg["tunekeys"]
    tune = String[]
    for key in keys(tunekeys)
        push!(tune, key)
    end
    arch = cfg["arch"]
    N_hyparams = length(cfg["tunekeys"])
    N_alpha = arch[1]*4
    N_w = sum([arch[i]*(arch[i-1]-(div(arch[i-1],4))) for i in 2:length(arch)]) +1
    N = N_alpha + N_w + N_hyparams

    best = Array{Float64}[]
    for i in 1:20
        push!(best,[])
    end
    best_fit = [-1.0 for i in 1:20]

    # create starting point
    # results = load_param("Julia/CartPole/LIF_base/results_HYP888bound_error.jld")
    # genes_init = results[]




    lambda = 4+floor(Int64,3*log(N))
    mu = floor(Int64,lambda/2)
    c = CMAES(N=N, μ=mu, λ=lambda , τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N))
    for i in 1:1000
        @time step!(c)
        bestind = argmin(c.F_λ)
        maxfit = -c.F_λ[bestind]
        println(i," ", maxfit, " (current best = $(maximum(best_fit)))")
        if maxfit > best_fit[1]
            best_fit[1] = maxfit
            best[1] = copy(c.offspring[bestind])
            idx = sortperm(best_fit)
            best_fit = best_fit[idx]
            best = best[idx]
        end
        if sum(best_fit) >= 3900
            break
        end
        if length(best[1]) > 0 && i%20 == 0
            save_param(best_fit,best, tune)
        end
    end
    return best_fit, best
end

# %% run sim

optim()
