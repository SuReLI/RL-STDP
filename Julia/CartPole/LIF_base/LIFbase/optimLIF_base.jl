# %% imports

include("../../../Modules/CMAES/cmaes.jl")
include("LIF_basesim.jl")
include("../../../Modules/save_param.jl")

# %%


function optim()
    #N_params = length(tunekeys)
    N_alpha = 8*4
    N_w = 8*8 + 8*8
    N = N_alpha + N_w
    #N = N_weights + N_params
    best = Array{Float64}[]
    for i in 1:20
        push!(best,[])
    end
    best_fit = [-1.0 for i in 1:20]



    # Create initial point (optional)
    results = load_param("Julia/CartPole/LIF_base/results/results_LIFbase888_bound.jld")
    genes_init = results["weights"][12]

    lambda = 4+floor(Int64,3*log(N))
    mu = floor(Int64,lambda/2)
    c = CMAES(genes_init, N=N, μ=mu, λ=lambda , τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N))
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
        if sum(best_fit) >= 4000
            break
        end
        if length(best[1]) > 0 && i%20 == 0
            save_param(best_fit,best)
        end
    end
    return best_fit, best
end

function viz_optim(best)
    tunekeys = ["imax", "vthresh"]
    for i in 1:length(best)
        params = Dict(tunekeys .=> rescale(best[i], tunekeys))
        @show params
    end
end

# %%

optim()
