# %% imports

include("../../Modules/CMAES/cmaes.jl")
include("../../Modules/save_param.jl")
include("critic_sim.jl")

# %%

function optim()
    cfg = YAML.load(open("Julia/CartPole/Criticv2/cfg_cmaes.yml"))
    N_hyp = length(cfg["tunekeys"])
    hyp = String[]
    for k in keys(cfg["tunekeys"])
        push!(hyp, k)
    end
    N_alpha = 4*4
    N = N_hyp + N_alpha


    best = Array{Float64}[]
    for i in 1:20
        push!(best,[])
    end
    best_fit = [1000.0 for i in 1:20]
    error = false

    # Create initial point (optional)
    # results = load_param("Julia/CartPole/LIF_base/results_LIF88888EI.jld")
    # genes_init = results["weights"][20]

    lambda = 4+floor(Int64,3*log(N))
    mu = floor(Int64,lambda/2)
    c = CMAES( N=N, μ=mu, λ=lambda , τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N))
    for i in 1:1000
        @time step!(c)
        bestind = argmin(c.F_λ)
        minfit = c.F_λ[bestind]
        println(i," ", minfit, " (current best = $(minimum(best_fit)))")
        if minfit < best_fit[1]
            best_fit[1] = minfit
            best[1] = copy(c.offspring[bestind])
            idx = sortperm(best_fit, rev = true)
            best_fit = best_fit[idx]
            best = best[idx]
        end


        if length(best[1]) > 0 && i%20 == 0
            save_param(best_fit,best,hyp)
        end
    end

    return best_fit, best
end

# %%

optim()
