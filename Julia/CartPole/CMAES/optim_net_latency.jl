# %% imports

include("cmaes.jl")
include("CMAES_latency.jl")

# %%


function optim()
    tunekeys = ["vthresh", "ae", "b", "c", "de",
                "m", "std", "apost", "apre"]
    cfg = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/CartPole/CMAES/cfg_cmaes.yml"))
    N=length(tunekeys)
    best = Array{Float64}[]
    for i in 1:10
        push!(best,[])
    end
    best_fit = [-1.0 for i in 1:10]
    init_params = [cfg[tunekeys[idx]]*1.0 for idx in 1:N]
    init_genes  = descale(init_params, tunekeys)
    lambda = 4 + floor(Int64, 3*log(N))
    mu = floor(Int64, lambda/2)
    c = CMAES(init_genes, N=N, μ=mu, λ=lambda, τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N))
    for i in 1:100
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
        if sum(best_fit) > 700
            break
        end
    end
    return best_fit, best
end

# %% run optim

best_fit, best = optim()

# %%

function viz_optim(best)
    tunekeys = ["vthresh", "ae", "b", "c", "de",
                 "m", "std", "apost", "apre"]
    for i in 1:length(best)
        params = Dict(tunekeys .=> rescale(best[i], tunekeys))
        @show params
    end
end

# %%
viz_optim(best)
