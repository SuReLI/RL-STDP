# %% imports

include("../../../Modules/CMAES/cmaes.jl")
include("noalpha_sim.jl")
include("../../../Modules/save_param.jl")


# %%


function optim()
    cfg = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes.yml"))
    arch = cfg["arch"]
    N = sum([arch[i]*arch[i-1] for i in 2:length(arch)])


    best = Array{Float64}[]
    for i in 1:20
        push!(best,[])
    end
    best_fit = [-1.0 for i in 1:20]
    error = false

    lambda = 4+floor(Int64,3*log(N))
    mu = floor(Int64,lambda/2)
    c = CMAES( N=N, μ=mu, λ=lambda , τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N))
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
        if sum(best_fit) >= 10000
            break
        end
        if i > 20 && best_fit[end] < 150
            error = true
            break
        end
        if length(best[1]) > 0 && i%20 == 0
            save_param(best_fit,best)
        end
    end
    return best_fit, best, error
end

# %% run sim

for opt in 1:20
    best_fit, best, error = optim()
    if error==false
        break
    end
end
