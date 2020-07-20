# %% imports

include("CMAES_GaussGrid.jl")
include("cmaes.jl")
include("save_param.jl")



# %%


function optim()


    tunekeys = ["vthresh", "ae", "b", "c", "de",
                "wei", "wie", "ai", "di", "imin", "imax"]
    N_p = length(tunekeys)
    N_w = 120*60 + 60*80
    N = N_p + N_w
    best = Array{Float64}[]
    for i in 1:20
        push!(best,[])
    end
    best_fit = [-1.0 for i in 1:20]
    lambda = 4+floor(Int64,3*log(N))
    mu = floor(Int64,lambda/2)
    c = CMAES(N=N, μ=mu, λ=lambda , τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N))
    for i in 1:2000
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
        if sum(best_fit) > 4000
            break
        end
        if length(best[1]) > 0 && i%20 == 0
            save_param(best_fit,best,tunekeys)
        end
    end
    return best_fit, best
end

# %% run optim

best_fit, best = optim()

# %%

function viz_optim(best)
    tunekeys = ["vthresh", "ae", "b", "c", "de",
                "wei", "wie", "ai", "di", "imin", "imax"]
    for i in 1:length(best)
        params = Dict(tunekeys .=> rescale(best[i], tunekeys))
        @show params
    end
end

# %%
viz_optim(best)

# %%
