# %% imports
using Random
using LinearAlgebra
using Statistics
using Plots
using YAML
# %% classes

cfg = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/CartPole/CMAES/cfg_cmaes.yml"))

mutable struct CMAES
    N::Int
    μ::Int
    λ::Int
    τ::Float64
    τ_c::Float64
    τ_σ::Float64
    population::Array{Array{Float64}}
    offspring::Array{Array{Float64}}
    F_μ::Array{Float64}
    F_λ::Array{Float64}
    C::Array{Float64}
    s::Array{Float64}
    s_σ::Array{Float64}
    σ::Float64
    E::Array{Float64}
    W::Array{Float64}
    x::Array{Float64}
    cfg::Dict
end

function CMAES(init::Array{Float64} ;N=2, μ=1, λ=10, τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N), cfg::Dict = cfg)
    x = init
    x = clamp.(x, -1.0, 1.0)
    population = fill(x, µ)
    offspring = Array{Array{Float64}}(undef, λ)
    F_µ = Inf .* ones(µ)
    F_λ = Inf .* ones(λ)
    C = Array(Diagonal{Float64}(I, N))
    s = zeros(N)
    s_σ = zeros(N)
    σ = 1.0
    E = zeros(N, λ)
    W = zeros(N, λ);
    CMAES(N, μ, λ, τ, τ_c, τ_σ, population, offspring, F_µ, F_λ, C, s, s_σ, σ, E, W, x, cfg)
end

function step!(c::CMAES; obj=objective, visualize=false, anim=Nothing)
    # L1
    rng_offspring = Random.MersenneTwister(1234)
    sqrt_c = cholesky((c.C + c.C') / 2.0).U
    for i in 1:c.λ
        c.E[:,i] = randn(rng_offspring, c.N)
        c.W[:,i] = c.σ * (sqrt_c * c.E[:,i])
        c.offspring[i] = c.x + c.W[:,i]
        c.offspring[i] = clamp.(c.offspring[i], -1.0, 1.0)
        c.F_λ[i] = obj(c.offspring[i])
    end
    # Select new parent population
    idx = sortperm(c.F_λ)[1:c.μ]
    for i in 1:c.μ
        c.population[i] = c.offspring[idx[i]]
        c.F_μ[i] = c.F_λ[idx[i]]
    end
    # L2
    w = vec(mean(c.W[:,idx], dims=2))
    c.x += w
    # L3
    c.s = (1.0 - 1.0/c.τ)*c.s + (sqrt(c.μ/c.τ * (2.0 - 1.0/c.τ))/c.σ)*w
    # L4
    c.C = (1.0 - 1.0/c.τ_c).*c.C + (c.s./c.τ_c)*c.s'
    # L5
    ɛ = vec(mean(c.E[:,idx], dims=2))
    c.s_σ = (1.0 - 1.0/c.τ_σ)*c.s_σ + sqrt(c.μ/c.τ_σ*(2.0 - 1.0/c.τ_σ))*ɛ
    # L6
    c.σ = c.σ*exp(((c.s_σ'*c.s_σ)[1] - c.N)/(2*c.N*sqrt(c.N)))
    if visualize
        plot(xs, ys, fz, st=:contour)
        scatter!([c.offspring[i][1] for i in 1:c.λ], [c.offspring[i][2] for i in 1:c.λ],
            xlims=(-5, 5), ylims=(-5, 5), legend=:none)
        scatter!([c.x[1]], [c.x[2]], color=:black, marker=:rect,
            xlims=(-5, 5), ylims=(-5, 5), legend=:none)
        frame(anim)
    end
    c
end

function plot_obj()
    c = CMAES()
    println("x initial: ", c.x)
    anim = Animation()
    for i in 1:100
        v = mod(i, 1) == 0
        step!(c, visualize=v, anim=anim)
    end
    println("x final: ", c.x)
    gif(anim)
end

# %% test

# solution = [3.5, -0.2]
# sphere(x::Array{Float64}) = sum((x .- solution).^2)
#
# xs = -5.0:0.1:5.0
# ys = -5.0:0.1:5.0
#
# objective = sphere # sphere, himmelblau, styblinski_tang, rastrigin
# fz(x, y) = objective([x, y])
# println(solution) # optimal for sphere and rastrigin
# plot_obj()
