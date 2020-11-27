# %% imports
using Cambrian
using EvolutionaryStrategies
using JLD
using PyCall
using Statistics
gym = pyimport("gym")
include("../Modules/Networks/net_arch.jl")
include("../Modules/Poisson/PoissonStateSpike.jl")
include("cartpole_fitness.jl")


# %% read the results file
#checker results
elites = load("Julia/Actors/cartpoleElites/sNESelites_cartpole_1.jld", "elites")
# @show typeof(elites)

# %% evaluate functions

function create_net(elites::Array{AbstractESIndividual,1}, indiv::Int64)
    params = YAML.load(open("Julia/Actors/cartpole_cfglif.yml"))
    arch = params["arch"]

    genes = elites[indiv].genes

    # Input linear combination
    genes_alpha = map(x->borne(x,-1.0, 1.0),genes[1:arch[1]*4])
    matalpha = reshape(genes_alpha, (arch[1],4))

    # Weights of the network
    genes_w = genes[arch[1]*4+1:end]

    # Initialize network
    n = Network(arch, params)
    exc, inh = ei_idx(arch)
    set_weight!(n,genes_w, exc, inh)

    return n, matalpha
end

# run one episode with no seeds
function sim(elites::Array{AbstractESIndividual,1}, indiv::Int64)
    net, matalpha = create_net(elites, indiv)
    reward = play_episode(net, matalpha, rng_seed = false)
    return reward
end


# %% evaluate the individuals

# test robustness

function test_random(elites::Array{AbstractESIndividual,1})
    mean_reward = Float64[]
    for i in 1:length(elites)
        tmp_reward = []
        for ep in 1:30
            reward = sim(elites,i)
            push!(tmp_reward, reward)
        end
        push!(mean_reward, mean(tmp_reward))
    end
    return mean_reward
end

function test_random(elites::Array{AbstractESIndividual,1}, indiv::Int64)
    tmp_reward = []
    for ep in 1:30
        reward = sim(res,i)
        push!(tmp_reward, reward)
    end
    mean_r = mean(tmp_reward)
    return mean_r
end
