# %% imports

using JLD
using PyCall
using Statistics
gym = pyimport("gym")
include("../Modules/Networks/net_arch.jl")
include("../Modules/Poisson/PoissonStateSpike.jl")
include("cartpole_fitness.jl")
include("gensaver.jl")

# %% read the results file
#checker results
results = load("Julia/Actors/results_sNES_cartpole.jld", "results")
@show results.values["fit"]

# %% evaluate functions

function create_net(res::Results, indiv::Int64)
    params = YAML.load(open("Julia/Actors/cartpole_cfglif.yml"))
    arch = params["arch"]

    individual = get_indiv(res,indiv)
    genes = individual["genes"]
    
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
#
# run one episode with no seeds
function sim(res::Results, indiv::Int64)
    net, matalpha = create_net(res, indiv)
    reward = play_episode(net, matalpha)
    return reward
end

# function sim(res::Results, indiv::Int64)
#
#     individual = get_indiv(res,indiv)
#     genes = individual["genes"]
#     @show genes
#     # load the optimisation parameters
#     params = YAML.load(open("Julia/Actors/cartpole_cfglif.yml"))
#     arch = params["arch"]
#
#     # Input linear combination
#     genes_alpha = map(x->borne(x,-1.0, 1.0),genes[1:arch[1]*4])
#     matalpha = reshape(genes_alpha, (arch[1],4))
#
#     # Weights of the network
#     genes_w = genes[arch[1]*4+1:end]
#
#     # Initialize network
#     n = Network(arch, params)
#     exc, inh = ei_idx(arch)
#     set_weight!(n,genes_w, exc, inh)
#
#     # Compute the fitness
#     reward = play_episode(n, matalpha)
#     return reward # (put - if ES try to minimize by default)
# end




# %% evaluate the individuals

# test robustness

function test_random(res::Results)
    mean_reward = Float64[]
    for i in 1:res.n_indiv
        tmp_reward = []
        for ep in 1:1
            reward = sim(res,i)
            push!(tmp_reward, reward)
        end
        push!(mean_reward, mean(tmp_reward))
    end
    return mean_reward
end

function test_random(res::Results, indiv::Int64)
    tmp_reward = []
    for ep in 1:30
        reward = sim(res,i)
        push!(tmp_reward, reward)
    end
    mean_r = mean(tmp_reward)
    return mean_r
end

means = test_random(results)
