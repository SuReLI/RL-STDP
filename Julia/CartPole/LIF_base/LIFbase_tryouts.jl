# %% imports

include("save_param.jl")
using Statistics
using PyCall
gym = pyimport("gym")
include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")
include("../../Modules/simviz.jl")

# %% Read the results

results = load_param("Julia/CartPole/LIF_base/results.jld")

# %% sim functions

function count_spikes(arch::Array{Int64}, spiked::Array{Int64}, layer::Int64)
    count_layer = zeros(arch[layer])
    interval = [sum(arch[1:layer-1])+1, sum(arch[1:layer])]
    spikes_in = filter(x-> interval[1] <= x <= interval[2], spiked)
    spikes_in .-= sum(arch[1:layer-1])
    count_layer[spikes_in] .+= 1
    return count_layer
end


function left_right(count_spikes::Array{Float64}, n_left::Int64, n_right::Int64)
    sl = sum(count_spikes[1:n_left])
    sr = sum(count_spikes[n_left+1:end])
    (sl >= sr) ? action = 0 : action = 1
    return action, sl, sr
end

function set_weight!(net::Network, genes_w::Array{Float64})
    w = net.s.e
    for mat_idx in 1:length(w)
        for idx in eachindex(w[mat_idx])
            offset = sum([length(mat) for mat in w[1:mat_idx-1]])
            net.s.e[mat_idx][idx] = (genes_w[idx+offset]+1)/2
        end
    end
end


# %% main functions

function play_episode(net::Network, matalpha::Array{Float64,2}, render::Bool = false, n_steps::Int64 = 200)
    env = gym.make("CartPole-v1")
    obs = env.reset()
    #rng_action = Random.MersenneTwister(0)
    win_size = 40 # ms
    spike_hist = Array{Int64}[]
    tot_reward = 0
    N_rand = 0
    N_tot = 0
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        obs_net = matalpha*obs
        spikes = int_spikes(obs_net) # state2spike function
        @inbounds for msec in 1:win_size
            spiked = step_LIF!(net, spikes, false)
            influence = 2
            if msec >= length(net.arch)-1
                count_tot .+= influence*count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.95
                push!(spike_hist,spiked)
            end
        end
        n_left = div(net.arch[end],2)
        n_right = net.arch[end] - n_left
        action, sl, sr = left_right(count_tot, n_left, n_right)
        if sl == sr
            action = rand(0:1)
            tot_reward -= 0.9
            N_rand += 1
        end
        N_tot += 1
        #push!(score_hist[1], sl)
        #push!(score_hist[2], sr)
        obs_new, reward, done, _ = env.step(action)
        if render==true
            env.render()
        end
        tot_reward += 1
        if done
            break
        end
        obs = obs_new
    end
    println("Rand = ", N_rand, "/", N_tot)
    env.close()
    env = nothing
    Base.GC.gc()
    return tot_reward, spike_hist
end


function sim(genes, render::Bool = false, n_steps::Int64 = 200)
    genes_alpha = genes[1:16]
    genes_w = genes[17:end]
    matalpha = reshape(genes_alpha, (4,4))
    #params = generate(genes_p,tunekeys)
    params = YAML.load(open("Julia/CartPole/LIF_base/cfglif4820.yml"))
    n = Network([4,8,20], params)
    set_weight!(n,genes_w)
    reward, spike_hist = play_episode(n, matalpha, render, n_steps)
    return reward, spike_hist, n.arch
end

# %% test robustness

function test_random(results::Dict)
    mean_reward = Float64[]
    for i in 1:length(results["fit"])
        genes = results["weights"][i]
        tmp_reward = []
        for ep in 1:30
            reward, _, _ = sim(genes)
            push!(tmp_reward, reward)
        end
        push!(mean_reward, mean(tmp_reward))
    end
    return mean_reward
end

means = test_random(results)

# %% Inhibitor vs. Exciting populations

function test_inhexc(results::Dict)
    inh = Float64[]
    for i in 1:length(results["fit"])
        n, n_i = 0, 0
        for w in results["weights"][i]
            n += 1
            n_i += (w<0)
        end
        push!(inh, n_i/n)
    end
    return inh
end

inh = test_inhexc(results)


# %% visualize the activity in the network

function activiz(results::Dict, indiv::Int64)
    genes = results["weights"][indiv]
    r, spike_hist, arch = sim(genes)
    spike_histogram(arch, spike_hist)
end


activiz(results, 19)

# %%

genes = results["weights"][20]
r, spike_hist, arch = sim(genes, true, 500)
