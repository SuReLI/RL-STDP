# %% imports

using PyCall
gym = pyimport("gym")

include("../../../Modules/Networks/net_arch.jl")
include("../../../Modules/Poisson/PoissonStateSpike.jl")

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

function borne(x,a,b)
    y = a + 0.5*(b-a)*(1+cos(pi*x/10))
    return y
end

# %%

function play_episode(net::Network, matalpha::Array{Float64,2}, n_steps::Int64 = 200)
    env = gym.make("CartPole-v1")
    env.seed(0)
    obs = env.reset()
    rng_action = Random.MersenneTwister(0)
    win_size = 40 # ms
    #spike_hist = Array{Int64}[]
    tot_reward = 0
    N_rand = 0
    N_tot = 0
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        obs_net = matalpha*obs
        spikes = int_spikes(obs_net) # state2spike function
        #reset voltage
        net.v .= net.param["c"]
        @inbounds for msec in 1:win_size
            spiked = step_LIF!(net, spikes, false)
            influence = 2
            if msec >= length(net.arch)-1
                count_tot .+= influence*count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.95
                #push!(spike_hist,spiked)
            end
        end
        n_left = div(net.arch[end],2)
        n_right = net.arch[end] - n_left
        action, sl, sr = left_right(count_tot, n_left, n_right)
        if sl == sr
            action = rand(rng_action, 0:1)
            tot_reward -= 0.9
            N_rand += 1
        end
        N_tot += 1
        #push!(score_hist[1], sl)
        #push!(score_hist[2], sr)
        obs_new, reward, done, _ = env.step(action)
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
    return tot_reward
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

function set_weight!(net::Network, genes_w::Array{Float64}, exc::Array{Int64}, inh::Array{Int64})
    w = net.s.e
    inh_count = 0
    for mat_idx in 1:length(w)
        for idx in eachindex(w[mat_idx])
            offset = sum([length(mat) for mat in w[1:mat_idx-1]]) - inh_count
            offset_neuron = sum(net.arch[1:mat_idx-1])
            (idx%size(w[mat_idx])[1]==0) ? line = size(w[mat_idx])[1] : line = idx%size(w[mat_idx])[1]
            current_neuron = line+offset_neuron
            if current_neuron in exc
                # net.s.e[mat_idx][idx] = abs(genes_w[idx+offset])
                net.s.e[mat_idx][idx] = borne(genes_w[idx+offset], 0.0, 1.1)
            elseif current_neuron in inh
                # net.s.e[mat_idx][idx] = - abs(genes_w[end])
                net.s.e[mat_idx][idx] = borne(genes_w[end], -3.0, 0.0)
                inh_count += 1
            end
        end
    end
end


function ei_idx(arch::Array{Int64})
    inh = Int64[]
    exc = Int64[]
    n_inh = [div(arch[i],4) for i in 1:length(arch)-1]
    push!(n_inh, 0)
    for l in 1:length(arch)
        for neuron in 1:arch[l]
            offset = sum(arch[1:l-1])
            if neuron <= n_inh[l]
                push!(inh, neuron+offset)
            else
                push!(exc, neuron+offset)
            end
        end
    end
    return exc, inh
end

# function to be completed as new hyparams arrive
function eval(key::String, gene::Float64, param::Dict)
    if key=="vthresh"
        value = borne(gene, 0.9, 4.0) + param["c"]
    end
    if key=="gain_I"
        value = abs(gene)
    end
    if key=="taulif"
        value = borne(gene, 1.0, 20.0)
    return value
end


function objective(genes::Array{Float64})
    #load the optimisation parameters
    optim_param = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes.yml"))
    arch = optim_param["arch"]
    tunekeys = optim_param["tunekeys"]
    n_keys = length("tunekeys")

    # separate the genum
    genes_key = genes[1:n_key]
    genes_alpha = map(x->borne(x,-1.0, 1.0),genes[n_key+1:arch[1]*4])
    genes_w = genes[n_key+arch[1]*4+1:end]

    # create the network according to the genum
    params = YAML.load(open("Julia/CartPole/LIF_base/cfglif4820.yml"))
    for idx,key in enumerate(keys(tunekeys))
        params[key] = eval(key, genes[idx],params)
    end
    n = Network(arch, params)
    exc, inh = ei_idx(arch)
    set_weight!(n,genes_w, exc, inh)
    matalpha = reshape(genes_alpha, (arch[1],4))
    reward = play_episode(n, matalpha)
    return -reward
end
