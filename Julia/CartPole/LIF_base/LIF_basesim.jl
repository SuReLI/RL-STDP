# %% imports

using PyCall
gym = pyimport("gym")

include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")

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

# %%

function play_episode(net::Network, n_steps::Int64 = 200)
    env = gym.make("CartPole-v1")
    obs = env.reset()
    rng_action = Random.MersenneTwister(0)
    win_size = 20 # ms
    #spike_hist = Array{Int64}[]
    tot_reward = 0
    N_rand = 0
    N_tot = 0
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        spikes = int_spikes(obs) # state2spike function
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

function rescale(genes::Array{Float64}, tunekeys::Array{String})
    cfg = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes_lif.yml"))
    N = length(genes)
    scaled_genes = zeros(N)
    n_param = length(tunekeys)
    for idx in 1:N
        if idx <= n_param
            down = cfg[tunekeys[idx]*"_d"]
            up = cfg[tunekeys[idx]*"_u"]
            scaled_genes[idx] = ((genes[idx]+1)*0.5)*(up-down) + down
        elseif idx > n_param
            scaled_genes[idx] = (genes[idx]+1)*0.5
        end
    end
    return scaled_genes
end

function init_param()
    cfg = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes_lif.yml"))
    init_params = Dict()
    for key in cfg.keys
        if key[end]!="u" | key[end]!="d"
            init_params[key] = cfg[key]
        end
    end
    init_params["std"] = cfg["std"]
    return init_params
end


function generate(genes_p::Array{Float64}, tunekeys::Array{String})
    params = Dict(tunekeys .=> rescale(genes_p,tunekeys))
    keys = ["imin", "vthresh", "ae", "ai", "b", "c", "de", "di", "apost", "apre", "wie", "wei" ,"tda", "wmax", "theta", "ttheta", "std", "m", "taulif"]
    values = [0.0, -50, 0, 0, 0, -70, 0, 0, 0, 0, 0, 0, 0.995, 1, 0.0, 0.998, 0.2, 0.8, 0.8]
    for idx in 1:length(keys)
        params[keys[idx]] = values[idx]
    end
    return params
end

function objective(genes::Array{Float64})
    tunekeys = ["imax"]
    n_p = length(tunekeys)
    genes_p = genes[1:n_p]
    genes_w = genes[n_p+1:end]
    params = generate(genes_p,tunekeys)
    n = Network([4,8,2], params)
    set_weight!(n,genes_w)
    reward = play_episode(n)
    return -reward
end

# %%
