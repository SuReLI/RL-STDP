# %% imports

using PyCall
gym = pyimport("gym")

include("../../Modules/Gridcells/StateSpikeGaussGrid.jl")
include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Gridcells/GridGenerator.jl")



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
    if sl == sr
        action = rand(0:1)
    end
    return action, sl, sr
end

# %% fitness function play 100 episodes and check les contraintes a chaque épisode

function play_episode(net::Network, n_steps::Int64 = 200)
    env = gym.make("CartPole-v1")
    obs = env.reset()
    win_size = 10 # ms
    grid_cells = grid_gen()
    spike_hist = Array{Int64}[]
    score_hist = [[],[]]
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        spikes = input_spikes(obs, grid_cells)
        influence = 2
        @inbounds for msec in 1:win_size
            (msec==10) ? train=true : train=false
            spiked = step!(net, spikes, train)
            if msec >= length(net.arch)-1
                count_tot .+= influence .* count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.95
                push!(spike_hist,spiked)
            end
        end
        n_left = div(net.arch[end],2)
        n_right = net.arch[end] - n_left
        action, sl, sr = left_right(count_tot, n_left, n_right)
        push!(score_hist[1], sl)
        push!(score_hist[2], sr)
        obs_new, reward, done, _ = env.step(action)
        obs = obs_new
    end
    env.close()
    env = nothing
    Base.GC.gc()
    return score_hist, spike_hist
end

# CONSTRAINTS
function constraints(score::Array{Array{Any,1},1}, spikes::Array{Array{Int64}}, net::Network)
    mw_min = 0.0 #FILL
    mw_max = 1.0#FILL
    s_min = 1000#FILL
    s_max = 15000#FILL
    m_min = -5.0#FILL
    m_max = 5.0 #FILL
    # constraint #1 weight average
    weights = net.s.e
    for layer in 1:length(net.arch)-1
        mw = mean(weights[layer])
        if (mw < mw_min) | (mw > mw_max)
            println("-----> weights mean out of bounds")
            return true
        end
    end
    # constraint #2 layer activity
    for layer in 1:length(net.arch)
        count_layer = zeros(net.arch[layer])
        for i in 1:length(spikes)
            count_layer .+= count_spikes(net.arch, spikes[i], layer)
        end
        spikes_layer = sum(count_layer)
        if (spikes_layer < s_min) | (spikes_layer > s_max)
            println("-----> spike activity out of bounds")
            return true
        end
    end
    # constraint #3 mean left right
    mlr = mean(score[1] .- score[2])
    if (mlr > m_max) | (mlr < m_min)
        println("-----> left right mean out of bounds")
        return true
    end
    return false
end

# plays 100 episodes and returns the fitness ( and tells wether to drop batch )
function fitness(episodes::Int64, net::Network)
    score_hist = [[0.,0.],[0.,0.]]
    drop = false
    pond = 0
    for episode in 1:episodes
        score, spikes = play_episode(net)
        drop = constraints(score, spikes, net)
        push!(score_hist[1],score[1]...)
        push!(score_hist[2],score[2]...)
        if drop
            break
        end
        pond += 2
    end
    fit = std(score_hist[1] .- score_hist[2]) + pond
    return fit
end



# %% objective function ( fitness + contraintes ) ( create new network and play 100 episodes with fitness function )

function rescale(genes::Array{Float64}, tunekeys::Array{String})
    cfg = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/CartPole/CMAES/cfg_cmaes.yml"))
    N = length(genes)
    scaled_genes = zeros(N)
    for idx in 1:N
        down = cfg[tunekeys[idx]*"_d"]
        up = cfg[tunekeys[idx]*"_u"]
        scaled_genes[idx] = ((genes[idx]+1)*0.5)*(up-down) + down
    end
    return scaled_genes
end

function descale(params::Array{Float64}, tunekeys::Array{String})
    cfg = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/CartPole/CMAES/cfg_cmaes.yml"))
    N = length(tunekeys)
    descaled_genes = zeros(N)
    for idx in 1:N
        down = cfg[tunekeys[idx]*"_d"]
        up = cfg[tunekeys[idx]*"_u"]
        descaled_genes[idx] = 2*(params[idx]-down)/(up-down) - 1
    end
    return descaled_genes
end

function generate(genes::Array{Float64})
    tunekeys = ["vthresh", "ae", "b", "c", "de","imin", "imax",
                "m", "std", "apost", "apre"]
    params = Dict(tunekeys .=> rescale(genes,tunekeys))
    keys = ["tda", "wmax" ,"theta", "ttheta", "ai", "di", "wei", "wie"]
    values = [0.995, 1, 0.0, 0.998, 0.1, 2.0, 0.0, 0.0]
    for idx in 1:length(keys)
        params[keys[idx]] = values[idx]
    end
    return params
end


function objective(genes::Array{Float64})
    params = generate(genes)
    n = Network([120,60,80],params)
    fit = fitness(50,n)
    -fit
end

# %% tests
#
# objective([-50.0,0.02,0.2,-65.0,8.0,0.5,0.1,0.0,0.998,1.0,1.5])
