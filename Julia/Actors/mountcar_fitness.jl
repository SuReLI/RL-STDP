# %% imports

using PyCall
gym = pyimport("gym")

using Cambrian

include("../Modules/Networks/net_arch.jl")
include("../Modules/Poisson/PoissonStateSpike.jl")

# %% function utils

function norm_mountcar(obs::Array{Float64})
    pos = obs[1]
    vel = obs[2]
    pos_scaled = (pos + 1.2)/(1.8)
    vel_scaled = (vel + 0.07)/0.14
    obs_scaled = [pos_scaled, vel_scaled]
    return obs_scaled
end

function count_spikes(arch::Array{Int64}, spiked::Array{Int64}, layer::Int64)
    count_layer = zeros(arch[layer])
    interval = [sum(arch[1:layer-1])+1, sum(arch[1:layer])]
    spikes_in = filter(x-> interval[1] <= x <= interval[2], spiked)
    spikes_in .-= sum(arch[1:layer-1])
    count_layer[spikes_in] .+= 1
    return count_layer
end


function push_or_not(count_spikes::Array{Float64}, n_left::Int64, n_middle::Int64, n_right::Int64)
    sl = sum(count_spikes[1:n_left])
    sm = sum(count_spikes[n_left+1:n_left+n_middle])
    sr = sum(count_spikes[n_left+n_middle+1:end])
    action = argmax([sl,sm,sr]) - 1
    return action, sl, sm, sr
end

function borne(x,a,b)
    y = a + 0.5*(b-a)*(1+cos(pi*x/10))
    return y
end

# %% play episode

function play_episode(net::Network, matalpha::Array{Float64,2}, n_steps::Int64 = 200)
    env = gym.make("MountainCar-v0")
    env.seed(0)
    rng_action = Random.MersenneTwister(0)
    win_size = 40 # ms
    tot_reward = 0
    N_rand = 0
    N_tot = 0

    obs = env.reset()

    @inbounds for j in 1:n_steps

        #Input computation
        obs_net = matalpha*norm_mountcar(obs)
        spikes = int_spikes(obs_net)

        count_tot = zeros(net.arch[end])
        net.v .= net.param["c"]

        # Count last layer spikes for win_size msec
        @inbounds for msec in 1:win_size
            spiked = step_LIF!(net, spikes, false)
            if msec >= length(net.arch)-1
                count_tot .+= count_spikes(net.arch, spiked, length(net.arch))
            end
        end

        # Decide action
        n_left = div(net.arch[end],3) #actual number of neurons attributed to the action selection in last layer
        n_middle = n_left
        n_right = net.arch[end] - 2*n_left
        action, sl, sm, sr = push_or_not(count_tot, n_left, n_middle, n_right)

        # if indecision increase negative reward (punish)
        fac_reward = 1
        difs = [sl-sr, sl-sm, sm-sr]
        pools = [[1,3], [1,2], [2,3]]
        for i in eachindex(difs)
            if difs[i] == 0.0
                action = rand(rng_action, pools[i]) - 1
                fac_reward = 2
                N_rand += 1
            end
        end
        N_tot += 1

        # gym env step
        obs_new, reward, done, _ = env.step(action)
        tot_reward += reward*fac_reward

        if done
            break
        end

        obs = obs_new

    end
    println("Rand = ", N_rand, "/", N_tot, ": R = ", tot_reward)
    env.close()
    env = nothing
    Base.GC.gc()
    return tot_reward
end

# %% Weight setting functions

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
                net.s.e[mat_idx][idx] = borne(genes_w[end], -2.0, 0.0)
                inh_count += 1
            end
        end
    end
end

# %% fitness function

function fitness(i::Individual)
    genes = i.genes
    # load the optimisation parameters
    params = YAML.load(open("Julia/Actors/mountcar_cfglif.yml"))
    arch = params["arch"]

    # Input linear combination
    genes_alpha = map(x->borne(x,-1.0, 1.0),genes[1:arch[1]*2])
    matalpha = reshape(genes_alpha, (arch[1],2))

    # Weights of the network
    genes_w = genes[arch[1]*4+1:end]

    # Initialize network
    n = Network(arch, params)
    exc, inh = ei_idx(arch)
    set_weight!(n,genes_w, exc, inh)

    # Compute the fitness
    reward = play_episode(n, matalpha)
    return [reward] # (put - if ES try to minimize by default)
end
