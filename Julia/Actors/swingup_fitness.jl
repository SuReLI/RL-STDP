# %% imports

using PyCall
gym = pyimport("gym")

using Cambrian

include("../Modules/Networks/net_arch.jl")
include("../Modules/Poisson/PoissonStateSpike.jl")
include("../Modules/Swingup/swingup.jl")

# %% function utils

function norm_swingup(obs::Array{Float64})
    pos = obs[1]
    theta = mod(obs[3]+pi, 2pi) -pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    theta_scaled = (theta + pi)/(2pi)
    vel_scaled = (vel .+ 7.5)/15
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
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

# %% play episode

function play_episode(net::Network, matalpha::Array{Float64,2}, n_steps::Int64 = 200)
    env = gym.make("CartPole-v1")
    env.seed(0)
    rng_action = Random.MersenneTwister(0)
    win_size = 40 # ms
    tot_reward = 0
    N_rand = 0
    N_tot = 0

    obs = swingup_reset!(env)

    @inbounds for j in 1:n_steps

        #Input computation
        obs_net = matalpha*norm_swingup(obs)
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
        n_left = div(net.arch[end],2)
        n_right = net.arch[end] - n_left
        action, sl, sr = left_right(count_tot, n_left, n_right)

        # if indecision reduce reward
        fac_reward = 1
        if sl == sr
            action = rand(rng_action, 0:1)
            fac_reward = 0.1
            N_rand += 1
        end
        N_tot += 1

        # gym env step
        obs_new, reward, done, _ = swingup_step(env,action,true)

        # increment total reward
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
    params = YAML.load(open("Julia/Actors/swingup_cfglif.yml"))
    arch = params["arch"]

    # Input linear combination
    genes_alpha = map(x->borne(x,-1.0, 1.0),genes[1:arch[1]*4])
    matalpha = reshape(genes_alpha, (arch[1],4))

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

# %% test play episode
# params = YAML.load(open("Julia/Actors/swingup_cfglif.yml"))
# arch = params["arch"]
# n = Network(arch,params)
# matalpha = randn(4,4)
# tot_reward = play_episode(n,matalpha)
