# %% imports

include("../../../Modules/save_param.jl")
using Statistics
using PyCall
gym = pyimport("gym")
include("../../../Modules/Networks/net_arch.jl")
include("../../../Modules/Poisson/PoissonStateSpike.jl")
include("../../../Modules/simviz.jl")

# %% Read the results

results = load_param("Julia/CartPole/LIF_base/results/results_EI888.jld")

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

function set_weight!(net::Network, genes_w::Array{Float64})
    w = net.s.e
    for mat_idx in 1:length(w)
        for idx in eachindex(w[mat_idx])
            offset = sum([length(mat) for mat in w[1:mat_idx-1]])
            net.s.e[mat_idx][idx] = (genes_w[idx+offset]+1)/2
        end
    end
end

function set_weight_bound!(net::Network, genes_w::Array{Float64})
    w = net.s.e
    for mat_idx in 1:length(w)
        for idx in eachindex(w[mat_idx])
            offset = sum([length(mat) for mat in w[1:mat_idx-1]])
            net.s.e[mat_idx][idx] = borne(genes_w[idx+offset], -2.0, 4.0)
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
                net.s.e[mat_idx][idx] = borne(genes_w[end], -2.0, 0.0)
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

function sig(z::Float64, alpha::Float64)
    return 1.0/(1.0+exp(-z/alpha))
end


function norm_sig(obs::Array{Float64})
    pos = obs[1] # in [-2.4,2.4]
    theta = mod(obs[3]+pi,2pi) - pi # in [-pi, pi]
    vel = [obs[2], obs[4]] # in [-15,15]
    alpha_pos = 2.4/5
    alpha_theta = pi/5
    alpha_vel = 15/5
    pos_scaled = sig(pos,alpha_pos)
    theta_scaled = sig(theta,alpha_theta)
    vel_scaled = sig.(vel,alpha_vel)
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    return obs_scaled
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
        #obs_net = matalpha*norm_sig(obs)
        obs_net = matalpha*norm_obs(obs)
        spikes = int_spikes(obs_net) # state2spike function
        net.v .= net.param["c"]
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
    genes_alpha = map(x->borne(x,-1.0, 1.0), genes[1:32])
    genes_w = genes[33:end]
    matalpha = reshape(genes_alpha, (8,4))
    #params = generate(genes_p,tunekeys)
    params = YAML.load(open("Julia/CartPole/LIF_base/EI/cfglifEI.yml"))
    arch = [8,8,8]
    n = Network(arch, params)
    exc, inh = ei_idx(arch)
    set_weight!(n,genes_w, exc, inh)
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

function test_random(results::Dict, indiv::Int64)
    genes = results["weights"][indiv]
    tmp_reward = []
    for ep in 1:30
        reward, _, _ = sim(genes)
        push!(tmp_reward, reward)
    end
    mean_r = mean(tmp_reward)
    return mean_r
end

means = test_random(results)




# %% visualize the activity in the network

function activiz(results::Dict, indiv::Int64)
    genes = results["weights"][indiv]
    r, spike_hist, arch = sim(genes)
    spike_histogram(arch, spike_hist)
end

for idx in [19,20]
    activiz(results, idx)
end

# %% run one episode

genes = results["weights"][20]
r, spike_hist, arch = sim(genes, false, 500)

# %%
using Plots

for indiv in 19:20
    display(histogram(map(x->borne(x,0.0, 1.1), results["weights"][indiv][33:end-1]), xlabel = "indiv $(indiv)",  bins = collect(0.0:0.1:1.1)))
    println("Inh weight $(indiv)= $(borne(results["weights"][indiv][end], -2.0, 0.0))")
end

# %% visualize matalpha
gr()
histogram(map(x->borne(x,-1.0,1.0),results["weights"][19][1:32]), bins = collect(-1.0:0.2:1.0))

# %% definitve matalpha try
alpha_ar = map(x->borne(x,-1.0,1.0),results["weights"][19][1:32])

@show alpha_ar
# %% study matalpha

matalpha = reshape(alpha_ar, (8,4))
@show matalpha
