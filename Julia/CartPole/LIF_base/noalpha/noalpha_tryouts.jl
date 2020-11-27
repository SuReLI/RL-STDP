# %% imports

include("../../../Modules/save_param.jl")
using Statistics
using PyCall
gym = pyimport("gym")
include("../../../Modules/Networks/net_arch.jl")
include("../../../Modules/Poisson/PoissonStateSpike.jl")
include("../../../Modules/simviz.jl")

# %% Read the results

results = load_param("Julia/CartPole/LIF_base/results/results_noalpharandn.jld")

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


function sig(z::Float64, alpha::Float64)
    return 1.0/(1.0+exp(-z/alpha))
end

function norm_obs(obs::Array{Float64})
    pos = obs[1]
    theta = obs[3]%pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    cone = 41.8*pi/180
    theta_scaled = (theta + cone)/(2*cone)
    vel_scaled = (vel .+ 7.5)/15
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    return obs_scaled
end

function norm_sig(obs::Array{Float64})
    pos = obs[1] # in [-2.4,2.4]
    theta = obs[3]%pi # in [-pi, pi]
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

function obs_input(obs::Array{Float64}, gain::Float64)
    input = Float64[]
    obs_s = norm_obs(obs)
    for idx in eachindex(obs_s)
        anti_obs = gain*(1.0-obs_s[idx])
        push!(input, obs_s[idx])
        push!(input, anti_obs)
    end
    return input
end




# %% main functions


function play_episode(net::Network, n_steps::Int64 = 500)
    env = gym.make("CartPole-v1")
    env.seed(0)
    #obs = swingup_reset!(env)
    obs = env.reset()
    rng_action = Random.MersenneTwister(0)
    win_size = 100 # ms
    tot_reward = 0
    N_rand = 0
    N_tot = 0
    matalpha =  randn((4,4))
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        obs_alpha = matalpha*obs
        input = obs_input(obs_alpha, 1.0)
        spikes = int_spikes(input) # state2spike function
        #reset voltage
        net.v .= net.param["c"]
        influence = 2
        @inbounds for msec in 1:win_size
            spiked = step_LIF!(net, spikes, false)
            if msec >= length(net.arch)-1
                count_tot .+= influence*count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.995
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
        #obs_new, reward, done, _ = swingup_step(env,action)
        obs_new, reward, done, _ = env.step(action)
        tot_reward += reward
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
            net.s.e[mat_idx][idx] = borne(genes_w[idx+offset], 0.0, 1.1)
        end
    end
end

function sim(genes, render::Bool = false, n_steps::Int64 = 500)
    #load the optimisation parameters
    optim_param = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes.yml"))
    arch = optim_param["arch"]
    genes_w = genes

    params = YAML.load(open("Julia/CartPole/LIF_base/noalpha/cfglifnoalpha.yml"))
    n = Network(arch, params)
    set_weight!(n,genes_w)
    reward = play_episode(n, n_steps)
    return reward, n.arch
end

# %% test robustness

function test_random(results::Dict)
    mean_reward = Float64[]
    for i in 1:length(results["fit"])
        genes = results["weights"][i]
        tmp_reward = []
        for ep in 1:30
            reward, _, _ = sim(genes, false, 200)
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
        reward, _ = sim(genes, false, 200)
        push!(tmp_reward, reward)
    end
    mean_r = mean(tmp_reward)
    return mean_r
end

means = test_random(results,20)




# %% visualize the activity in the network

function activiz(results::Dict, indiv::Int64)
    genes = results["weights"][indiv]
    r, spike_hist, arch = sim(genes)
    spike_histogram(arch, spike_hist)
end

for idx in [20]
    activiz(results, idx)
end

# %% run one episode

genes = results["weights"][20]
r, spike_hist, arch = sim(genes, true, 500)

# %%
using Plots

for indiv in 20
    display(histogram(map(x->borne(x,0.0, 1.1), results["weights"][indiv]), xlabel = "indiv $(indiv)",  bins = collect(0.0:0.1:1.1)))
end
