# %% imports
using Random
using PyCall
gym = pyimport("gym")

include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")
include("../../Modules/simviz.jl")


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
    env.seed(0)
    obs = env.reset()
    rng_action = Random.MersenneTwister(0)
    win_size = 20 # ms
    spike_hist = Array{Int64}[]
    vhist = Float64[]
    tot_reward = 0
    N_rand = 0
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        spikes = int_spikes(obs) # state2spike function
        @inbounds for msec in 1:win_size
            spiked = step_LIF!(net, spikes, false)
            influence = 2
            if msec >= length(net.arch)-1
                count_tot .+= influence*count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.95
                push!(spike_hist,spiked)
                push!(vhist,net.v[1])
            end
        end
        n_left = div(net.arch[end],2)
        n_right = net.arch[end] - n_left
        action, sl, sr = left_right(count_tot, n_left, n_right)
        if sl == sr
            action = rand(rng_action, 0:1)
            N_rand += 1
        end
        #push!(score_hist[1], sl)
        #push!(score_hist[2], sr)
        obs_new, reward, done, _ = env.step(action)
        tot_reward += 1
        if done
            break
        end
        obs = obs_new
    end
    println("Rand = ", N_rand, "/", tot_reward)
    env.close()
    env = nothing
    Base.GC.gc()
    return tot_reward,spike_hist,vhist
end

# %% tests
params = YAML.load(open("Julia/CartPole/LIF_base/cfglif.yml"))
n = Network([4,8,2], params)
r, shist,vhist = play_episode(n)




# %% viz
spike_histogram(n.arch, shist)
#%%
weight_viz(n,"histogram")
#%%

plot(vhist, legend=false)
