# %% imports
cd("Julia/CartPole/CMAES_colab")



using PyCall
gym = pyimport("gym")

include("StateSpikeGaussGrid.jl")
include("new_net_arch.jl")
include("CartpoleGridGenerator.jl")

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
    (sl > sr) ? action = 0 : action = 1
    return action, sl, sr
end

# %% fitness function play 100 episodes and check les contraintes a chaque épisode

function play_episode(net::Network, n_steps::Int64 = 200)
    env = gym.make("CartPole-v1")
    env.seed(0)
    rng_action = Random.MersenneTwister(0)
    obs = env.reset()
    grid_cells = grid_gen()
    win_size = 10 # ms
    reward_tot = 0
    @inbounds for j in 1:n_steps
        reward_tmp = 0
        count_tot = zeros(net.arch[end])
        spikes = input_spikes(obs, grid_cells)
        influence = 2
        @inbounds for msec in 1:win_size
            spiked = step!(net, spikes, false)
            if msec >= length(net.arch)-1
                count_tot .+= influence .* count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.95
            end
        end
        n_left = div(net.arch[end],2)
        n_right = net.arch[end] - n_left
        action, sl, sr = left_right(count_tot, n_left, n_right)
        if sl != sr
            reward_tmp += 0.01*(abs(sl-sr)) # essayer de mettre un plus petit coeff (0.001 ) pour que cela ai moins d'influence
        end
        if sl == sr
            action = rand(rng_action, 0:1)
            reward_tmp -= 0.9
        end
        obs_new, reward, done, _ = env.step(action)
        if done
            break
        end
        reward_tot += reward+reward_tmp
        obs = obs_new
    end
    env.close()
    env = nothing
    Base.GC.gc()
    return reward_tot
end


# %% objective function ( fitness + contraintes ) ( create new network and play 100 episodes with fitness function )

function rescale(genes::Array{Float64}, tunekeys::Array{String})
    cfg = YAML.load(open("cfg_cmaes.yml"))
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


function generate(genes_p::Array{Float64}, tunekeys::Array{String})
    params = Dict(tunekeys .=> rescale(genes_p,tunekeys))
    keys = ["tda", "wmax", "theta", "ttheta", "std", "m"]
    values = [0.995, 1, 0.0, 0.998, 0.2, 0.5]
    for idx in 1:length(keys)
        params[keys[idx]] = values[idx]
    end
    return params
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



function objective(genes::Array{Float64})
    tunekeys = ["vthresh", "ae", "b", "c", "de",
                "wei", "wie", "ai", "di", "imin", "imax"]
    n_params = length(tunekeys)
    genes_p = genes[1:n_params]
    genes_w = genes[n_params + 1 : end]
    params = generate(genes_p, tunekeys)
    n = Network([120,60,80],params)
    set_weight!(n,genes_w)
    fit = play_episode(n)
    @show fit
    -fit
end

# %% tests
