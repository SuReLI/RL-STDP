
function norm_obs(obs::Array{Float64})
    pos = obs[1]
    theta = obs[3]%pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 0.6)/1.2
    theta_scaled = (theta + pi)/(2*pi)
    vel_scaled = (vel .+ 3.75)/7.5
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    return obs_scaled
end
# %%
# poisson spikes for 4 neuron input
function poisson_spike(obs::Array{Float64})
    spikes = []
    obs_s = norm_obs(obs)
    for idx in eachindex(obs_s)
        value = rand()
        if value > obs_s[idx]
            push!(spikes, idx)
        end
    end
    spikes
end

# # no stochasticity input spikes
function int_spikes(obs::Array{Float64})
    spikes =  Tuple{Int64,Float64}[]
    obs_s = norm_obs(obs)
    for idx in eachindex(obs_s)
        push!(spikes, (idx,obs_s[idx]))
    end
    return spikes
end

# # %% test int_spikes
#
# using PyCall
# gym = pyimport("gym")
#
# function test_intspikes()
#     env = gym.make("CartPole-v1")
#     env.seed(0)
#     obs = env.reset()
#     for step in 1:60
#         action = rand(0:1)
#         @show obs
#         spikes = int_spikes(obs)
#         @show spikes
#         obs_new, _, done, _ = env.step(action)
#         if done
#             break
#         end
#         obs = obs_new
#     end
# end
#
# # %%
#
# test_intspikes()
