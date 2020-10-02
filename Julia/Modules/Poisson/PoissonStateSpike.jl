# %% imports



function norm_obs(obs::Array{Float64})
    pos = obs[1]
    theta = mod(obs[3]+pi, 2pi) - pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    cone = 12*pi/180
    theta_scaled = (theta + cone)/(2*cone)
    vel_scaled = (vel .+ 7.5)/15
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    return obs_scaled
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
function int_spikes_normed(obs::Array{Float64})
    spikes =  Tuple{Int64,Float64}[]
    obs_s = norm_sig(obs)
    for idx in eachindex(obs_s)
        push!(spikes, (idx,obs_s[idx]))
    end
    return spikes
end


# # no stochasticity input spikes
function int_spikes(obs::Array{Float64})
    spikes =  Tuple{Int64,Float64}[]
    for idx in eachindex(obs)
        push!(spikes, (idx,obs[idx]))
    end
    return spikes
end

# # %% test int_spikes
