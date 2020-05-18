using Discretizers

function lindisc(observation_space, neurons::Array{Int64} = [200,150,100,150])

    maxs::Array{Float64} = observation_space.high
    mins::Array{Float64} = observation_space.low

    pos_edges = range(mins[1],stop=maxs[1],length=neurons[1]+1)
    v_edges = range(-50.0,stop=50,length=neurons[2]+1)
    theta_edges = range(mins[3],stop=maxs[3],length=neurons[3]+1)
    thetav_edges = range(-50.0,stop=50.0,length=neurons[4]+1)

    lindisc_pos = LinearDiscretizer(pos_edges)
    lindisc_v = LinearDiscretizer(v_edges)
    lindisc_theta = LinearDiscretizer(theta_edges)
    lindisc_thetav = LinearDiscretizer(thetav_edges)

    lindisc_ = [lindisc_pos,lindisc_v,lindisc_theta,lindisc_thetav]

    return lindisc_
end

function encode_obs(obs::Array{Float64},lindisc_::Array{Discretizers.LinearDiscretizer{Float64,Int64},1})
    obs_lin = zeros(4)
    obs_bis = copy(obs)
    for i in 1:4
        obs_lin[i] = encode(lindisc_[i],obs_bis[i])
    end
    return obs_lin
end

function input_spikes(obs::Array{Float64}, obs_pre::Array{Float64},
                     lindisc_::Array{Discretizers.LinearDiscretizer{Float64,Int64},1},
                     neurons::Array{Int64} = [200,150,100,150])
    spikes = Int64[]
    obs_lin = encode_obs(obs, lindisc_)
    obs_pre_lin = encode_obs(obs_pre, lindisc_)
    delta = obs_lin .- obs_pre_lin
    for i in eachindex(delta)
        for j in 1:abs(delta[i])
            if delta[i] > 0
                push!(spikes, obs_lin[i]-j+2*sum(neurons[1:i-1]))
            elseif delta[i] < 0
                push!(spikes, obs_lin[i]+j+2*sum(neurons[1:i-1])+neurons[i])
            end
        end
    end
    return spikes
end
