module StateSpike

#### imports
using Discretizers

#### exports
export input_spikes
export lindisc

#### functions

function lindisc(observation_space)

    maxs::Array{Float64} = observation_space.high
    mins::Array{Float64} = observation_space.low

    pos_edges = range(mins[1],stop=maxs[1],length=12000)
    v_edges = range(-10.0,stop=10,length=2000)
    theta_edges = range(mins[3],stop=maxs[3],length=1000)
    thetav_edges = range(-10.0,stop=10.0,length=2000)

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


function input_spikes(obs::Array{Float64},obs_pre::Array{Float64},lindisc_::Array{Discretizers.LinearDiscretizer{Float64,Int64},1})
    spikes = zeros(8)
    obs_lin = encode_obs(obs,lindisc_)
    obs_pre_lin = encode_obs(obs_pre,lindisc_)
    delta = obs_lin .- obs_pre_lin
    spikes[1:4] .= clamp.(delta,0.0,1.0)
    spikes[5:8] .= clamp.(-delta,0.0,1.0)
    return spikes
end

end #end module
