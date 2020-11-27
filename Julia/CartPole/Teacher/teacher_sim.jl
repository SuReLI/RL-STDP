# %% imports

using Statistics
using PyCall
gym = pyimport("gym")

include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")


# %% functions

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
    if sl==sr
        action = rand(0:1)
    end
    return action, sl, sr
end

function norm_obs(obs::Array{Float64})
    pos = obs[1]
    theta = mod(obs[3]+pi, 2pi) - pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    cone = 12*pi/180
    theta_scaled = (theta + pi)/(2pi)
    vel_scaled = (vel .+ 7.5)/15
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    return obs_scaled
end

function norm_cartpole(obs::Array{Float64})
    pos = obs[1]
    theta = mod(obs[3]+pi,2pi)-pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    cone = 12*pi/180
    theta_scaled = (theta + cone)/(2*cone)
    vel_scaled = (vel .+ 7.5)/15
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    return obs_scaled
end



# %% sim steps

function play_decision(net::Network, state_alpha::Array{Float64}, train::Bool)
    win_size = 40 #ms
    spikes = int_spikes(state_alpha)
    net.v .= net.param["c"]
    count_tot = zeros(net.arch[end])

    @inbounds for msec in 1:win_size
        spiked = step_LIF!(net, spikes, train)
        if msec >= length(net.arch)-1
            count_tot .+= count_spikes(net.arch, spiked, length(net.arch))
            #push!(spike_hist,spiked)
        end
    end

    n_left = div(net.arch[end],2)
    n_right = net.arch[end] - n_left
    action, sl, sr = left_right(count_tot, n_left, n_right)

    return action, sl, sr
end

function reward_net(state::Array{Float64}, action::Int64)
    env = gym.make("CartPole-v1")
    env.reset()
    env.env.state = state
    new_state, r, done, _ = env.step(action)
    env.close()
    env = nothing
    return r,new_state,done
end

function dopa_inc(r_stud::Real, r_teach::Real)
    if r_stud == r_teach
        da = 1.0
    elseif r_stud != r_teach
        da = - 200.0
    end
    return da
end

function play_episode(n_stud::Network, n_teach::Network, matalpha::Array{Float64,2}, n_steps::Int64 = 200)
    # initial point
    env = gym.make("CartPole-v1")
    state = env.reset()
    env.close()
    env = nothing
    Base.GC.gc()

    state_alpha = matalpha * norm_cartpole(state)

    #record da evo
    da_evo = []
    r_tot = 0
    sls = []
    srs = []
    # sim loop
    for j in 1:n_steps
        state_alpha = matalpha * norm_cartpole(state)

        action_stud, sl_stud, sr_stud = play_decision(n_stud, state_alpha, true)
        @show sl_stud-sr_stud
        push!(sls, sl_stud)
        push!(srs, sr_stud)
        action_teach, sl_teach, sr_teach = play_decision(n_teach, state_alpha, false)
        #@show sl_teach,sr_teach
        r_stud, new_state, done= reward_net(state, action_stud)
        r_teach, _,_ = reward_net(state,action_teach)
        # r_teach, new_state, done= reward_net(state, action_teach)
        # r_stud, _,_ = reward_net(state,action_stud)

        # Compare the rewards
        da_inc = dopa_inc(action_stud, rand(0:1))
        n_stud.da += da_inc * n_stud.param["da_inc"]
        push!(da_evo,n_stud.da)
        r_tot += r_stud

        if done
            break
        end
        state = new_state
    end
    ss = [mean(sls), mean(srs)]
    return r_tot,da_evo, ss
end

# %% tests

include("load_net_2.jl")

filepath = "Julia/Actors/cartpoleElites/sNESelites_cartpole_1.jld"

n_teach, n_stud, matalpha = init_teacher_student(filepath, 1)
weight_viz(n_stud)
rewards = []
sls = []
srs = []

for epoch in 1:1000
    @show epoch
    @time r_ep, da_evo, ss= play_episode(n_stud, n_teach, matalpha,200)
    @show r_ep
    push!(sls, ss[1])
    push!(srs, ss[2])
    #weight_viz(n_stud, "heatmap")
    #display(plot(da_evo))
    push!(rewards, r_ep)
end


# %% viz rewards

plot(rewards, label = "Student network", legend = true, xlabel = "Episode", ylabel = "Reward" )
plot!([200], seriestype = "hline", color = :green, label = "Teacher Network")

# %%
plot(sls, label= "Mean action score left", title = "Student action scores", xlabel = "Episode", ylabel = "Score")
plot!(srs, label = "Mean action score right")

# %%
plot(sls-srs)
