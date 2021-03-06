# %% imports

include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")
include("../Teacher/load_net_2.jl")
using Statistics
using PyCall
gym = pyimport("gym")

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
    return action, sl, sr
end

function norm_obs_actor(obs::Array{Float64})
    pos = obs[1]
    theta = mod(obs[3]+pi, 2pi) - pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    cone = 41.8*pi/180
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

function state_critic(obs::Array{Float64})
    #norm state
    pos = obs[1]
    theta = mod(obs[3]+pi,2pi)-pi
    vel = [obs[2], obs[4]]
    pos_scaled = (pos + 2.4)/4.8
    cone = 12*pi/180
    theta_scaled = (theta + cone)/(2*cone)
    vel_scaled = (vel .+ 7.5)/15
    obs_scaled = [pos_scaled, vel_scaled[1], theta_scaled, vel_scaled[2]]
    #v, 1-v
    state_critic = Float64[]
    for i in 1:length(obs_scaled)
        push!(state_critic, obs_scaled[i])
        push!(state_critic, 1 - obs_scaled[i])
    end
    return state_critic
end

function norm_state_critic(obs::Array{Float64})
    #norm state
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



# %% play decisions functions

function play_critic(n::Network, tderror::Float64, state_alpha::Array{Float64})
    n.da += exp(-abs(tderror)*n.param["beta"])
    n.tderror = tderror
    win_size = 40 #ms
    spikes = int_spikes(state_alpha)
    #n.v .= n.param["c"]
    value_trace_exc = 0.0
    value_trace_inh = 0.0
    #calculate index of last layer neuron
    value_neuron_exc = [1,2,3,4] .+ sum([n.arch[i] for i in 1:length(n.arch)-1])
    value_neuron_inh = [5,6,7,8] .+ sum([n.arch[i] for i in 1:length(n.arch)-1])
    @inbounds for msec in 1:win_size
        spiked = step_LIF_critic!(n, spikes, true)
        for idx in 1:4
            if value_neuron_exc[idx] in spiked
                value_trace_exc += n.param["vthresh"]-n.param["c"]
            end
            if value_neuron_inh[idx] in spiked
                value_trace_inh += n.param["vthresh"]-n.param["c"]
            end
        end
        rate = (1-1/n.param["taulif"])
        value_trace_exc *= rate
        value_trace_inh *= rate
    end

    new_value_exc = sum(n.v[value_neuron_exc]) + value_trace_exc
    new_value_inh = sum(n.v[value_neuron_inh]) + value_trace_inh
    new_value = abs(new_value_exc - new_value_inh)

    return new_value
end

function play_actor(n::Network, state_alpha::Array{Float64})
    win_size = 40 #ms
    spikes = int_spikes(state_alpha)
    n.v .= n.param["c"]
    count_tot = zeros(n.arch[end])

    @inbounds for msec in 1:win_size
        spiked = step_LIF!(n, spikes, false)
        if msec >= length(n.arch)-1
            count_tot .+= count_spikes(n.arch, spiked, length(n.arch))
            #push!(spike_hist,spiked)
        end
    end

    n_left = div(n.arch[end],2)
    n_right = n.arch[end] - n_left
    action, sl, sr = left_right(count_tot, n_left, n_right)

    return action, sl, sr
end

function step_actor(state::Array{Float64}, action::Int64)
    env = gym.make("CartPole-v1")
    env.reset()
    env.env.state = state
    new_state, r, done, _ = env.step(action)
    env.close()
    env = nothing
    return new_state,r,done
end

# %% Compute play episode ( calculate tderror )

function play_episode(n_actor::Network, n_critic::Network, matalpha::Array{Float64,2}, n_steps::Int64 = 200)
    # initial point
    env = gym.make("CartPole-v1")
    env.seed(0)
    state = env.reset()
    env.close()
    env = nothing
    Base.GC.gc()

    # matalpha


    # init tderror and Value fonction
    tderror = 2.0
    #in_critic = matalpha_critic * norm_state_critic(state)
    in_critic = state_critic(state)
    value = play_critic(n_critic, tderror, in_critic)
    @show value

    td_errors = []

    # sim loop
    for j in 1:n_steps

        state_alpha = matalpha * norm_cartpole(state)
        action, sl, sr = play_actor(n_actor, state_alpha)
        new_state,r,done = step_actor(state,action)

        new_in_critic = state_critic(new_state)

        new_value = play_critic(n_critic, tderror, new_in_critic)
        @show new_value

        #calculate tderror
        gamma = 0.95
        tderror =  r + gamma*new_value - value
        push!(td_errors, abs(tderror))

        value = new_value
        state = new_state

        # evaluate mean of weights ( and max )
        w = n_critic.s.e
        w_line = []
        for i in 1:length(w)
            w_line = [w_line ; w[i]...]
        end

        @show j
        @show tderror

        if done
            println("DONE")
            break
        end
    end
    return tderror, td_errors
end

# %% compute objective function ( CMAES )

function borne(x,a,b)
    y = a + 0.5*(b-a)*(1+cos(pi*x/10))
    return y
end

function mod_genes(gene::Float64, hyp::String)
    if hyp=="tde"
        tau = borne(gene,50.0,200.0)
        mg = (1-1/tau)
    end
    if hyp=="ttrace"
        tau = borne(gene, 5.0, 20.0)
        mg = (1-1/tau)
    end
    if hyp=="taulif"
        mg = borne(gene, 1.0, 15.0)
    end
    if hyp=="eta"
        pow = borne(gene, 1.0, 5.0)
        mg = 10^(-pow)
    end
    if hyp=="winh"
        mg = borne(gene, -1.0, 0.0)
    end
    if hyp=="beta"
        mg = borne(gene, 1.0,10.0)
    end
    if hyp=="phi"
        mg = borne(gene, 5.0,20.0)
    end
    if hyp=="apost"
        pow = borne(gene, 0.0, 2.0)
        mg = 1.0*10^(-pow)
    end
    if hyp=="apre"
        pow = borne(gene, 0.0, 2.0)
        mg = 1.5*10^(-pow)
    end
    return  mg
end

function assert_genes!(net::Network, genes::Array{Float64}, hyp::Array{String})
    #modify genes
    modified_gene = Float64[]
    for i in 1:length(hyp)
        g = genes[i]
        h = hyp[i]
        mod = mod_genes(g,h)
        push!(modified_gene, mod)
    end
    for i in 1:length(hyp)
        net.param[hyp[i]] = modified_gene[i]
    end
end


function objective(genes::Array{Float64})
    #create the actor and critic
    filepath = "Julia/CartPole/LIF_base/results/results_EI888.jld"
    n_actor, _ = init_teacher_student(filepath, 19)

    arch_critic = [8,8,2]
    params = YAML.load(open("Julia/Cartpole/Criticv2/cfg_default_critic.yml"))

    n_critic = Network(arch_critic, params)

    dic = YAML.load(open("Julia/Cartpole/Criticv2/cfg_cmaes.yml"))
    hyp = String[]
    for k in keys(dic["tunekeys"])
        push!(hyp, k)
    end
    genes_hyp = genes[1:length(hyp)]
    genes_alpha = genes[length(hyp)+1:end]
    alpha_ar = map(x->borne(x, 0.0, 10.0),genes_alpha)
    matalpha = reshape(alpha_ar, (4,4))
    assert_genes!(n_critic, genes_hyp, hyp)
    td_errors = Float64[]
    # play a certain number of episode (count in terms of number of decision to be more precise)
    for epoch in 1:20
        td_error, td_errors = play_episode(n_actor, n_critic, matalpha)
    end

    #redefine fitness !!!!!
    fit = maximum(td_errors)
    @show fit
    return fit
end

#%% sim function ( no cmaes )

function sim()
    #create the actor and critic
    filepath = "Julia/Actors/cartpoleElites/sNESelites_cartpole_1.jld"
    n_actor, _ , matalpha= init_teacher_student(filepath, 1)

    arch_critic = [8,8,8]
    params = YAML.load(open("Julia/Cartpole/Criticv2/cfg_default_critic.yml"))
    n_critic = Network(arch_critic, params)

    # play a certain number of episode (count in terms of number of decision to be more precise)
    td_error = 0
    td_errors = Float64[]
    weights = [[],[]]
    for epoch in 1:100
        @show epoch
        td_error, tderrs = play_episode(n_actor, n_critic, matalpha)
        #average weight of each layer
        for i in eachindex(n_critic.s.e)
            push!(weights[i], mean(n_critic.s.e[i]))
        end
        @show td_error
        #weight_viz(n_critic, "heatmap")
        td_errors = [td_errors ; tderrs]
    end


    return td_errors, weights
end

# %% tests on sim

td_errs, weights = sim()
# %%

plot(td_errs, legend = false, xlabel = "Time (msec)", ylabel = "TD-error")

# %% check activity in net
plot(weights[1],label = "Mean synaptic strength layer 1-2", xlabel = "Episode", ylabel = "Synaptic strength", color = :blue )
plot!(weights[2], label = "Mean synaptic strength layer 2-3", color = :green)
