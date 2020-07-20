# %% imports

using Plots
using Statistics
using PyCall
gym = pyimport("gym")

include("../../Modules/Gridcells/StateSpikeGaussGrid.jl")
include("../../Modules/Networks/net_res.jl")

# %% Build the grids

include("../../Modules/Gridcells/HexGrid.jl")

env = gym.make("CartPole-v1")

cells_tmp_pos = HexModule[]
cells_tmp_vel = HexModule[]

maxs = env.observation_space.high
mins = env.observation_space.low

borpos = [convert(Float64,mins[1]),convert(Float64,maxs[1])]
borvpos = [-15.0,15.0]
#bortheta = [convert(Float64,mins[3]),convert(Float64,maxs[3])]
bortheta = [-pi,pi]
borvtheta = [-15.0,15.0]

for neuron in 1:16 # parameter
    res = 0.07 # parameter
    dilat = rand()*(1.4-0.7)+0.7 # parameter
    theta = rand()*(2pi/6)-pi/6
    mod_pos = HexModule(res,dilat = dilat, theta = theta, bor_x = borpos, bor_y = bortheta)
    mod_vel = HexModule(res,dilat = dilat, theta = theta, bor_x = borvpos, bor_y = borvtheta)
    push!(cells_tmp_pos,mod_pos)
    push!(cells_tmp_vel,mod_vel)
end

const cells_pos = cells_tmp_pos
const cells_vel = cells_tmp_vel

env.close()

# %% import the trained cartpole

torch = pyimport("torch")
nn    = pyimport("torch.nn")
f     = pyimport("torch.nn.functional")

@pydef mutable struct DQLnet <: nn.Module
    function __init__(self)
        pybuiltin(:super)(DQLnet, self).__init__()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,2)
    end

    function forward(self, s)
        s = f.relu(self.fc1(s))
        s = f.relu(self.fc2(s))
        s = self.fc3(s)
        return s
    end
end

Qnet = DQLnet()
Qnet.load_state_dict(torch.load("/Users/titou/Documents/PFE/RL-STDP/Julia/CartPole/saved_cartpole.net"))

# %% Function

# general function
function count_spikes(net::Network, spiked::Array{Int64}, layer::Int64)
    count_layer = zeros(net.arch[layer])
    interval = [sum(net.arch[1:layer-1])+1, sum(net.arch[1:layer])]
    spikes_in = filter(x-> interval[1] <= x <= interval[2], spiked)
    spikes_in .-= sum(net.arch[1:layer-1])
    count_layer[spikes_in] .+= 1
    return count_layer
end

# not general function ( cartpole specific )
function left_right(count_spikes::Array{Float64}, n_left::Int64, n_right::Int64)
    sl = sum(count_spikes[1:n_left])
    sr = sum(count_spikes[n_left+1:end])
    (sl >= sr) ? action = 0 : action = 1
    return action, sl, sr
end



# %% Simulation

function run_sim(net::Network, n_steps::Int64, layer::Int64, render::Bool=false;
                 method::String = "snn")
    env = gym.make("CartPole-v1")
    env.seed(0)
    obs = env.reset()
    count_lay = zeros(net.arch[layer])
    @inbounds for j in 1:n_steps
        spikes = input_spikes(obs, cells_pos, cells_vel)
        count_tot = zeros(net.arch[end])
        @inbounds for msec in 1:10
            (msec==10) ? train=true : train=false
            spiked = step!(net, spikes, train)
            if msec >= length(net.arch)-1
                count_tot .+= count_spikes(net, spiked, length(net.arch))
                count_lay .+= count_spikes(net, spiked, layer)
            end
        end
        if method == "snn"
            n_left = div(net.arch[end],2)
            n_right = net.arch[end] - n_left
            action, sl, sr = left_right(count_tot, n_left, n_right)
        elseif method == "trained"
            n_left = div(net.arch[end],2)
            n_right = net.arch[end] - n_left
            action = torch.argmax(Qnet(torch.tensor(obs).float())).item()
            _, sl, sr = left_right(count_tot, n_left, n_right)
        elseif method == "random"
            n_left = div(net.arch[end],2)
            n_right = net.arch[end] - n_left
            action = rand(0:1)
            _, sl, sr = left_right(count_tot, n_left, n_right)
        end
        obs_new, reward, done, _ = env.step(action)
        if render==true
            env.render()
        end
        obs = obs_new
    end
    return count_lay
end

function histo(net::Network, n_steps::Int64, layer::Int64, render::Bool=false;
               method::String = "snn")
    count_lay = run_sim(net, n_steps, layer, render, method = method)
    weights = net.s_exc
    filtered = filter(x->x!=0.0, weights)
    display(histogram(filtered, legend = false))
    if layer == 4
        l = div(net.arch[layer],2)
        ml = mean(count_lay[1:l])
        mr = mean(count_lay[l+1:end])
        m = [(i<=l) ? ml : mr for i in 1:net.arch[layer]]
        display(bar(m, color = :orange))
        display(bar!(count_lay,
                    xlabel = "neuron",
                    ylabel = "spiking distribution",
                    legend = false,
                    color = :blue,
                    title = " Spiking in layer $(layer)"))
    else
        display(bar(count_lay,
                    xlabel = "neuron",
                    ylabel = "spiking distribution",
                    legend = false,
                    title = " Spiking in layer $(layer)"))
    end
end


# %% Create the network


# randomize the syn_weight from [1 to 3] # parameters
function randomize!(net::Network)
    net.s_exc .+= rand(net.n_neurons, net.n_exc)*2
    net.s_exc .*= net.connection_exc
end

n = Network([128,64,64,80]) # parameter
randomize!(n)

# %% run the simulation

@time histo(n,200,4,method = "trained")
