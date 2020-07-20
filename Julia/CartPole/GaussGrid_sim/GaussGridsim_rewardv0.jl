# %% imports

using PyCall
gym = pyimport("gym")

include("../../Modules/Gridcells/StateSpikeGaussGrid.jl")
include("../../Modules/Networks/net_arch.jl")
include("simviz.jl")

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
dilats = [1, sqrt(2), 2, 2*sqrt(2), 4]

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
    if sl == sr
        action = rand(0:1)
    end
    return action, sl, sr
end



# %% Simulation

global s_hist = [[],[]]

function run_sim(net::Network, n_steps::Int64, render::Bool=false;
                 method::String = "snn")
    env = gym.make("CartPole-v1")
    env.seed(0)
    obs = env.reset()
    spike_history = Array{Int64}[]
    reward_count = 0
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        spikes = input_spikes(obs, cells_pos, cells_vel)
        influence = 3
        @inbounds for msec in 1:10
            (msec==10) ? train=true : train=false
            spiked = step!(net, spikes, train)
            if msec >= length(net.arch)-1
                count_tot .+= influence .* count_spikes(net.arch, spiked, length(net.arch))
                influence *= 0.95
                push!(spike_history,spiked)
            end
        end
        if method == "snn"
            n_left = div(net.arch[end],2)
            n_right = net.arch[end] - n_left
            action, sl, sr = left_right(count_tot, n_left, n_right)
            push!(s_hist[1], sl)
            push!(s_hist[2], sr)
        elseif method == "trained"
            n_left = div(net.arch[end],2)
            n_right = net.arch[end] - n_left
            action = torch.argmax(Qnet(torch.tensor(obs).float())).item()
            _, sl, sr = left_right(count_tot, n_left, n_right)
            push!(s_hist[1], sl)
            push!(s_hist[2], sr)
        elseif method == "random"
            n_left = div(net.arch[end],2)
            n_right = net.arch[end] - n_left
            action = rand(0:1)
            _, sl, sr = left_right(count_tot, n_left, n_right)
            push!(s_hist[1], sl)
            push!(s_hist[2], sr)
        end
        obs_new, reward, done, _ = env.step(action)
        reward_count += reward
        if reward == 1
            net.da += 0.1 # parameter
        end
        if render==true
            env.render()
        end
        obs = obs_new
    end
    return spike_history, reward_count
end


# %% Create the network

# randomize the syn_weight mean = 2, std = 2 # parameters
function randomize!(net::Network)
    net.s_ee .= randn(net.n_exc, net.n_exc)*0.5 .+ 2
    net.s_ee .= clamp.(net.s_ee, 0.0, net.param["wmax"])
    net.s_ee .*= net.connection_ee

end

parameters = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/Modules/cfg.yml"))

n = Network([128,64,64,80], parameters)
randomize!(n)

global rewards = Int64[]

# %% run the simulation

@time spike_history,reward = run_sim(n, 500, method = "snn")
push!(rewards, reward)

# for _ in 1:9
#     @time spike_history,reward = run_sim(n, 500, method = "snn")
#     push!(rewards, reward)
# end

# %% Plots
x = collect(1:length(s_hist[1]))
plot(x,s_hist,
     xlabel = "Time step cartpole",
     ylabel = "Sum spikes left and right",
     lab = ["left" "right"],
     title = "Left or Right plot")


# %% Plots difference
x = collect(1:length(s_hist[1]))
y = s_hist[2] .- s_hist[1]
 plot(x,y,
      xlabel = "Time step cartpole",
      ylabel = "Sum spikes left minus right",
      title = "Difference between left or right",
      legend = false)


# %% layer history
layer_spike_hist(n.arch, spike_history, 1)



# %% histograms
spike_histogram(n.arch, spike_history)

# %% weights
weight_viz(n, "heatmap")

# %%
weight_viz(n, "histogram")

# %%
 heatmap(n.connection)

# %%
plot(rewards)
