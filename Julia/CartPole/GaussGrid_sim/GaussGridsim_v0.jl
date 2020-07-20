# %% imports

using PyCall
gym = pyimport("gym")

include("../../Modules/Gridcells/StateSpikeGaussGrid.jl")
include("../../Modules/Networks/net_arch.jl")
include("simviz.jl")

# %% Build the grids

include("../../Modules/Gridcells/GridGenerator.jl")

grid_cells = grid_gen()




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
    @inbounds for j in 1:n_steps
        count_tot = zeros(net.arch[end])
        spikes = input_spikes(obs, grid_cells)
        influence = 2
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
        if render==true
            env.render()
        end
        obs = obs_new
    end
    return spike_history
end




parameters = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/Modules/cfg.yml"))

n = Network([120,60,80], parameters)
#randomize!(n)


# %% run the simulation

@time spike_history = run_sim(n, 200, method = "snn")

for _ in 1:9
    @time spike_history .= run_sim(n, 200, method = "snn")
end


# %% Plots difference
x = collect(1:length(s_hist[1]))
y = s_hist[2] .- s_hist[1]
plot(x,y,
     xlabel = "Time step cartpole",
     ylabel = "Sum spikes left minus right",
     title = "Difference between left or right",
     legend = false)

# %% Plot s_hist
x = collect(1:length(s_hist[1]))
plot(x,s_hist,
     xlabel = "Time step cartpole",
     ylabel = "Sum spikes left and right",
     lab = ["left" "right"],
     title = "Left or Right plot")


# %% spike history plot per layer
layer_spike_hist(n.arch, spike_history, 1)

# %% spike distribution for all layers
spike_histogram(n.arch, spike_history)

# %% weights visualization heatmap
weight_viz(n, "heatmap")

# %% weights visualization histogram
weight_viz(n, "histogram")

# %% visualize network connection
heatmap(n.connection)
