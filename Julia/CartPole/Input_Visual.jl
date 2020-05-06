using PyCall

gym = pyimport("gym")

torch = pyimport("torch")
nn    = pyimport("torch.nn")
f     = pyimport("torch.nn.functional")

# %% modules

include("StateSpike.jl")

using .StateSpike
using Plots

# %% network

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
# %% functions



function play_env(render=false)
    env = gym.make("CartPole-v1")
    env.seed(0)
    obs = env.reset()
    lindisc_ = lindisc(env.observation_space)
    done = false
    display_spikes = zeros(8)
    for _ in 1:100
        action = torch.argmax(Qnet(torch.tensor(obs).float())).item()
        obs_new, reward, done, _ = env.step(action)
        if render==true
            env.render()
            sleep(0.05)
        end
        display_spikes = hcat(display_spikes, input_spikes(obs_new,obs,lindisc_))
        obs = obs_new
    end
    x_spikes,y_spikes = scatter_spikes(display_spikes)
    println("Plotting...")
    display(scatter(x_spikes, y_spikes , title = "Input visualisation",
                    markersize = 2, yticks = collect(1:8), legend = false))
    env.close()
    env = nothing
end

function scatter_spikes(spikes::Array{Float64,2})
    x = Int64[]
    y = Int64[]
    for linindex in eachindex(spikes)
        if spikes[linindex]>0.5
            push!(x, (linindex%8==0) ? div(linindex,8) : div(linindex,8) + 1)
            push!(y, (linindex%8==0) ? 8 : linindex%8)
        end
    end
    return x,y
end



# %% run visualization

play_env(Qnet)
