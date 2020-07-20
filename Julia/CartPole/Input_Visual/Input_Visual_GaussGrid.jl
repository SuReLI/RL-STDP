using PyCall
using JLD

gym = pyimport("gym")

torch = pyimport("torch")
nn    = pyimport("torch.nn")
f     = pyimport("torch.nn.functional")


# %%

include("../../Modules/Gridcells/StateSpikeGaussGrid.jl")
include("animation.jl")

# %% define the constant cells

env = gym.make("CartPole-v1")

cells_tmp_pos = HexModule[]
cells_tmp_vel = HexModule[]

maxs = env.observation_space.high
mins = env.observation_space.low

borpos = [convert(Float64,mins[1]),convert(Float64,maxs[1])]
borvpos = [-5.0,5.0]
#bortheta = [convert(Float64,mins[3]),convert(Float64,maxs[3])]
bortheta = [-pi,pi]
borvtheta = [-15.0,15.0]

for neuron in 1:16
    res = 0.07
    dilat = rand()*(1.4-0.7)+0.7
    theta = rand()*(2pi/6)-pi/6
    mod_pos = HexModule(res,dilat = dilat, theta = theta, bor_x = borpos, bor_y = bortheta)
    mod_vel = HexModule(res,dilat = dilat, theta = theta, bor_x = borvpos, bor_y = borvtheta)
    push!(cells_tmp_pos,mod_pos)
    push!(cells_tmp_vel,mod_vel)
end

const cells_pos = cells_tmp_pos
const cells_vel = cells_tmp_vel

env.close()

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

function convert_rgb(img::Array{UInt8,3})
    x,y,_ = size(img)
    img_rgb = fill(RGB(1.0,1.0,1.0),x,y)
    for i in 1:x
        for j in 1:y
            img_rgb[i,j] = RGB(img[i,j,1]/256,img[i,j,2]/256,img[i,j,3]/256)
        end
    end
    return img_rgb
end

function scatter_spikes(spikes::Array{Any,1})
    x = Int64[]
    y = Int64[]
    for i in 1:length(spikes)
        for j in eachindex(spikes[i])
            push!(x,i)
            push!(y,spikes[i][j])
        end
    end
    return x,y
end

function play_env(n_steps::Int64, render::Bool=false, random::Bool=false)
    env = gym.make("CartPole-v1")
    env.seed(0)
    obs = env.reset()
    done = false
    display_spikes = []
    display_img = []
    for j in 1:n_steps
        if random == false
            action = torch.argmax(Qnet(torch.tensor(obs).float())).item()
        else
            action = rand(0:1)
        end
        obs_new, reward, done, _ = env.step(action)
        if render==true
            # img = env.render(mode = "rgb_array")
            # push!(display_img, convert_rgb(img))
            env.render()
        end
        @show j
        @show obs_new
        view_spikes =input_spikes(obs_new,cells_pos,cells_vel)
        @show view_spikes
        push!(display_spikes, view_spikes)
        obs = obs_new
    end

    x_spikes, y_spikes = scatter_spikes(display_spikes)
    env.close()
    env = nothing
    save("Julia/CartPole/Heatmaps/disp_spikesGaussGrid.jld", "display_spikes", display_spikes)
    # scatter(x_spikes,y_spikes,
    #         title ="HexGrid input viz",
    #         markersize = 2,
    #         xlabel = "Time step",
    #         ylabel = "Neuron",
    #         legend = false)
    # plot!([64], seriestype = "hline", color = :black)
    #savefig("input_gaussgrid_rand.png")
    # plot()
    # anim = @animate for i in 1:n_steps
    #     anim_viz(x_spikes,y_spikes,display_img[i],160,i)
    # end
    # gif(anim, "Julia/Cartpole/Input_Visual/anim_grid.gif", fps = 20)
end



# %% run visualization


play_env(200,true,true)
