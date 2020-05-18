include("../../Modules/HexGrid.jl")

function dist(p1::Array{Float64},p2::Array{Float64})
    d = sqrt((p1[1]-p2[1])^2+(p1[2]-p2[2])^2)
    return d
end


gauss(d::Float64, node_size::Float64) = exp(-2*(d/node_size))


function input_spikes(obs::Array{Float64}, cells_pos::Array{HexModule}, cells_vel::Array{HexModule})
    pos = [obs[1],obs[3]]     # obs pos and theta
    vel = [obs[2],obs[4]]     # obs pos_v and theta_v
    n_module = length(cells_pos)
    spikes = Int64[]
    neuron_tot = sum([cells_pos[i].n_grid for i in 1:n_module])
    neuron_passed = 0
    for i in 1:n_module
        d_pos = cells_pos[i].grids[1].node_size
        d_vel = cells_vel[i].grids[1].node_size
        for j in 1:cells_pos[i].n_grid
            close_pos = filter(p -> dist(p,pos) < d_pos, cells_pos[i].grids[j].grid)
            close_vel =  filter(p -> dist(p,vel) < d_vel, cells_vel[i].grids[j].grid)
            if length(close_pos) != 0
                neuron = neuron_passed + j
                d = dist(pos,close_pos[1])
                if rand() < gauss(d,d_pos)
                    push!(spikes,neuron)
                end
            end
            if length(close_vel) != 0
                neuron = neuron_tot + neuron_passed + j
                d = dist(pos,close_vel[1])
                if rand() < gauss(d,d_vel)
                    push!(spikes,neuron)
                end
            end
        end
        neuron_passed += cells_pos[i].n_grid
    end
    return spikes
end

# %% test input_spikes

cells_tmp_pos = HexModule[]
cells_tmp_vel = HexModule[]


borpos = [-4.1,4.1]
borvpos = [-50.0,50.0]
bortheta = [-pi,pi]
borvtheta = [-50.0,50.0]

for neuron in 1:40
    res = 0.05
    dilat = rand()*(1.4-0.9)+0.9
    theta = rand()*(2pi/6)-pi/6
    mod_pos = HexModule(res,dilat = dilat, theta = theta, bor_x = borpos, bor_y = bortheta)
    mod_vel = HexModule(res,dilat = dilat, theta = theta, bor_x = borvpos, bor_y = borvtheta)
    push!(cells_tmp_pos,mod_pos)
    push!(cells_tmp_vel,mod_pos)
end

const cells_pos = cells_tmp_pos
const cells_vel = cells_tmp_vel

# %%

obs = [0.5,0.3,0.4,0.2]


spikes = input_spikes(obs,cells_pos,cells_vel)
