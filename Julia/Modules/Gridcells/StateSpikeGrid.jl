
include("../../Modules/Gridcells/HexGrid.jl")

# %%


function input_spikes(obs::Array{Float64}, cells_pos::Array{HexModule}, cells_vel::Array{HexModule})
    pos = [obs[1],obs[3]%(pi)]     # obs pos and theta
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
            close_vel = filter(p -> dist(p,vel) < d_vel, cells_vel[i].grids[j].grid)
            if length(close_pos) != 0
                neuron = neuron_passed + j
                push!(spikes,neuron)
            end
            if length(close_vel) != 0
                neuron = neuron_tot + neuron_passed + j
                push!(spikes,neuron)
            end
        end
        neuron_passed += cells_pos[i].n_grid
    end
    return spikes
end

# # %% test input spikes
#
# obs = [0.5,0.3,0.4,0.2]
#
# borpos = [0,1]
# borvpos = [0,1]
# bortheta = [0,1]
# borvtheta = [0,1]
#
# cells = HexModule[]
#
# for neuron in 1:20
#     res = 0.05
#     dilat = rand()*(1.4-0.9)+0.9
#     theta = rand()*(2pi/6)-pi/6
#     mod = HexModule(res,dilat = dilat, theta = theta)
#     push!(cells,mod)
# end
#
# spikes = input_spikes(obs,cells,cells)

# # %%
#
# cells_tmp_vel = HexModule[]
#
#
# borvpos = [-50.0,50.0]
# borvtheta = [-50.0,50.0]
#
# for neuron in 1:16
#     res = 0.05
#     dilat = rand()*(1.4-0.9)+0.9
#     theta = rand()*(2pi/6)-pi/6
#     mod_vel = HexModule(res,dilat = dilat, theta = theta, bor_x = borvpos, bor_y = borvtheta)
#     push!(cells_tmp_vel,mod_vel)
# end
#
# const cells_vel = cells_tmp_vel
#
# # %%
# obs1 = [0.8513784840643901, 2.047026939313532, 2.8789489142285225, -8.070231180900828]
# obs2 = [0.015865733422793675, -0.5221358389902105, -0.09518066236104987, 0.49140483556663733]
#
# spikes = input_spikes(obs1,cells_vel,cells_vel)
