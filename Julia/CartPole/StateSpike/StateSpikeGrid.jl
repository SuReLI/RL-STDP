
include("../../Modules/HexGrid.jl")

# %%


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
#
# cells_tmp_pos = HexModule[]
# cells_tmp_vel = HexModule[]
#
#
# borpos = [-4.1,4.1]
# borvpos = [-50.0,50.0]
# bortheta = [-0.41,0.41]
# borvtheta = [-50.0,50.0]
#
# for neuron in 1:40
#     res = 0.05
#     dilat = rand()*(1.4-0.9)+0.9
#     theta = rand()*(2pi/6)-pi/6
#     mod_pos = HexModule(res,dilat = dilat, theta = theta, bor_x = borpos, bor_y = bortheta)
#     mod_vel = HexModule(res,dilat = dilat, theta = theta, bor_x = borvpos, bor_y = borvtheta)
#     push!(cells_tmp_pos,mod_pos)
#     push!(cells_tmp_vel,mod_pos)
# end
#
# const cells_pos = cells_tmp_pos
# const cells_vel = cells_tmp_vel
# #
# # %%
# obs = [-0.4577148456105377, 0.058890696314322866, 0.025517456325011803, -0.2936023243762348]
#
# spikes = input_spikes(obs,cells_pos,cells_vel)
