include("../../Modules/Gridcells/HexGrid.jl")

function dist(p1::Array{Float64},p2::Array{Float64})
    d = sqrt((p1[1]-p2[1])^2+(p1[2]-p2[2])^2)
    return d
end

                                # for the intensity to index coding (never equals one)
gauss(d::Float64, node_size::Float64) = 0.999999 * exp(-2*(d/node_size)) # considered 2sigma as d in the gaussian


function input_spikes(obs::Array{Float64}, cells_pos::Array{HexModule}, cells_vel::Array{HexModule})
    pos = [obs[1],obs[3]%pi]     # obs pos and theta
    vel = [obs[2],obs[4]]     # obs pos_v and theta_v
    n_module = length(cells_pos)
    spikes = Tuple{Int64,Float64}[]
    neuron_tot = sum([cells_pos[i].n_grid for i in 1:n_module])
    neuron_passed = 0
    for i in 1:n_module
        d_pos = cells_pos[i].grids[1].node_size
        d_vel = cells_vel[i].grids[1].node_size
        for j in 1:cells_pos[i].n_grid           # parameter
            close_pos = filter(p -> dist(p,pos) < 1.5*d_pos, cells_pos[i].grids[j].grid) # 1.5 is to reach up to 3sigmas
            close_vel =  filter(p -> dist(p,vel) < 1.5*d_vel, cells_vel[i].grids[j].grid)
            if length(close_pos) != 0
                neuron = neuron_passed + j
                d = dist(pos,close_pos[1])
                intensity = gauss(d,d_pos)
                push!(spikes,(neuron,intensity))
            end
            if length(close_vel) != 0
                neuron = neuron_tot + neuron_passed + j
                d = dist(vel,close_vel[1])
                intensity = gauss(d,d_vel)
                push!(spikes,(neuron,intensity))
            end
        end
        neuron_passed += cells_pos[i].n_grid
    end
    return spikes
end

function latency_spikes(obs::Array{Float64}, cells_pos::Array{HexModule}, cells_vel::Array{HexModule}, win_size::Int64)
    pos = [obs[1],obs[3]%pi]     # obs pos and theta
    vel = [obs[2],obs[4]]     # obs pos_v and theta_v
    n_module = length(cells_pos)
    spikes = [Int64[] for _ in 1:win_size]
    neuron_tot = sum([cells_pos[i].n_grid for i in 1:n_module])
    neuron_passed = 0
    for i in 1:n_module
        d_pos = cells_pos[i].grids[1].node_size
        d_vel = cells_vel[i].grids[1].node_size
        for j in 1:cells_pos[i].n_grid           # parameter
            close_pos = filter(p -> dist(p,pos) < 1.25*d_pos, cells_pos[i].grids[j].grid) # 1.5 is to reach up to 3sigmas
            close_vel =  filter(p -> dist(p,vel) < 1.25*d_vel, cells_vel[i].grids[j].grid)
            if length(close_pos) != 0
                neuron = neuron_passed + j
                d = dist(pos,close_pos[1])
                intensity = gauss(d,d_pos)
                ind = win_size - floor(Int64, win_size * intensity)
                push!(spikes[ind],neuron)
            end
            if length(close_vel) != 0
                neuron = neuron_tot + neuron_passed + j
                d = dist(vel,close_vel[1])
                intensity = gauss(d,d_vel)
                ind = win_size - floor(Int64, win_size * intensity)
                push!(spikes[ind],neuron)
            end
        end
        neuron_passed += cells_pos[i].n_grid
    end
    return spikes
end

function input_spikes(obs::Array{Float64}, grid_cells::Array{HexGrid})
    pos = [obs[1],obs[3]%pi]     # obs pos and theta
    vel = [obs[2],obs[4]]     # obs pos_v and theta_v
    bound = [-15.0, 15.0]
    pos_scaled = (pos .- bound[1]) / (bound[2]-bound[1])
    vel_scaled = (vel .- bound[1]) / (bound[2]-bound[1])
    n_grid = length(grid_cells)
    spikes = Tuple{Int64,Float64}[]
    for i in 1:n_grid
        d = grid_cells[i].node_size
        close_pos = filter(p -> dist(p,pos_scaled) < 1.5*d, grid_cells[i].grid) # 1.5 is to reach up to 3sigmas
        close_vel =  filter(p -> dist(p,vel_scaled) < 1.5*d, grid_cells[i].grid)
        if length(close_pos) != 0
            neuron = i
            d_pos = dist(pos_scaled,close_pos[1])
            intensity = gauss(d_pos,d)
            push!(spikes,(neuron,intensity))
        end
        if length(close_vel) != 0
            neuron = n_grid + i
            d_vel = dist(vel_scaled,close_vel[1])
            intensity = gauss(d_vel,d)
            push!(spikes,(neuron,intensity))
        end
    end
    return spikes
end

function latency_spikes(obs::Array{Float64}, grid_cells::Array{HexGrid}, win_size::Int64)
    pos = [obs[1],obs[3]%pi]     # obs pos and theta
    vel = [obs[2],obs[4]]     # obs pos_v and theta_v
    bound = [-15.0, 15.0]
    pos_scaled = (pos .- bound[1]) / (bound[2]-bound[1])
    vel_scaled = (vel .- bound[1]) / (bound[2]-bound[1])
    n_grid = length(grid_cells)
    spikes = [Int64[] for _ in 1:win_size]
    for i in 1:n_grid
        d = grid_cells[i].node_size
        close_pos = filter(p -> dist(p,pos_scaled) < 1.25*d, grid_cells[i].grid) # 1.5 is to reach up to 3sigmas
        close_vel =  filter(p -> dist(p,vel_scaled) < 1.25*d, grid_cells[i].grid)
        if length(close_pos) != 0
            neuron = i
            d_pos = dist(pos_scaled,close_pos[1])
            intensity = gauss(d_pos,d)
            ind = win_size - floor(Int64, win_size * intensity)
            push!(spikes[ind],neuron)
        end
        if length(close_vel) != 0
            neuron = n_grid + i
            d_vel = dist(vel_scaled,close_vel[1])
            intensity = gauss(d_vel,d)
            ind = win_size - floor(Int64, win_size * intensity)
            push!(spikes[ind],neuron)
        end
    end
    return spikes
end

# # %% test input_spikes
#
# cells_tmp_pos = HexModule[]
# cells_tmp_vel = HexModule[]
#
#
# borpos = [-4.1,4.1]
# borvpos = [-50.0,50.0]
# bortheta = [-pi,pi]
# borvtheta = [-50.0,50.0]
#
# for neuron in 1:16 # parameter
#     res = 0.07 # parameter
#     dilat = rand()*(1.4-0.7)+0.7 # parameter
#     theta = rand()*(2pi/6)-pi/6
#     mod_pos = HexModule(res,dilat = dilat, theta = theta, bor_x = borpos, bor_y = bortheta)
#     mod_vel = HexModule(res,dilat = dilat, theta = theta, bor_x = borvpos, bor_y = borvtheta)
#     push!(cells_tmp_pos,mod_pos)
#     push!(cells_tmp_vel,mod_vel)
# end
#
# const cells_pos = cells_tmp_pos
# const cells_vel = cells_tmp_vel
#
# # %%
#
# obs = [0.523324424,0.323242424243,0.4242424225,0.224253525253]
#
#
# @time spikes = input_spikes(obs,cells_pos,cells_vel)
