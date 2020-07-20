
include("HexGrid.jl")

function dist(p1::Array{Float64},p2::Array{Float64})
    d = sqrt((p1[1]-p2[1])^2+(p1[2]-p2[2])^2)
    return d
end

                                # for the intensity to index coding (never equals one)
gauss(d::Float64, node_size::Float64) = 0.999999 * exp(-2*(d/node_size)) # considered 2sigma as d in the gaussian


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
# include("CartpoleGridGenerator.jl")
# grids = grid_gen()
#
# # %%
#
# obs = [0.523324424,0.323242424243,0.4242424225,0.224253525253]
#
#
# @time spikes = latency_spikes(obs,grids,7)
