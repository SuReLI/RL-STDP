# %% imports

import Random
include("../../Modules/Gridcells/HexGrid.jl")


# %%

function grid_gen()
    Random.seed!(0)

    dilats = [2*sqrt(2), 2, sqrt(2), 1]

    grids = HexGrid[]

    for mod in 1:4
        n_neuron = 2^(mod+1)
        for neuron in 1:n_neuron
            res = 0.03*dilats[mod]
            node_size = 0.25*sqrt(3)*res*dilats[mod]
            theta = rand()*(2pi/6)-pi/6
            grid = HexGrid(res, theta = theta, node_size = node_size)
            push!(grids, grid)
        end
    end
    return grids
end
