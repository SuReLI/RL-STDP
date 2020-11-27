

mutable struct HexGrid

    node_size::Float64
    grid::Array{Array{Float64}}

    function HexGrid(res::Float64; dilat::Float64 = 1.0, theta::Float64 = 0.0,
                     bor_x::Array{Float64,1} = [0.0,1.0], bor_y::Array{Float64,1} = [0.0,1.0],
                     node_size::Float64 = 1.0, scale::Bool = true)
        grid = ref_grid(res)
        if dilat != 1.0
            dilat_grid!(grid, dilat)
        end
        if theta != 0
            rotate_grid!(grid, theta)
        end
        if scale
            scale_grid!(grid,bor_x[1], bor_x[2], bor_y[1], bor_y[2])
        end
        new(node_size,grid)
    end

end


function ref_grid(hex_size::Float64)
    w = sqrt(3)*hex_size
    h = 2*hex_size
    hi,lo = 2,-1
    center_x = [i*w/2+lo for i in 0:div(hi-lo, w/2)]
    center_y = [j*0.75*h+lo for j in 0:div(hi-lo, 0.75*h)]
    grid = Array{Float64}[]
    for i in eachindex(center_x)
        for j in eachindex(center_y)
            if (i+j)%2==0; push!(grid, [center_x[i], center_y[j]]); end
        end
    end
    return grid
end

function rotate_grid!(hex_grid::Array{Array{Float64}}, theta::Float64)
    rot = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    center = [0.5,0.5]
    for i in eachindex(hex_grid)
        hex_grid[i] .= rot*(hex_grid[i] .- center) .+ center
    end
    nothing
end

function dilat_grid!(hex_grid::Array{Array{Float64}},ratio::Float64)
    for i in eachindex(hex_grid)
        hex_grid[i] .*= ratio
    end
    nothing
end

function dilat_grid!(hex_grid::HexGrid,ratio::Float64)
    for i in eachindex(hex_grid.grid)
        hex_grid.grid[i] .*= ratio
    end
    hex_grid.node_size *= ratio
    nothing
end

function trunc_grid!(hex_grid::Array{Array{Float64}}, lo_x::Float64 = 0.0,
                     hi_x::Float64 = 1.0, lo_y::Float64 = 0.0,
                     hi_y::Float64 = 1.0)
    filter!(x->(lo_x<=x[1]<=hi_x) && (lo_y<=x[2]<=hi_y),hex_grid)
    nothing
end


function scale_grid!(hex_grid::Array{Array{Float64}}, lo_x::Float64, hi_x::Float64,
                     lo_y::Float64, hi_y::Float64)
    l1 = hi_x - lo_x
    l2 = hi_y - lo_y
    if l1 >= l2
        len = l1
        trunc_grid!(hex_grid,0.0,1.0,0.0,l2/l1)
    end
    if l2 > l1
        len = l2
        trunc_grid!(hex_grid,0.0,l1/l2)
    end
    lo = [lo_x,lo_y]
    for i in eachindex(hex_grid)
        hex_grid[i] .= hex_grid[i] .*len .+lo
    end
    nothing
end

function scale_grid!(hex_grid::HexGrid, lo_x::Float64, hi_x::Float64,
                     lo_y::Float64, hi_y::Float64)
    l1 = hi_x - lo_x
    l2 = hi_y - lo_y
    if l1 >= l2
        len = l1
        trunc_grid!(hex_grid.grid,0.0,1.0,0.0,l2/l1)
    end
    if l2 > l1
        len = l2
        trunc_grid!(hex_grid.grid,0.0,l1/l2)
    end
    lo = [lo_x,lo_y]
    for i in eachindex(hex_grid.grid)
        hex_grid.grid[i] .= hex_grid.grid[i] .*len .+lo
    end
    hex_grid.node_size *= len
    nothing
end



function dist(p1::Array{Float64},p2::Array{Float64})
    d = sqrt((p1[1]-p2[1])^2+(p1[2]-p2[2])^2)
    return d
end

# %% hexmodule struct
mutable struct HexModule

    n_grid::Int64
    grids::Array{HexGrid}

    function HexModule(res::Float64; dilat::Float64 = 1.0, theta::Float64 = 0.0,
                       bor_x::Array{Float64,1} = [0.0,1.0], bor_y::Array{Float64,1} = [0.0,1.0],
                       mod_size::Int64 = 2) # parameter
        ref = HexGrid(res, scale = false)
        grids = HexGrid[]
        for i in 1:mod_size^2
            new_grid = deepcopy(ref)
            trans_grid!(new_grid, i, mod_size)
            dilat_grid!(new_grid, dilat)
            rotate_grid!(new_grid.grid, theta)
            scale_grid!(new_grid,bor_x[1], bor_x[2], bor_y[1], bor_y[2])
            push!(grids, new_grid)
        end
        new(mod_size^2,grids)
    end
end

function trans_grid!(hex_grid::HexGrid, placement::Int64, mod_size::Int64)
    D = dist(hex_grid.grid[1],hex_grid.grid[2])
    d = D/sqrt(3)
    d_trans = d/(mod_size)
    theta_i = pi/3
    theta_j = -pi/3
    i = div(placement-1,mod_size)
    j = (placement-1)%mod_size
    translation_i = d_trans*[cos(theta_i), sin(theta_i)]
    translation_j = d_trans*[cos(theta_j), sin(theta_j)]
    for _ in 1:i
        for index in eachindex(hex_grid.grid)
            hex_grid.grid[index] .+= translation_i
        end
    end
    for _ in 1:j
        for index in eachindex(hex_grid.grid)
            hex_grid.grid[index] .+= translation_j
        end
    end
    hex_grid.node_size = d_trans/2
    nothing
end

# %% Visualization
using Plots
gr()

function scatter_grid(hex_grid::HexGrid)
    x = []
    y = []
    for point in hex_grid.grid
        push!(x,point[1])
        push!(y,point[2])
    end
    display(scatter(x,y,color = :red))
    nothing
end

function scatter_grid!(hex_grid::HexGrid)
    x = []
    y = []
    for point in hex_grid.grid
        push!(x,point[1])
        push!(y,point[2])
    end
    display(scatter!(x,y,legend = false, xlabel = "Variable 1", ylabel = "Variable 2"))
    nothing
end

function scatter_module(hex_mod::HexModule)
    for i in 1:hex_mod.n_grid
        scatter_grid!(hexmod.grids[i])
    end
end




# %% hexmodule test

borx = [0.0,1.0]
bory = [0.0,1.0]
hexmod = HexModule(0.03, dilat = 1.0, theta = pi/10, bor_x = borx,
                   bor_y = bory, mod_size = 2)

plot(size = (600,600))
scatter_grid!(hexmod.grids[1])
#
#
# # %%
#
# @show hexmod.grids[1].grid[1]
# @show hexmod.grids[1].grid[2]
# @show hexmod.grids[1].grid[3]
# @show hexmod.grids[1].grid[4]
