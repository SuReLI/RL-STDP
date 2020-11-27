# %% imports

using Clustering
using OrderedCollections

include("create_mat.jl")
include("hclustrecipe.jl")

# %% Clustering functions


function cos_dist(vec1::Array{Float64}, vec2::Array{Float64})
    if length(vec1) != length(vec2)
        error("dimension mismatch in the vectors (cosine_dist())")
    end
    dist = sum([vec1[i] * vec2[i] for i in eachindex(vec1)])
    norm1 = sqrt(sum([vec1[i]^2 for i in eachindex(vec1)]))
    norm2 = sqrt(sum([vec2[i]^2 for i in eachindex(vec2)]))
    dist /= norm1*norm2
    return 1 - dist
end

function eucl_dist(vec1::Array{Float64}, vec2::Array{Float64})
    if length(vec1) != length(vec2)
        error("dimension mismatch in the vectors (cosine_dist())")
    end
    dist = sqrt(sum(([(vec1[i] - vec2[i])^2 for i in eachindex(vec1)])))
    return dist
end


# create the distance matrix d
function compute_d(mc::MatrixClust, dist::String = "cosine")
    # compute the d matrix
    n = length(mc.segs) ; d = Array{Float64,2}(undef, (n,n)) ; segs = mc.segs
    for i in 1:n
        for j in 1:n
            if dist == "cosine"
                d[i,j] = cos_dist(segs[i], segs[j])
            elseif dist == "euclidean"
                d[i,j] = eucl_dist(segs[i], segs[j])
            else
                error("no distance named")
            end
        end
    end
    return d
end

# Find the n-th highest jump in height in the dendrogram (n is the order)
function h_cut(h::Hclust, order::Int64 = 1)
    diffs = Float64[] ; heights = h.heights ; l = length(heights) ; idx = 0

    for i in 2:l
        diff = heights[i] - heights[i-1]
        push!(diffs, diff)
    end

    for i in 1:order
        idx = argmax(diffs)
        diffs[idx] = - Inf
    end

    return (heights[idx] + heights[idx-1])/2
end

# Clusterize the matrice lines (number of clusters define by the order of the cut)
function clusterize_mat(mc::MatrixClust;  dist::String = "euclidean", order::Int64 = 1, metric::Symbol = :single)
    d = compute_d(mc, dist)
    h_clust = hclust(d, linkage = metric)
    cut = h_cut(h_clust, order)
    z = cutree(h_clust, h=cut)
    return z
end

# count number of one (function probably already implemented in julia)
function one_count(z::Array{Int64})
    return sum([z[i] == 1 for i in 1:length(z)])
end

function plot_dend(mc::MatrixClust;  dist::String = "euclidean", metric::Symbol = :single)
    d = compute_d(mc, dist)
    h_clust = hclust(d, linkage = metric)
    display(plot(hclustplot(h_clust, true),
                seriestype = :path,
                ylabel = "Height",
                grid=false,
                legend = false,
                xlabel = "M_alpha lines"))
end



# %% Main functions (run the best parameters research and then use those parameters to label the matrices)

#param research
function find_top_param(mc::MatrixClust)
        metrics = [:single, :average, :complete, :ward]
        orders = [1, 2, 3, 4 ,5, 6, 7, 8, 9, 10]
        results = Dict(0 => ["order", "metric", "n_clust", "size_bigger_clust"]) ; i = 1
        for order in orders
                for metric in metrics
                        z = clusterize_mat(mc, order = order, metric = metric)
                        n_clust = maximum(z)
                        size_bigger_clust = one_count(z)
                        results_tmp = Dict(i => [order, metric, n_clust, size_bigger_clust])
                        results = merge(results, results_tmp)
                        i += 1
                end
        end
        results = sort(results)
        return results
end

# Label the matrices to be compared after
function label_mat(mc::MatrixClust, params::Array{Any,1})
    mat_label = Dict(0 => ["clusters"]) ; l_seg = length(mc.segs) ; l_mat = length(mc.matrices)
    z = clusterize_mat(mc, order = params[1], metric = params[2])
    for i in 1:l_mat
        mat_label_tmp = Dict(i => Any[])
        mat_label = merge(mat_label, mat_label_tmp)
    end
    for i in 1:l_seg
        label = z[i]
        idx_mat = mc.s_to_m[i]
        push!(mat_label[idx_mat] , label)
    end
    mat_label = sort(mat_label)
    return mat_label
end

# Sort clusters for each matrix
function unique_label(mat_label::OrderedCollections.OrderedDict{Int64,Array{Any,1}})
    uni_label = deepcopy(mat_label)
    for i in 1:length(mat_label)-1
        uni_label[i] = sort(unique(mat_label[i]))
    end
    return uni_label
end
