# %% imports

include("create_mat.jl")
include("clusterize_mat.jl")

# %% code to classify and visualize the clusters

# Load the data
dic = load("Julia/MatrixCluster/mc.jld")
mc = dic["mc"]

# run a param research select most appropriate one manually
# (best compromise between lowest order, higher n_clust, smaller size_bigger_clust)
param_search = find_top_param(mc)
@show param_search

# %% Label the matrices section

params = param_search[26]       ## "cosine" :: 19, 23, 36, 39 ## "euclidean" ::15, 26, 35
labels = label_mat(mc, params)
uni = unique_label(labels)
 
