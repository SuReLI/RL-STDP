# %% imports

include("create_mat.jl")
include("clusterize_mat.jl")
include("hclustrecipe.jl")

# %% code to classify and visualize the clusters

# Load the data
dic = load("Julia/MatrixCluster/mc.jld")
mc = dic["mc"]

# run a param research select most appropriate one manually
# (best compromise between lowest order, higher n_clust, smaller size_bigger_clust)
param_search = find_top_param(mc)

# %% Label the matrices section

params = param_search[26]       ## "cosine" :: 19, 23, 36, 39 ## "euclidean" ::15, 26, 35
labels = label_mat(mc, params)
uni = unique_label(labels)

# %% plot the dendrogram
params = param_search[26]
plot_dend(mc, metric = params[2])

# %% plot the matrice distribution

#from uni extract a vector 1D
di = []
sizes = []
for idx in 1:length(uni)-1
    push!(di, uni[idx]...)
    push!(sizes, length(uni[idx]))
end

histogram(di, bins = 1:1:23,
            xlabel = "Clusters",
            ylabel = "Number of M_alpha using each cluster",
            legend = false,
            )

m = mean(sizes)
