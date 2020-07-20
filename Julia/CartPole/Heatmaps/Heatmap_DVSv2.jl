# imports

using JLD


# %% Create the network

include("../../Modules/Networks/net_res.jl")

n = Network([1200,64,64,2])

# %% load display_spikes

display_spikes = load("Julia/CartPole/Heatmaps/disp_spikesDVSv2.jld", "display_spikes")

# %% main sim

spike_hist = Array{Int64}[]


@inbounds for sec in 0:9
    @time @inbounds for msec in 1:1000
        time = 1000*sec+msec
        train = false
        if msec%10 == 0
            train = true
        end
        if time < 4000 && msec%20 == 0
            step_ = div(time,20) + 1
            input_spikes = display_spikes[step_]
            spiked = step!(n, input_spikes, train)
        else
            spiked = step!(n,train)
        end
        push!(spike_hist,spiked)
    end
end

# %% create a heatmap

using Plots
gr()
function heatmap_spikes(spikes::Array{Array{Int64}}, n_interval::Array{Int64},layer_bound::Array{Int64};
                        window::Int64 = 50, save::Bool = false)
    mat_spikes = zeros(n_interval[2],length(spikes))
    for time in 1:length(spikes)
        col = zeros(n_interval[2])
        if time > window
            for i in time-window:time
                indices = filter(x->n_interval[1]<= x <= n_interval[2] ,spikes[i])
                col[indices] .+= 1
            end
            col ./= window
            col ./= maximum(col)
        else
            for i in 1:time
                indices = filter(x->n_interval[1]<= x <= n_interval[2] ,spikes[i])
                col[indices] .+= 1
            end
            col ./= time
            col ./= maximum(col)
        end
        mat_spikes[:,time] .= col
    end
    display_mat = mat_spikes[n_interval[1]:end,:]
    heatmap(display_mat)
    plot!(title = "Network activity neuron $(n_interval[1]) to $(n_interval[2])",
          xlabel = "Time (msec)",
          ylabel = "Neuron")
    for i in layer_bound
        display(plot!([i], seriestype = "hline", color = :white, legend = false))
    end
    if save == true
        savefig("Julia/CartPole/Heatmaps/heatmapDVSv2.png")
    end
end

# %%

heatmap_spikes(spike_hist, [1200,1330], [64,128], save = true)
