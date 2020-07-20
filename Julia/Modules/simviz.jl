# %% imports
using Statistics
using Plots
gr()

# %% based on a spike history of the simulation layout

function count_spikes(arch::Array{Int64}, spiked::Array{Int64}, layer::Int64)
    count_layer = zeros(arch[layer])
    interval = [sum(arch[1:layer-1])+1, sum(arch[1:layer])]
    spikes_in = filter(x-> interval[1] <= x <= interval[2], spiked)
    spikes_in .-= sum(arch[1:layer-1])
    count_layer[spikes_in] .+= 1
    return count_layer
end

function spike_histogram(arch::Array{Int64,1}, spike_history::Array{Array{Int64}})
    p = Any[]
    for layer in eachindex(arch)
        count_layer = zeros(arch[layer])
        for i in 1:length(spike_history)
            count_layer .+= count_spikes(arch, spike_history[i], layer)
        end
        p_tmp = bar(count_layer,
                    xlabel = "neuron",
                    ylabel = "spiking distribution",
                    legend = false)
        push!(p,p_tmp)

    end
    display(plot(p..., layout = length(arch)))
end

function layer_spike_hist(arch::Array{Int64}, spike_history::Array{Array{Int64}}, layer::Int64)
    interval = [sum(arch[1:layer-1])+1, sum(arch[1:layer])]
    gap = 10 - (length(arch) -1) + 1
    neurons = Array{Int64}[]
    for neuron in interval[1]:interval[2]
        spikebin = [(neuron in spike_history[i]) for i in 1:length(spike_history)]
        neuronspike = [sum(spikebin[i:i+gap-1]) for i in 1:gap:length(spike_history)-gap]
        push!(neurons, neuronspike)
    end
    display(plot(neurons,
                 xlabel = "Time step",
                 ylabel = "Number spikes per neuron",
                 legend = false))
end
