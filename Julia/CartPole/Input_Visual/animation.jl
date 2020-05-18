using Plots
using ColorTypes
using Images
gr()


function anim_viz(x_spikes::Array{Int64}, y_spikes::Array{Int64},
                  img::Array{RGB{Float64},2}, n_neurons::Int64, step::Int64,
                  window::Int64 = 60)
    if step > window
        p1 = scatter(
            x_spikes[step-window:step],
            y_spikes[step-window:step],
            ylim = [0,n_neurons],
            xlabel = "Step",
            ylabel = "Neuron",
            legend = false
        )
        p2 = plot(img, ticks = nothing)
        plot(p1, p2, layout = 2)
    end
end
