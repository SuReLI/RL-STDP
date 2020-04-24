module NeuronModel

export izhikevicmodel_fire
export izhikevicmodel_step


const N = 1000
const Ne = 800
const a = [(i<=Ne) ? 0.02 : 0.1 for i in 1:N]
const d = [(i<=Ne) ? 8.0 : 2.0 for i in 1:N]

function izhikevicmodel_step(v::Array{Float64},u::Array{Float64},I::Array{Float64}) # Module neuron_model
    v .= @. v+0.5*((0.04*v+5)*v+140-u+I)
    v .= @. v+0.5*((0.04*v+5)*v+140-u+I)
    u .= @. u+a*(0.2*v-u)
    return v,u
end

function izhikevicmodel_fire(v::Array{Float64},u::Array{Float64},fired::Array{Int64}) # Module neuron_model
    if length(fired) != 0
        v[fired] .= -65.0
        u[fired] .= u[fired] .+ d[fired]
    end
    return v,u
end

end #end of module
