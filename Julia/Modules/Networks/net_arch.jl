using LinearAlgebra
using YAML

parameters = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/Modules/cfg.yml"))

mutable struct Weights

    e::Array{Array{Float64,2}}
    ie::Array{Any}
    ei::Array{Any}
    de::Array{Array{Float64,2}}

    function Weights(arch::Array{Int64}, param::Dict = parameters)
        # exc
        e = Array{Float64,2}[]
        de = Array{Float64,2}[]
        for l in 1:length(arch)-1
            push!(e, clamp.(randn(arch[l],arch[l+1]) .* param["std"] .+ param["m"], 0, param["wmax"]))
            push!(de, zeros(arch[l],arch[l+1]))
        end
        # inh
        ie = Array{Any}[]
        ei = Array{Any}[]
        for l in 1:length(arch)
            if l>1 && l!=length(arch)
                push!(ie, param["wie"] .* (ones(arch[l],arch[l])-I(arch[l])))
                push!(ei, param["wei"] .* I(arch[l]))
            else
                push!(ie, [[] for _ in 1:arch[l]])
                push!(ei, [[] for _ in 1:arch[l]])
            end
        end
        new(e,ie,ei,de)
    end
end

mutable struct Ind

    exc::Array{Int64}
    inh::Array{Int64}
    el::Array{Array{Int64}}
    il::Array{Array{Int64}}

    function Ind(arch)
        n_exc = sum(arch)
        n_inh = sum(arch[2:end-1])
        n_neurons = n_exc + n_inh
        exc = collect(1:n_exc)
        inh = collect(n_exc+1:n_neurons)

        architr = [0; arch]
        el = Array{Int64}[]
        il = Array{Int64}[]

        for l in 2:length(architr)
            push!(el, collect(sum(architr[1:l-1])+1:sum(architr[1:l])))
            if l > 2 && l!=length(architr)
                push!(il, collect(sum(architr[2:l-1])+1:sum(architr[2:l])) .+ n_exc .- arch[1])
            else
                push!(il, [])
            end
        end
        new(exc,inh,el,il)
    end
end



mutable struct Network

    n_exc::Int64
    n_inh::Int64
    n_neurons::Int64

    i::Ind

    v::Array{Float64}
    u::Array{Float64}
    d::Array{Float64}
    a::Array{Float64}

    s::Weights
    trace_e::Array{Float64}
    ho::Array{Float64}

    I::Array{Float64}
    da::Float64
    arch::Array{Int64}
    param::Dict

    function Network(arch::Array{Int64}, param::Dict = parameters)
        n_exc = sum(arch)
        n_inh = sum(arch[2:end-1])
        n_neurons = n_exc + n_inh

        i = Ind(arch)

        v = param["c"]*ones(n_neurons)
        u = 0.2 * v
        d = [(i<=n_exc) ? param["de"] : param["di"] for i in 1:n_neurons]
        a = [(i<=n_exc) ? param["ae"] : param["ai"] for i in 1:n_neurons]

        s = Weights(arch,param)
        trace_e = zeros(n_exc)
        ho = zeros(n_neurons)

        I = zeros(n_neurons)
        da = 0.0
        arch = arch

        new(n_exc,n_inh,n_neurons,i,v,u,d,a,s,trace_e,ho,I,da,arch,param)
    end
end


# %% functions

# direct spike of the neuron
function input!(net::Network, input_spikes::Array{Int64})
    net.v[input_spikes] .= 40.0
end

# superthreshold current processed by the neuron
function input!(net::Network, input_spikes::Array{Tuple{Int64,Float64}})
    #net.I = 13 .* (rand(net.n_neurons) .- 0.5)
    net.I = zeros(net.n_neurons)
    max_current = net.param["imax"]
    min_current = net.param["imin"]
    indices = [input_spikes[i][1] for i in eachindex(input_spikes)]
    intensity = [input_spikes[i][2] for i in eachindex(input_spikes)]
    net.I[indices] .+= (max_current-min_current) .* intensity .+ min_current
    net.v[indices] .= @. net.v[indices]+0.5*((0.04*net.v[indices]+5)*net.v[indices]+140-net.u[indices]+net.I[indices])
    net.v[indices] .= @. net.v[indices]+0.5*((0.04*net.v[indices]+5)*net.v[indices]+140-net.u[indices]+net.I[indices])
end

# superthreshold current processed by the neuron
function input_LIF!(net::Network, input_spikes::Array{Tuple{Int64,Float64}})
    #net.I = 13 .* (rand(net.n_neurons) .- 0.5)
    net.I = zeros(net.n_neurons)
    max_current = net.param["imax"]
    min_current = net.param["imin"]
    indices = [input_spikes[i][1] for i in eachindex(input_spikes)]
    intensity = [input_spikes[i][2] for i in eachindex(input_spikes)]
    net.I[indices] .+= (max_current-min_current) .* intensity .+ min_current
    net.v[indices] .= @. net.v[indices] + (1/net.param["taulif"])*(-net.v[indices]+net.I[indices]+net.param["c"])
end

function spike!(net::Network)
    # spikes
    net.I = zeros(net.n_neurons)
    spiked = findall(x->x>net.param["vthresh"],net.v .- net.ho)
    spiked_e = filter(x->(x in net.i.exc), spiked)
    spiked_ho = filter(x-> x > net.arch[1], spiked_e)
    net.ho[spiked_ho] .+= net.param["theta"]
    net.v[spiked] .= net.param["c"]
    net.u[spiked] .+= net.d[spiked]
    # increment current
    for neuron in spiked
        for l in 1:length(net.arch)-1
            if neuron in net.i.el[l]
                neur = neuron - (net.i.el[l][1] - 1)
                net.I[net.i.el[l+1]] .+= net.s.e[l][neur,:]
                net.I[net.i.il[l]] .+= net.s.ei[l][neur,:]
            end
            if neuron in net.i.il[l]
                neur = neuron - (net.i.il[l][1] - 1)
                net.I[net.i.el[l]] .+= net.s.ie[l][neur,:]
            end
        end
    end

    #time step (Izhikevic model)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.u .= @. net.u+net.a*(0.2*net.v-net.u)
    net.trace_e[spiked_e] .= 0.1

    # LTP & LTD (STDP)
    for neuron in spiked_e
        for l in 1:length(net.arch)
            if neuron in net.i.el[l]
                neur = neuron - (net.i.el[l][1] - 1)
                if l > 1
                    net.s.de[l-1][:,neur] .+= net.param["apost"] .* net.trace_e[net.i.el[l-1]]
                end
                if l < length(net.arch)
                    net.s.de[l][neur,:] .-= net.param["apre"] .* net.trace_e[net.i.el[l+1]]
                end
            end
        end
    end


    # time decay
    net.ho .*= net.param["ttheta"]
    net.trace_e .*= 0.95
    net.da *= net.param["tda"]
    spiked
end

function spike_LIF!(net::Network)
    # spikes
    net.I = zeros(net.n_neurons)
    spiked = findall(x->x>net.param["vthresh"],net.v .- net.ho)
    spiked_e = filter(x->(x in net.i.exc), spiked)
    spiked_ho = filter(x-> x > net.arch[1], spiked_e)
    net.ho[spiked_e] .+= net.param["theta"] #homeostasis on every layer
    #net.ho[spiked_ho] .+= net.param["theta"] #homeostasis on all but first layer
    net.v[spiked] .= net.param["c"]
    # increment current
    for neuron in spiked
        for l in 1:length(net.arch)-1
            if neuron in net.i.el[l]
                neur = neuron - (net.i.el[l][1] - 1)
                net.I[net.i.el[l+1]] .+= net.s.e[l][neur,:]
                net.I[net.i.il[l]] .+= net.s.ei[l][neur,:]
            end
            if neuron in net.i.il[l]
                neur = neuron - (net.i.il[l][1] - 1)
                net.I[net.i.el[l]] .+= net.s.ie[l][neur,:]
            end
        end
    end

    #time step (LIF model)
    net.v .= @. net.v + (1/net.param["taulif"])*(-net.v+net.I+net.param["c"])
    net.trace_e[spiked_e] .= 0.1

    # LTP & LTD (STDP)
    for neuron in spiked_e
        for l in 1:length(net.arch)
            if neuron in net.i.el[l]
                neur = neuron - (net.i.el[l][1] - 1)
                if l > 1
                    net.s.de[l-1][:,neur] .+= net.param["apost"] .* net.trace_e[net.i.el[l-1]]
                end
                if l < length(net.arch)
                    net.s.de[l][neur,:] .-= net.param["apre"] .* net.trace_e[net.i.el[l+1]]
                end
            end
        end
    end


    # time decay
    net.ho .*= net.param["ttheta"]
    net.trace_e .*= 0.95
    net.da *= net.param["tda"]
    spiked
end

function learn!(net::Network)
    #increase the synaptic weigths
    for l in 1:length(net.arch)-1
        net.s.e[l] .+= (0.002+net.da) .* net.s.de[l]
        net.s.e[l] .= clamp.(net.s.e[l], 0, net.param["wmax"])
        net.s.de[l] .*= 0.99
    end
end

# %% main functions

function step!(net::Network, train::Bool=false)
    spiked = spike!(net)
    if train
        learn!(net)
    end
    spiked
end

function step!(net::Network, input_spikes::Array{Int64}, train::Bool=false)
    input!(net,input_spikes)
    spiked = spike!(net)
    if train
        learn!(net)
    end
    spiked
end

function step_LIF!(net::Network, input_spikes::Array{Tuple{Int64,Float64}}, train::Bool=false)
    input_LIF!(net,input_spikes)
    spiked = spike_LIF!(net)
    if train
        learn!(net)
    end
    spiked
end

function step!(net::Network, input_spikes::Array{Tuple{Int64,Float64}}, train::Bool=false)
    input!(net,input_spikes)
    spiked = spike!(net)
    if train
        learn!(net)
    end
    spiked
end

# %% visualization function

function weight_viz(net::Network, method::String = "histogram")
    weights = net.s.e
    if method == "histogram"
        h = []
        for i in 1:length(net.arch)-1
            vec = filter(x-> x>=0 ,weights[i])
            h_tmp = histogram(vec,
                          legend = false,
                          xlabel = "Synaptic strength layer $(i)",
                          ylabel = "Synapse number")
            push!(h, h_tmp)
        end
        display(plot(h..., layout = length(net.arch)-1))

    elseif method == "heatmap"
        h = []
        for i in 1:length(net.arch)-1
            h_tmp = heatmap(weights[i],
                            xlabel = "connection matrix layer $(i)")
            push!(h, h_tmp)
        end
        display(plot(h..., layout = length(net.arch)-1))

    end
end
