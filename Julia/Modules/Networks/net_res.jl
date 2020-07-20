using LinearAlgebra
using YAML

parameters = YAML.load(open("/Users/titou/Documents/PFE/RL-STDP/Julia/Modules/cfg.yml"))

mutable struct Network

    n_exc::Int64
    n_neurons::Int64

    v::Array{Float64}
    u::Array{Float64}
    d::Array{Float64}
    a::Array{Float64}

    connection::Array{Int64,2}
    connection_ee::Array{Int64,2}
    connection_ei::Array{Int64,2}
    connection_ie::Array{Int64,2}
    s_ee::Array{Float64,2}
    s_ei::Array{Float64,2}
    s_ie::Array{Float64,2}
    sd_ee::Array{Float64,2}
    sd_ei::Array{Float64,2}
    trace_e::Array{Float64}
    trace_i::Array{Float64}
    ho::Array{Float64}

    exc::Array{Int64}
    inh::Array{Int64}

    I::Array{Float64}
    da::Float64
    arch::Array{Int64}
    param::Dict

    function connect(n_neurons::Int64, n_exc::Int64, connectivity::Float64,
                     arch::Array{Int64}, inh_per::Float64 ) # parameter
        if arch == [0]
            connection = (rand(n_neurons,n_neurons) .< connectivity)
            n_inh = n_neurons - n_exc
            connection[n_exc+1:n_neurons,n_exc+1:n_neurons] .= zeros(n_inh, n_inh)
        else
            architr = [0; arch]
            connection = zeros(n_neurons,n_neurons)
            for i in 1:length(architr)-2
                si1 = sum(architr[1:i])
                si2 = sum(architr[1:i+1])
                si3 = sum(architr[1:i+2])
                ai2 = architr[i+1]
                ai3 = architr[i+2]
                connection[1+si1:si2,1+si2:si3] .= ones(ai2,ai3)
            end
            archinh = [0;arch[2:end-1]]
            for i in 1:length(archinh)-1
                sii1 = sum(archinh[1:i])
                sii2 = sum(archinh[1:i+1])
                si2 = sum(architr[1:i+1])
                si3 = sum(architr[1:i+2])
                a2 = arch[i+1]
                n_con_inh = floor(Int64,a2*inh_per)
                #mat_ie = ones(a2,a2) .- I(a2)
                mat_ie = zeros(a2,a2)
                for lin in 1:a2
                    ind = rand(1:a2,n_con_inh)
                    ind_f = filter(x -> x != lin, ind)
                    mat_ie[lin,ind_f] .+= 1
                end
                connection[n_exc+sii1+1:n_exc+sii2, si2+1:si3] .= mat_ie
                connection[si2+1:si3, n_exc+sii1+1:n_exc+sii2] .= I(a2)
            end
        end
        exc = collect(1:n_exc)
        inh = collect(n_exc+1:n_neurons)
        return connection, exc, inh
    end

    function Network(arch::Array{Int64}, param::Dict = parameters)
        n_exc = sum(arch)
        n_neurons = n_exc+sum(arch[2:end-1])
        v = param["c"]*ones(n_neurons)
        u = 0.2 * v
        d = [(i<=n_exc) ? param["de"] : param["di"] for i in 1:n_neurons]
        a = [(i<=n_exc) ? param["ae"] : param["ai"] for i in 1:n_neurons]
        n_inh = n_neurons-n_exc
        connection, exc, inh = connect(n_neurons, n_exc, 0.1, arch, param["i_per"])
        connection_ee = connection[exc,exc]
        connection_ei = connection[exc,inh]
        connection_ie = connection[inh,exc]
        # maybe set the Exc-> Inh conn weights to high value
        s_ee = param["wee"] .* ones(n_exc, n_exc)
        s_ei = param["wei"] .* ones(n_exc, n_inh)
        s_ie = param["wie"] .* ones(n_inh, n_exc)
        s_ee .*= connection_ee
        s_ei .*= connection_ei
        s_ie .*= connection_ie
        sd_ee = zeros(n_exc,n_exc)
        sd_ei = zeros(n_exc,n_inh)
        trace_e = zeros(n_exc)
        trace_i = zeros(n_inh)
        ho = zeros(n_neurons)
        #I = 13 .* (rand(n_neurons) .- 0.5)
        I = zeros(n_neurons)
        da = 0.0
        arch = arch
        new(n_exc,n_neurons,v,u,d,a,connection,connection_ee,connection_ei,connection_ie,
            s_ee,s_ei,s_ie,sd_ee,sd_ei,trace_e,trace_i,ho,exc,inh,I,da,arch,param)
    end

    function Network(n_neurons::Int64, connectivity::Float64, n_exc::Int64, param::Dict = parameters)
        v = param["c"]*ones(n_neurons)
        u = 0.2 * v
        d = [(i<=n_exc) ? param["de"] : param["di"] for i in 1:n_neurons]
        a = [(i<=n_exc) ? param["ae"] : param["ai"] for i in 1:n_neurons]
        n_inh = n_neurons-n_exc
        connection, exc, inh = connect(n_neurons, n_exc, connectivity, [0])
        connection_ee = connection[exc,exc]
        connection_ei = connection[exc,inh]
        connection_ie = connection[inh,exc]
        # maybe set the Exc-> Inh conn weights to high value
        s_ee = param["wee"] .* ones(n_exc, n_exc)
        s_ei = param["wei"] .* ones(n_exc, n_inh)
        s_ie = param["wie"] .* ones(n_inh, n_exc)
        s_ee .*= connection_ee
        s_ei .*= connection_ei
        s_ie .*= connection_ie
        sd_ee = zeros(n_exc,n_exc)
        sd_ei = zeros(n_exc,n_inh)
        trace_e = zeros(n_exc)
        trace_i = zeros(n_inh)
        ho = zeros(n_neurons)
        #I = 13 .* (rand(n_neurons) .- 0.5)
        I = zeros(n_neurons)
        da = 0.0
        arch = [0]
        new(n_exc,n_neurons,v,u,d,a,connection,connection_ee,connection_ei,connection_ie,
            s_ee,s_ei,s_ie,sd_ee,sd_ei,trace_e,trace_i,ho,exc,inh,I,da,arch, param)
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


function spike!(net::Network)
    # spikes
    #net.I = 13 .* (rand(net.n_neurons) .- 0.5)
    net.I = zeros(net.n_neurons)
    spiked = findall(x->x>30,net.v .- net.ho)
    spiked_e = filter(x->(x in net.exc), spiked)
    spiked_i = filter(x->(x in net.inh), spiked)
    spiked_ho = filter(x-> x > net.arch[1], spiked_e)
    net.ho[spiked_ho] .+= net.param["theta"]
    net.v[spiked] .= net.param["c"]
    net.u[spiked] .+= net.d[spiked]
    # increment current
    for neuron in spiked
        if neuron in net.exc
            net.I[net.exc] .+= net.s_ee[neuron,:]
            net.I[net.inh] .+= net.s_ei[neuron,:]
        elseif neuron in net.inh
            net.I[net.exc] .+= net.s_ie[neuron-net.n_exc,:]
        end
    end
    #time step (Izhikevic model)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.u .= @. net.u+net.a*(0.2*net.v-net.u)
    net.trace_e[spiked_e] .= 0.1
    net.trace_i[spiked_i .- net.n_exc] .= 0.1
    # LTP & LTD
    for neuron in spiked_e
        net.sd_ee[:,neuron] .+= net.param["apost"] .* net.trace_e #parameter (a_post)
        net.sd_ee[neuron,:] .-= net.param["apre"] .* net.trace_e # parameter (a_pre)
        if net.arch == [0]
            net.sd_ei[neuron,:] .-= net.param["apre"] .* net.trace_i
        end
    end
    if net.arch == [0]
        for neuron in spiked_i
            net.sd_ei[:,neuron - net.n_exc] .+= net.param["apost"] .* net.trace_e
        end
    end

    # time decay
    net.ho .*= net.param["ttheta"]
    net.trace_e .*= 0.95
    net.trace_i .*= 0.95
    net.da *= net.param["tda"]
    spiked
end

function learn!(net::Network)
    #increase the synaptic weigths
    net.s_ee .+= (0.002+net.da) .* net.sd_ee
    net.s_ee .= clamp.(net.s_ee, 0, net.param["wmax"])
    net.s_ee .*= net.connection_ee
    net.sd_ee .*= 0.99
    if net.arch == [0]
        net.s_ei .+= (0.002+net.da) .* net.sd_ei
        net.s_ei .= clamp.(net.s_ei, 0, net.param["wmax"])
        net.s_ei .*= net.connection_ei
        net.sd_ei .*= 0.99
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
    weights = net.s_ee
    if method == "histogram"
        filtered = filter(x-> x!=0.0, weights)
        display(histogram(filtered,
                        legend = false,
                        xlabel = "Synaptic strength",
                        ylabel = "Synapse number"))
    elseif method == "heatmap"
        heatmap(weights,
                title = "connection matrix")
    end
end
