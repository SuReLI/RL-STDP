using LinearAlgebra


mutable struct Network

    n_exc::Int64
    n_neurons::Int64

    v::Array{Float64}
    u::Array{Float64}
    d::Array{Float64}
    a::Array{Float64}

    connection::Array{Int64,2} # WARNING read by columns !!!!
    connection_exc::Array{Int64,2} # WARNING read by columns !!!!
    connection_inh::Array{Int64,2} # WARNING read by columns !!!!
    s_exc::Array{Float64,2} # WARNING read by columns !!!!
    s_inh::Array{Float64,2} # WARNING read by columns !!!!
    sd::Array{Float64,2} # WARNING read by columns !!!!
    STDP::Array{Float64}
    ho::Array{Float64}

    exc::Array{Int64}
    inh::Array{Int64}

    I::Array{Float64}
    da::Float64
    arch::Array{Int64}
    #param::Dict

    function connect(n_neurons::Int64, n_exc::Int64, connectivity::Float64,
                     arch::Array{Int64}; inh_per::Float64 = 0.05 ) # parameter
        if arch == [0]
            connection = (rand(n_neurons,n_neurons) .< connectivity)
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
                    ind_f = filter(x -> x != i, ind)
                    mat_ie[i,ind_f] .+= 1
                end
                connection[n_exc+sii1+1:n_exc+sii2, si2+1:si3] .= mat_ie
                connection[si2+1:si3, n_exc+sii1+1:n_exc+sii2] .= I(a2)
            end
        end
        exc = collect(1:n_exc)
        inh = collect(n_exc+1:n_neurons)
        return connection, exc, inh
    end

    function Network(arch::Array{Int64})
        n_exc = sum(arch)
        n_neurons = n_exc+sum(arch[2:end-1])
        v = -65.0*ones(n_neurons) #parameter
        u = 0.2 * v # parameter
        d = [(i<=n_exc) ? 8.0 : 2.0 for i in 1:n_neurons] # parameter
        a = [(i<=n_exc) ? 0.02 : 0.1 for i in 1:n_neurons] # parameter
        n_inh = n_neurons-n_exc
        connection, exc, inh = connect(n_neurons, n_exc, 0.1, arch)
        connection = transpose(connection)
        connection_exc = connection[:,exc]
        connection_inh = connection[:,inh]
        # maybe set the Exc-> Inh conn weights to high value
        s_exc = 1.0 .* ones(n_neurons, n_exc) # parameter
        s_inh = -0.5 .* ones(n_neurons, n_inh) # parameter
        s_exc .*= connection_exc
        s_inh .*= connection_inh
        sd = zeros(n_neurons,n_exc)
        STDP = zeros(n_neurons)
        ho = zeros(n_neurons)
        I = 13 .* (rand(n_neurons) .- 0.5) # parameters (noise in input current) random thalamic input
        da = 0.0
        arch = arch
        new(n_exc,n_neurons,v,u,d,a,connection,connection_exc,connection_inh,
            s_exc,s_inh,sd,STDP,ho,exc,inh,I,da,arch)
    end

    function Network(n_neurons::Int64, connectivity::Float64, n_exc::Int64)
        v = -65.0*ones(n_neurons)
        u = 0.2 * v
        d = [(i<=n_exc) ? 8.0 : 2.0 for i in 1:n_neurons]
        a = [(i<=n_exc) ? 0.02 : 0.1 for i in 1:n_neurons]
        n_inh = n_neurons-n_exc
        connection, exc, inh = connect(n_neurons, n_exc, connectivity, [0])
        connection = transpose(connection)
        connection_exc = connection[:,exc]
        connection_inh = connection[:,inh]
        # maybe set the Exc-> Inh conn weights to high value
        s_exc = 1.0 .* ones(n_neurons, n_exc) # parameter
        s_inh = -0.5 .* ones(n_neurons, n_inh) # parameter (fixed inhibitory weight)
        s_exc .*= connection_exc
        s_inh .*= connection_inh
        sd = zeros(n_neurons,n_exc)
        STDP = zeros(n_neurons)
        ho = zeros(n_neurons)
        I = Float64[]
        da = 0.0
        arch = [0]
        new(n_exc,n_neurons,v,u,d,a,connection,connection_exc,connection_inh,
            s_exc,s_inh,sd,STDP,ho,exc,inh,I,da,arch)
    end
end


# %% functions

# direct spike of the neuron
function input!(net::Network, input_spikes::Array{Int64})
    net.v[input_spikes] .= 40.0
end

# superthreshold current processed by the neuron
function input!(net::Network, input_spikes::Array{Tuple{Int64,Float64}})
    net.I = 13 .* (rand(net.n_neurons) .- 0.5)
    max_current = 20.0                  # parameter (max suprathreshold current)
    min_current = 0.0                  # parameter (min suprathreshold current)
    indices = [input_spikes[i][1] for i in eachindex(input_spikes)]
    intensity = [input_spikes[i][2] for i in eachindex(input_spikes)]
    net.I[indices] .+= (max_current-min_current) .* intensity .+ min_current
    net.v[indices] .= @. net.v[indices]+0.5*((0.04*net.v[indices]+5)*net.v[indices]+140-net.u[indices]+net.I[indices])
    net.v[indices] .= @. net.v[indices]+0.5*((0.04*net.v[indices]+5)*net.v[indices]+140-net.u[indices]+net.I[indices])
end


function spike!(net::Network)
    # spikes
    net.I = 13 .* (rand(net.n_neurons) .- 0.5)
    spiked = findall(x->x>30,net.v .- net.ho) # parameter
    spiked_exc = filter(x->(x in net.exc), spiked)
    spiked_ho = filter(x-> x > net.arch[1], spiked)
    net.ho[spiked_ho] .+= 2.0 # parameter (theta)
    net.v[spiked] .= -65.0 # parameter (c)
    net.u[spiked] .+= net.d[spiked]
    # increment current
    for neuron in spiked
        if neuron in net.exc
            net.I .+= net.s_exc[:,neuron]
        elseif neuron in net.inh
            net.I .+= net.s_inh[:,neuron-net.n_exc]
        end
    end
    #time step (Izhikevic model)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.u .= @. net.u+net.a*(0.2*net.v-net.u)   # parameters (b)
    net.STDP[spiked] .= 0.1  # parameter (STDP_step)
    # LTP & LTD
    for neuron in spiked
        if neuron in net.exc
            net.sd[neuron,:] .+= 1.0 .* net.STDP[1:net.n_exc] #parameter (a_post) post synaptic spike
            net.sd[1:n_exc,neuron] .-= 1.5 .* net.STDP[1:net.exc] # parameter (a_pre) pre synaptic spike
        end
    end
    # time decay
    net.ho .*= 0.9999 # parameter (tau_theta)
    net.STDP .*= 0.95 # parameter (tau_STDP)
    net.da *= 0.995 # parameter (tau_da)
    spiked
end

function learn!(net::Network)
    #increase the synaptic weigths
    net.s_exc .+= (0.002+net.da) .* net.sd  # parameters
    net.s_exc .= clamp.(net.s_exc, 0, 4) # parameters (max synaptic weight)
    net.s_exc .*= net.connection_exc
    # time decay
    net.sd .*= 0.99 # parameter
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

function weight_viz(net::Network)
    weights = net.s_exc
    filtered = filter(x-> x!=0.0, weights)
    display(histogram(filtered,
                      legend = false,
                      xlabel = "Synaptic strength",
                      ylabel = "Synapse number"))
end
