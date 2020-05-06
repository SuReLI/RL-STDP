using LinearAlgebra


mutable struct Network

    v::Array{Float64}
    u::Array{Float64}
    d::Array{Float64}
    a::Array{Float64}

    connection::Array{Int64,2}
    s::Array{Float64,2}
    sd::Array{Float64,2}
    STDP::Array{Float64}

    exc::Array{Int64}
    inh::Array{Int64}

    I::Array{Float64}
    da::Float64
    #param::Dict

    function connect(n_neurons::Int64, n_exc::Int64, connectivity::Float64,
                     arch::Array{Int64})
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
                connection[n_exc+sii1+1:n_exc+sii2, si2+1:si3] .= ones(a2,a2) .- I(a2)
                connection[si2+1:si3, n_exc+sii1+1:n_exc+sii2] .= I(a2)
            end
        end
        exc = collect(1:n_exc)
        inh = collect(n_exc+1:n_neurons)
        return connection, exc, inh
    end


    function Network(n_neurons::Int64, connectivity::Float64, n_exc::Int64,
                     arch::Array{Int64} = [0])
        v = -65.0*ones(n_neurons)
        u = 0.2 * v
        d = [(i<=n_exc) ? 8.0 : 2.0 for i in 1:n_neurons]
        a = [(i<=n_exc) ? 0.02 : 0.1 for i in 1:n_neurons]
        n_inh = n_neurons-n_exc
        connection, exc, inh = connect(n_neurons, n_exc, connectivity, arch)
        s = zeros(n_neurons,n_neurons)
        # maybe set the Exc-> Inh conn weights to high value
        s[exc,:] .= 1.0 .* ones(n_exc, n_neurons)
        s[inh,:] .= -1.0 .* ones(n_inh, n_neurons)
        s .*= connection
        sd = zeros(n_neurons,n_neurons)
        STDP = zeros(n_neurons)
        I = Float64[]
        da = 0.0
        new(v,u,d,a,connection,s,sd,STDP,exc,inh,I,da)
    end
end


# %% functions

function input!(net::Network, input_spikes::Array{Int64})
    net.v[input_spikes] .= 40.0
end


function spike!(net::Network)
    # spikes
    net.I = 13 .* (rand(length(net.STDP)) .- 0.5)
    spiked = findall(x->x>30,net.v)
    net.v[spiked] .= -65.0
    net.u[spiked] .+= net.d[spiked]
    net.STDP[spiked] .= 0.1
    # LTP & LTD
    for neuron in spiked
        net.sd[:,neuron] .+= 1.0 .* net.STDP
        net.sd[neuron,:] .-= 1.5 .* net.STDP
    end
    # increment current
    for neuron in spiked
        net.I .+= net.s[neuron,:]
    end
    #time step (Izhikevic model)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.v .= @. net.v+0.5*((0.04*net.v+5)*net.v+140-net.u+net.I)
    net.u .= @. net.u+net.a*(0.2*net.v-net.u)
    # time decay
    net.sd .*= 0.999
    net.STDP .*= 0.95
    net.da *= 0.995
    spiked
end

function learn!(net::Network)
    #@time net.s[net.exc,:] .= net.s[net.exc,:] + (0.002+net.da) .* net.sd[net.exc,:]
    @time net.s[net.exc,:] .+= (0.002+net.da) .* net.sd[net.exc,:]
    net.s .*= net.connection
    net.s .= clamp.(net.s,0,4)
end



function step!(net::Network, train::Bool=false)
    spiked = spike!(net)
    if train == true
        learn!(net,spiked)
    end
    spiked
end

function step!(net::Network, input_spikes::Array{Int64}, train::Bool=true)
    input!(net,input_spikes)
    spiked = spike!(net)
    if train == true
        learn!(net,spiked)
    end
    spiked
end

# %% test the net by implementing basic DASTDP
