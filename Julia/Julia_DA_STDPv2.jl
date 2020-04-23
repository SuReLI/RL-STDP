# %% Define constants

const M = 100
const D = 1
const Ne = 800
const Ni = 200
const N = Ne+Ni
const sm = 4
const T = 3600
const n1 = 1
const syn = 1
const interval = 20


post_tmp = vcat(rand(1:N,Ne,M),rand(1:Ne,Ni,M))
const n2 = post_tmp[n1,syn]

function initcon(N,Ne,D,M,post)
    delays = []
    pre = []
    for i in 1:N
        if i <= Ne
            del_temp = []
            for j in 1:D
                start = Int64((M/D)*(j-1)+1)
                fin = Int64(M/D*(j))
                append!(del_temp,[collect(start:fin)])
            end
        else
            del_temp = [[] for i in 1:D]
            del_temp[1] = collect(1:M)
        end
        delays = vcat(delays,del_temp)
        append!(pre,[collect([index for index in findall(x->x==i,post) if index[1]<=800])])
    end
    return pre,delays
end

pre_tmp ,delays_tmp= initcon(N,Ne,D,M,post_tmp)

const post = post_tmp
const pre = pre_tmp
const delays = delays_tmp


const a = [(i<=Ne) ? 0.02 : 0.1 for i in 1:N]
const d = [(i<=Ne) ? 8.0 : 2.0 for i in 1:N]

const thresh = 30;

# %% define network struct

mutable struct NeuralNet
    v::Array{Float64}
    u::Array{Float64}
    s::Array{Float64,2}
    sd::Array{Float64,2}
    STDP::Array{Float64,2}
    firings::Array{Int64,2}
    DA::Float64
    rew::Array{Int64}
    n1f::Array{Int64}
    n2f::Array{Int64}
    I::Array{Float64}

    function NeuralNet()
        v = -65*ones(N)
        u = 0.2*v
        s = vcat(ones(Ne,M),-ones(Ni,M))
        s[n1,syn] = 0
        sd = zeros(N,M)
        STDP = zeros(N,1001+D)
        firings = [-D 0]
        DA = 0
        rew = []
        n1f = [-100]
        n2f = []
        I = []
        new(v,u,s,sd,STDP,firings,DA,rew,n1f,n2f,I)
    end
end


# %% functions

function izhikevicmodel_step!(v::Array{Float64},u::Array{Float64},I::Array{Float64}) # Module neuron_model
    v .= @. v+0.5*((0.04*v+5)*v+140-u+I)
    v .= @. v+0.5*((0.04*v+5)*v+140-u+I)
    u .= @. u+a*(0.2*v-u)
end

function izhikevicmodel_fire!(v::Array{Float64},u::Array{Float64},fired::Array{Int64}) # Module neuron_model
    if length(fired) != 0
        v[fired] .= -65.0
        u[fired] .= u[fired] .+ d[fired]
    end
end

function STDP_fire!(STDP::Array{Float64,2},fired::Array{Int64},msec::Int64) # Module DA_STDP
    if length(fired) != 0
        STDP[fired,msec+D] .= 0.1
    end
end

function LTP!(STDP::Array{Float64,2},sd::Array{Float64,2},fired::Array{Int64},msec::Int64)  # Module DA_STDP
    for neuron in fired
        pre_neurons = [pre[neuron][i][1] for i in eachindex(pre[neuron])]
        sd[pre[neuron]] .= sd[pre[neuron]] .+ STDP[pre_neurons,msec]
    end
end

function LTD!(STDP::Array{Float64,2},sd::Array{Float64,2},s::Array{Float64,2},firings::Array{Int64,2},I::Array{Float64},time::Int64,msec::Int64) # Module DA_STDP
    last_ = length(firings[:,1])
    while firings[last_,1]>time-D
        del = delays[firings[last_,2]][time-net.firings[last_,1]+1]
        ind = post[firings[last_,2], del]
        I[ind] = I[ind] .+ s[firings[last_,2],del]
        sd[firings[last_,2],del] = sd[firings[last_,2],del] .- (1.5 .* STDP[ind,msec+D])
        last_ -= 1
    end
end

function DA_STDP_step!(STDP::Array{Float64,2},DA::Float64,msec::Int64) # Module DA_STDP
    STDP[:,msec+D+1] .= 0.95 .* STDP[:,msec+D]
    DA = DA*0.995
end

function synweight_step!(sd::Array{Float64,2},s::Array{Float64,2},DA::Float64) # Module DA_STDP
    s[1:Ne,:] = max.(0,min.(sm,s[1:Ne,:] .+ ((0.002+DA) .* sd[1:Ne,:])))
    sd = 0.99 .* sd
end

function reward_fire!(n1f::Array{Int64},n2f::Array{Int64},rew::Array{Int64},fired::Array{Int64},time::Int64) # Module DA_STDP
    if n1 in fired
        append!(n1f,time)
    end
    if n2 in fired
        append!(n2f,time)
        if (time-last(n1f)<interval) && (last(n2f)>last(n1f))
            append!(rew,time+1000+rand(1:2000))
        end
    end
end

function DA_inc!(rew::Array{Int64},DA::Float64,time::Int64) # Module DA_STDP
    if time in rew
        DA += 0.5
    end
end

function time_reset!(STDP::Array{Float64,2},firings::Array{Int64,2}) # Module DA_STDP
    STDP[:,1:D+1]=STDP[:,1001:1001+D]
    ind = findall(x->x>1001-D,firings[:,1])
    firings = vcat([-D 0],hcat(firings[ind,1] .- 1000,firings[ind,2]))
end


#%% main loop

net = NeuralNet()

@inbounds for sec in 0:T-1
    @time @inbounds for msec in 1:1000
        net.I = 13*(rand(N).-0.5)
        time = 1000*sec+msec
        fired = findall(x->x>=thresh,net.v)
        izhikevicmodel_fire!(net.v,net.u,fired)
        STDP_fire!(net.STDP,fired,msec)
        reward_fire!(net.n1f,net.n2f,net.rew,fired,time)
        LTP!(net.STDP,net.sd,fired,msec)
        net.firings = vcat(net.firings,hcat(msec.*ones(length(fired)),fired))
        LTD!(net.STDP,net.sd,net.s,net.firings,net.I,time,msec)
        izhikevicmodel_step!(net.v,net.u,net.I)
        DA_STDP_step!(net.STDP,net.DA,msec)
        DA_inc!(net.rew,net.DA,time)
    end
    time_reset!(net.STDP,net.firings)
    if sec%100==0
        print("\rsec = $sec")
    end
end
