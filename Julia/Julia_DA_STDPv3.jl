# %% Modules

include("Modules/DASTDP.jl")
include("Modules/NeuronModel.jl")

using .DASTDP
using .NeuronModel

# %% Constants

const Ni = 200
const T = 3600
const thresh = 30;

# %% Network Structure

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
        v = -65.0*ones(N)
        u = 0.2*v
        s = vcat(1.0 .* ones(Ne,M),-1.0 .* ones(Ni,M))
        s[n1,syn] = 0.0
        sd = 0.0 .* zeros(N,M)
        STDP = 0.0 .* zeros(N,1001+D)
        firings = [-D*1.0 0.0]
        DA = 0.0
        rew = []
        n1f = [-100]
        n2f = []
        I = Float64[]
        new(v,u,s,sd,STDP,firings,DA,rew,n1f,n2f,I)
    end
end

# %% Main loop

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
