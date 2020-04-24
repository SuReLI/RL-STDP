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
    shist::Array{Float64,2}

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
        shist = zeros(1000*T, 2)
        new(v,u,s,sd,STDP,firings,DA,rew,n1f,n2f,I,shist)
    end
end
# %% Main loop

net = NeuralNet()

@inbounds for sec in 0:T-1
    @time @inbounds for msec in 1:1000
        net.I = 13*(rand(N).-0.5)
        time = 1000*sec+msec
        fired = findall(x->x>=thresh,net.v)
        net.v,net.u = izhikevicmodel_fire(net.v,net.u,fired)
        net.STDP = STDP_fire(net.STDP,fired,msec)
        net.sd = LTP(net.STDP,net.sd,fired,msec)
        net.firings = vcat(net.firings,hcat(msec.*ones(length(fired)),fired))
        net.I,net.sd = LTD(net.STDP,net.sd,net.s,net.firings,net.I,msec)
        net.v,net.u = izhikevicmodel_step(net.v,net.u,net.I)
        net.STDP,net.DA = DA_STDP_step(net.STDP,net.DA,msec)
        net.s,net.sd = synweight_step(net.sd,net.s,net.DA,msec)
        net.n1f,net.n2f,net.rew = reward_fire(net.n1f,net.n2f,net.rew,fired,time)
        net.DA = DA_inc(net.rew,net.DA,time)
        net.shist[time,:] = [net.s[n1,syn],net.sd[n1,syn]]
    end
    net.STDP,net.firings = time_reset(net.STDP,net.firings)
    if sec%100==0
        print("\rsec = $sec")
    end
end

# %% Plot learning of the targeted synapse
using Plots
gr()
x1 = 0.001.*collect(1:length(net.shist[:,1]))
y1 = net.shist[:,1]
x2 = x1
y2 = 10*net.shist[:,2]
fig = plot(xlims = (0,250))
plot!(x1,y1,color="blue",label="synapse weight", legend = true)
plot!(x2,y2,color="green",label="eligibilty trace", legend = true)
xlabel!("Time (sec)")
