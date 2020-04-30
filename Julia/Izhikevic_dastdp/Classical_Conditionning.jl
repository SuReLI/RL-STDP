# %% Modules

include("../Modules/NeuronModel.jl")
include("../Modules/DASTDP.jl")

using .DASTDP
using .NeuronModel
using Statistics

# %% Constants

const S = [rand(1:800,50) for _ in 1:100] # Stimulis Sk
const T = 3600

# %% functions

function stimuli_fire(s_index::Int64,I::Array{Float64})
    ind = S[s_index]
    I[ind] = I[ind] .+ 200  #super threshold current
    return I
end

function means(s::Array{Float64,2})
    mean_cs = mean(s[S[1],:])
    mean_us = mean([mean(s[S[index],:]) for index in 2:50])
    return mean_cs,mean_us
end

function reward(rew::Array{Int64},s_index::Int64,time::Int64)
    if s_index == 1
        append!(rew,time+rand(1:1000))
    end
    return rew
end


# %% Network structure

mutable struct NeuralNet
    v::Array{Float64}
    u::Array{Float64}
    s::Array{Float64,2}
    sd::Array{Float64,2}
    STDP::Array{Float64,2}
    firings::Array{Int64,2}
    DA::Float64
    rew::Array{Int64}
    s_del::Array{Int64}
    I::Array{Float64}
    shist::Array{Float64,2}

    function NeuralNet()
        v = -65.0*ones(N)
        u = 0.2*v
        s = vcat(0.5 .* ones(Ne,M),-0.5 .* ones(Ni,M))
        sd = 0.0 .* zeros(N,M)
        STDP = 0.0 .* zeros(N,1001+D)
        firings = [-D 0]
        DA = 0.0
        rew = []
        s_del = [0,0]
        I = Float64[]
        shist = zeros(1000*T, 2)
        new(v,u,s,sd,STDP,firings,DA,rew,s_del,I,shist)
    end
end

# %% main loop

net = NeuralNet()

for sec in 0:T-1
    @time for msec in 1:1000
        net.I = 13*(rand(N).-0.5)
        time = 1000*sec+msec
        fired = findall(x->x>=thresh,net.v)
        net.v,net.u = izhikevicmodel_fire(net.v,net.u,fired)
        net.STDP = STDP_fire(net.STDP,fired,msec)
        net.sd = LTP(net.STDP,net.sd,fired,msec)
        net.firings = vcat(net.firings,hcat(msec.*ones(length(fired)),fired))
        net.I,net.sd = LTD(net.STDP,net.sd,net.s,net.firings,net.I,msec)
        if net.s_del[1] == net.s_del[2]
            s_index = rand(1:100)
            net.I = stimuli_fire(s_index,net.I)
            net.s_del = [0,rand(100:300)]
            net.rew = reward(net.rew,s_index,time)
        end
        net.v,net.u = izhikevicmodel_step(net.v,net.u,net.I)
        net.STDP,net.DA = DA_STDP_step(net.STDP,net.DA,msec)
        net.s,net.sd = synweight_step(net.sd,net.s,net.DA,msec)
        net.DA = DA_inc(net.rew,net.DA,time)
        net.shist[time,:] .= means(net.s)
        net.s_del[1] += 1
    end
    net.STDP,net.firings = time_reset(net.STDP,net.firings)
    if sec%100==0
        print("\rsec = $sec")
    end
end

# %% Plot means
using Plots
gr()
x1 = 0.001.*collect(1:length(net.shist[:,1]))
y1 = net.shist[:,1]
x2 = x1
y2 = net.shist[:,2]
fig = plot()
plot!(x1,y1,color="blue",label="from S1", legend = true)
plot!(x2,y2,color="green",label="mean", legend = true)
xlabel!("Time (sec)")
ylabel!("synaptic weigth (mV)")
