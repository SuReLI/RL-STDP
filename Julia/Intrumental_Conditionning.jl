# %% Modules

include("Modules/NeuronModel.jl")
include("Modules/DASTDP.jl")

using .DASTDP
using .NeuronModel
using Statistics

# %% Constants

const T = 400
randarray = rand(1:998,50)
const A = randarray
const B = randarray .+ 1
const S = randarray .+ 2

syn_SA_tmp = []
syn_SB_tmp = []

for pre_neuron in pre[A]
    indexes_A = [cartind for cartind in pre_neuron if cartind[1] in S]
    append!(syn_SA_tmp,indexes_A)
end

for pre_neuron in pre[B]
    indexes_B = [cartind for cartind in pre_neuron if cartind[1] in S]
    append!(syn_SB_tmp,indexes_B)
end

const syn_SA = syn_SA_tmp
const syn_SB = syn_SB_tmp

# learning rate related
const DAinc = 0.5
const STDPinc = 0.1
const sm = 15
const LTDinc = 1.5


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
    card::Array{Int64}                       # card[A,B]
    target::Int64                            # target = 1 : group A , target = 2 : group B
    I::Array{Float64}
    shist::Array{Float64,2}
    phist::Array{Float64,2}

    function NeuralNet()
        v = -65.0*ones(N)
        u = 0.2*v
        s = vcat(1.0 .* ones(Ne,M),-2.0 .* ones(Ni,M))
        sd = 0.0 .* zeros(N,M)
        STDP = 0.0 .* zeros(N,1001+D)
        firings = [-D 0]
        DA = 0.0
        rew = []
        card = [0,0]
        target = 1
        I = Float64[]
        shist = zeros(1000*T,4)
        phist = zeros(T, 2)
        new(v,u,s,sd,STDP,firings,DA,rew,card,target,I,shist,phist)
    end
end

# %% Functions

function stimuli_fire(I::Array{Float64})
    I[S] = I[S] .+ 16
    return I
end

function cardAB(fired::Array{Int64},card::Array{Int64})
    for neuron in fired
        if neuron in A
            card[1] += 1
        elseif neuron in B
            card[2] += 1
        end
    end
    return card
end

function reward(rew::Array{Int64},card::Array{Int64},target::Int64,time::Int64)
    if card[target]>card[3-target]
        delay = 1000*(card[3-target]/card[target])
        append!(rew,time + 1 + round(delay))
    end
    return rew
end

function probAB(card::Array{Int64},phist::Array{Float64,2},sec::Int64)
    if sum(card)!=0
        pA_new = card[1]/(sum(card))
        pB_new = card[2]/(sum(card))
    else
        pA_new = 0
        pB_new = 0
    end
    if sec>=1
        ind = min(sec,20)
        meanwindA = mean(phist[sec+1-ind:sec,1])
        meanwindB = mean(phist[sec+1-ind:sec,2])
        pA_new = mean([meanwindA,pA_new])
        pB_new = mean([meanwindB,pB_new])
    end
    return pA_new,pB_new
end

function means(s::Array{Float64,2},sd::Array{Float64,2})
    @inbounds mean_SA = mean(s[syn_SA])
    @inbounds mean_SB = mean(s[syn_SB])
    @inbounds meansdA = mean(sd[syn_SA])
    @inbounds meansdB = mean(sd[syn_SB])
    return mean_SA,mean_SB,meansdA,meansdB
end


# %% Main loop

net = NeuralNet()

@inbounds for sec in 0:T-1
    @time @inbounds for msec in 1:1000
        net.I = 13*(rand(N).-0.5)
        time = 1000*sec+msec
        fired = findall(x->x>=thresh,net.v)
        net.v,net.u = izhikevicmodel_fire(net.v,net.u,fired)
        net.STDP = STDP_fire(net.STDP,fired,msec,STDPinc)
        net.sd = LTP(net.STDP,net.sd,fired,msec)
        net.firings = vcat(net.firings,hcat(msec.*ones(length(fired)),fired))
        ind = 0
        net.I,net.sd = LTD(net.STDP,net.sd,net.s,net.firings,net.I,msec,LTDinc)
        if msec==1
            net.I = stimuli_fire(net.I)
            net.card = [0,0]
        end
        if msec<=20
            net.card = cardAB(fired,net.card)
        end
        if msec==20
            net.rew = reward(net.rew,net.card,net.target,time)
            net.phist[sec+1,:] .= probAB(net.card,net.phist,sec)
        end
        net.v,net.u = izhikevicmodel_step(net.v,net.u,net.I)
        net.STDP,net.DA = DA_STDP_step(net.STDP,net.DA,msec)
        net.s,net.sd = synweight_step(net.sd,net.s,net.DA,msec,sm)
        net.DA = DA_inc(net.rew,net.DA,time,DAinc)
        net.shist[time,:] .= means(net.s,net.sd)
    end
    if sec==199
        net.target = 2
    end
    net.STDP,net.firings = time_reset(net.STDP,net.firings)
    if sec%100==0
        print("\rsec = $sec")
    end
end

# %% Plot probability
using Plots
gr()
x1 = collect(1:length(net.phist[:,1]))
y1 = net.phist[:,1]
x2 = x1
y2 = net.phist[:,2]
fig = plot()
plot!(x1,y1,color="blue",label="Group A", legend = true)
plot!(x2,y2,color="green",label="Group B", legend = true)
xlabel!("Trial")
ylabel!("Probability of response")

# %% Plot synaptic weight in A and B
using Plots
gr()
x1 = 0.001.*collect(1:length(net.shist[:,1]))
y1 = net.shist[:,1]
x2 = x1
y2 = net.shist[:,2]
fig = plot()
plot!(x1,y1,color="blue",label="S->A", legend = true)
plot!(x2,y2,color="green",label="S->B", legend = true)
xlabel!("Time (sec)")
ylabel!("mean synaptic weigth (mV)")

# %% Plot mean sd for A and B
using Plots
gr()
x1 = 0.001.*collect(1:length(net.shist[:,1]))
y1 = net.shist[:,3]
x2 = x1
y2 = net.shist[:,4]
fig = plot()
plot!(x1,y1,color="blue",label="S->A", legend = true)
plot!(x2,y2,color="green",label="S->B", legend = true)
xlabel!("Time (sec)")
ylabel!("mean synaptic weigth derivative")
