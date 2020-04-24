# %% Define constants

const M= 100
const D = 1
const Ne = 800
const Ni = 200
const N = Ne+Ni
const sm = 4
const T = 3600
const n1 = 1
const syn= 1
const interval = 20


post_tmp = vcat(rand(1:N,Ne,M),rand(1:Ne,Ni,M))
const n2 = post_tmp[n1,syn]

function initcon(N,M,post)
    delays = Array{Int64}[]
    pre = Array{CartesianIndex{2}}[]
    for i in 1:N
        delays = vcat(delays,[collect(1:M)])
        append!(pre,[collect([index for index in findall(x->x==i,post) if index[1]<=800])])
    end
    return pre,delays
end

pre_tmp ,delays_tmp= initcon(N,M,post_tmp)

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
    shist::Array{Float64,2}

    function NeuralNet()
        v = -65.0*ones(N)
        u = 0.2*v
        s = vcat(1.0 .* ones(Ne,M),-1.0 .* ones(Ni,M))
        s[n1,syn] = 0.0
        sd = 0.0 .* zeros(N,M)
        STDP = 0.0 .* zeros(N,1001+D)
        firings = [-D 0]
        DA = 0.0
        rew = []
        n1f = [-100]
        n2f = []
        I = Float64[]
        shist = zeros(1000*T, 2)
        new(v,u,s,sd,STDP,firings,DA,rew,n1f,n2f,I,shist)
    end
end


# %% functions

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

function STDP_fire(STDP::Array{Float64,2},fired::Array{Int64},msec::Int64) # Module DA_STDP
    if length(fired) != 0
        STDP[fired,msec+D] .= 0.1
    end
    return STDP
end

function LTP(STDP::Array{Float64,2},sd::Array{Float64,2},fired::Array{Int64},msec::Int64)  # Module DA_STDP
    for neuron in fired
        pre_neurons = [pre[neuron][i][1] for i in eachindex(pre[neuron])]
        sd[pre[neuron]] .= sd[pre[neuron]] .+ STDP[pre_neurons,msec]
    end
    return sd
end

function LTD(STDP::Array{Float64,2},sd::Array{Float64,2},s::Array{Float64,2},firings::Array{Int64,2},I::Array{Float64},msec::Int64) # Module DA_STDP
    last_ = length(firings[:,1])
    while firings[last_,1]>msec-D
        del = delays[firings[last_,2]][msec-firings[last_,1]+1]
        ind = post[firings[last_,2], del]
        I[ind] = I[ind] .+ s[firings[last_,2],del]
        sd[firings[last_,2],del] = sd[firings[last_,2],del] .- (1.5 .* STDP[ind,msec+D])
        last_ -= 1
    end
    return I,sd
end

function DA_STDP_step(STDP::Array{Float64,2},DA::Float64,msec::Int64) # Module DA_STDP
    STDP[:,msec+D+1] .= 0.95 .* STDP[:,msec+D]
    DA = DA*0.995
    return STDP,DA
end

function synweight_step(sd::Array{Float64,2},s::Array{Float64,2},DA::Float64,msec::Int64) # Module DA_STDP
    if msec%10==0
        s[1:Ne,:] .= max.(0,min.(sm,s[1:Ne,:] .+ ((0.002+DA) .* sd[1:Ne,:])))
        sd = 0.99*sd
    end
    return s,sd
end

function reward_fire(n1f::Array{Int64},n2f::Array{Int64},rew::Array{Int64},fired::Array{Int64},time::Int64) # Module DA_STDP
    if n1 in fired
        append!(n1f,time)
    end
    if n2 in fired
        append!(n2f,time)
        if (time-last(n1f)<interval) && (last(n2f)>last(n1f))
            append!(rew,time+1000+rand(1:2000))
        end
    end
    return n1f,n2f,rew
end

function DA_inc(rew::Array{Int64},DA::Float64,time::Int64) # Module DA_STDP
    if time in rew
        DA += 0.5
    end
    return DA
end

function time_reset(STDP::Array{Float64,2},firings::Array{Int64,2}) # Module DA_STDP
    STDP[:,1:D+1]=STDP[:,1001:1001+D]
    ind = findall(x->x>1001-D,firings[:,1])
    firings = vcat([-D 0],hcat(firings[ind,1].-1000,firings[ind,2]))
    return STDP,firings
end


#%% main loop

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

# %% Plot 100 sec
using Plots
gr()
x1 = 0.001.*collect(1:length(net.shist[:,1]))
y1 = net.shist[:,1]
x2 = x1
y2 = 10*net.shist[:,2]
fig = plot(xlims = (0,1000))
plot!(x1,y1,color="blue",label="synapse weight", legend = true)
plot!(x2,y2,color="green",label="eligibilty trace", legend = true)
xlabel!("Time (sec)")
