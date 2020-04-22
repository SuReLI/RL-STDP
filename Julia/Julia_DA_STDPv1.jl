# %% imports

# %% define structures (Network,Parameters)

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
    shist::Array{Float64,2}
    index_I::Array{Tuple{Int64,Float64}}
end

struct Parameters
    M::Int64
    D::Int64
    Ne::Int64
    Ni::Int64
    N::Int64
    sm::Int64
    T::Int64
    n1::Int64
    syn::Int64
    n2::Int64
    interval::Int64
end

struct Connect
    post::Array{Int64,2}
    pre::Vector{Any}
    delays::Vector{Any}
end

struct Step
    a::Array{Float64}
    d::Array{Int64}
end

# %% define init functions

function initnet(p::Parameters)
    v = -65*ones(p.N)
    u = 0.2*v
    s = vcat(ones(p.Ne,p.M),-ones(p.Ni,p.M))
    s[p.n1,p.syn] = 0
    sd = zeros(p.N,p.M)
    STDP = zeros(p.N,1001+p.D)
    firings = [-p.D 0]
    DA = 0
    rew = []
    n1f = [-100]
    n2f = []
    shist = zeros(1000*p.T, 2)
    ind_I = []
    net = NeuralNet(v,u,s,sd,STDP,firings,DA,rew,n1f,n2f,shist,ind_I)
    return net
end

function initcon(net::NeuralNet,p::Parameters)
    post = vcat(rand(1:p.N,p.Ne,p.M),rand(1:p.Ne,p.Ni,p.M))
    post[p.n1,p.syn] = p.n2
    delays = []
    pre = []
    for i in 1:p.N
        if i <= p.Ne
            del_temp = []
            for j in 1:p.D
                start = (p.M/p.D)*(j-1)+1
                fin = p.M/p.D*(j)
                append!(del_temp,[collect(start:fin)])
            end
        else
            del_temp = [[] for i in 1:p.D]
            del_temp[1] = collect(1:p.M)
        end
        delays = vcat(delays,del_temp)
        append!(pre,[collect([index for index in findall(x->x==i,post) if net.s[index]>0])])
    end
    con = Connect(post,pre,delays)
    return con
end


# %% Construct param, initialize net, connections and step

p = Parameters(100,1,800,200,1000,4,3600,1,1,rand(2:800),20)
net = initnet(p)
con = initcon(net,p)
step = Step([(i<=p.Ne) ? 0.02 : 0.1 for i in 1:p.N],[(i<=p.Ne) ? 8 : 2 for i in 1:p.N])

# %% secondary functions

msec_time(msec::Int64,sec::Int64) = 1000*sec+msec

v_step!(v::Array{Float64},u::Array{Float64},I::Array{Float64}) = @. v+0.5*((0.04*v+5)*v+140-u+I)
u_step!(v::Array{Float64},u::Array{Float64},a::Array{Float64}) = @. u+a*(0.2*v-u)

function s_step!(s::Array{Float64,2},sd::Array{Float64,2},DA::Float64,p::Parameters)
    s[1:p.Ne,:] = @. max(0,min(p.sm,s[1:p.Ne,:]+(0.002+DA)*sd[1:p.Ne,:]))
    sd = @. 0.99*sd
end

function shist_step!(shist::Array{Float64,2},s::Array{Float64,2},sd::Array{Float64,2},msec::Int64,sec::Int64,p::Parameters)
    shist[msec_time(msec,sec),:] = [s[p.n1,p.syn],sd[p.n1,p.syn]]
end

function new_I!(index_I::Array{Tuple{Int64,Float64}})
    I = @. 13*(rand(N)-0.5)
    for tuple in ind_I
        I[tuple[1]] += tuple[2]
    end
    ind_I = []
    return I
end


# %% primal functions (BIG)

function fireall!(net::NeuralNet,con::Connect,step::Step,p::Parameters,sec::Int64,msec::Int64)
    time = msec_time(msec,sec)
    fired = findall(x->x>=30,net.v)
    net.v[fired] .= -65
    net.u[fired] .= net.u[fired] .+ step.d[fired]
    net.firings = vcat(net.firings,hcat(time*ones(length(fired)),fired))
    if length(fired)!=0
        net.STDP[fired,msec+p.D] .= 0.1
    end
    for k in fired
        pre_neurons_k = [con.pre[k][i][1] for i in 1:length(con.pre[k])]
        net.sd[con.pre[k]] = net.sd[con.pre[k]]  .+  net.STDP[pre_neurons_k,msec]
    end
    if p.n1 in fired
        append!(net.n1f,msec_time)
    end
    if p.n2 in fired
        append!(net.n2f,msec_time)
        if (msec_time-last(net.n1f)<p.interval) && (last(net.n2f)>last(net.n1f))
            append!(net.rew,msec_time+1000+rand(1:2000))
        end
    end
end

function LTD!(net::NeuralNet,con::Connect,step::Step,sec::Int64,msec::Int64)
    time = msec_time(msec,sec)
    last_ = length(net.firings[:,1])
    while net.firings[last_,1]>time-D
        del = con.delays[net.firings[last_,2]][time-net.firings[last_,1]+1]
        ind = con.post[net.firings[last_,2], del]
        append!(net.index_I,(ind,net.s[net.firings[last_,2],del]))
        net.sd[net.firings[last_,2],del] = net.sd[net.firings[last_,2],del] .- 1.5 .* net.STDP[ind,t+D]
        last_ -= 1
    end
end

function net_step!(net::NeuralNet,p::Parameters,msec::Int64,sec::Int64)
    time = msec_time(msec,sec)
    net.STDP[:,msec+p.D+1] = 0.95 .* net.STDP[:,msec+p.D]
    net.DA = net.DA*0.995
    if time in net.rew
        DA += 0.5
    end
    I = new_I(net.index_I)
    net.v = v_step!(net.v,net.u,I)
    net.v = v_step!(net.v,net.u,I)
    net.u = u_step!(net.v,net.u,net.a)
    if msec%10===0
        s_step!(net.s,net.sd,net.DA,p)
    end
    shist_step!(net.shist,net.s,net.sd,msec,sec,p)
end

function time_reset!(net::NeuralNet, p::Parameters)
    net.STDP[:,1:p.D+1]=net.STDP[:,1001:1001+p.D]
    ind = findall(x->x>1001-p.D,net.firings[:,1])
    net.firings = vcat([-p.D 0],hcat(net.firings[ind,1] .- 1000,net.firings[ind,2]))
end


#%% main loop

function main_loop!(net::NeuralNet,con::Connect,step::Step,p::Parameters)
    for sec in 0:p.T-1
        for msec in 1:1000
            fireall!(net,con,step,p,sec,msec)
            LTD!(net,con,step,sec,msec)
            net_step!(net,p,msec,sec)
        end
        time_reset!(net,p)
        if sec%100==0
            print("\rsec = $sec")
        end
    end
end

# %%
main_loop!(net,con,step,p)
