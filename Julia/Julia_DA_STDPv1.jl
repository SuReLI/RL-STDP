# %% imports

# %% define structures (Network,Parameters)

struct NeuralNet
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
    interval::Int64
end

struct Connect
    post::Array{Int64,2}
    pre::Vector{Any}
    delays::Vector{Any}
end

# %% define functions

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
    net = NeuralNet(v,u,s,sd,STDP,firings,DA,rew,n1f,n2f,shist)
    return net
end

function initcon(net::NeuralNet,p::Parameters)
    post = vcat(rand(1:p.N,p.Ne,p.M),rand(1:p.Ne,p.Ni,p.M))
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

# %% Construct param, initialize net and connections

p = Parameters(100,1,800,200,1000,4,3600,1,1,20)
net = initnet(p)
con = initcon(net,p)

# %% Rest of the initialization I don't know how to implement in struct or function

a = [(i<=p.Ne) ? 0.02 : 0.1 for i in 1:p.N]
d = [(i<=p.Ne) ? 8 : 2 for i in 1:p.N]
n2 = con.post[p.n1,p.syn];

# %% main loop implemented in functions
