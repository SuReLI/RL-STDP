using Plots

function lif!(v,I)
    v = v + 0.2*(I-v-70)
    return v
end

function step_v()
    vhist = []
    v = -50
    I = 0.0
    for i in 1:30
        if i%4==0
            I += 2
        end
        v = lif!(v,I)
        @show v
        push!(vhist, v)
    end
    return vhist
end
# %%
vhist = step_v()
# %%

plot(vhist)

#%%
v= -50
#%%
v = lif!(v,0.0)

# %%

using Plots

include("../../../Modules/Networks/net_arch.jl")
include("../../../Modules/Poisson/PoissonStateSpike.jl")

params = YAML.load(open("Julia/CartPole/LIF_base/cfglif4820.yml"))
n = Network([1,1], params)

# %%
n.v[1] = 0
n.s.e[1][1] = 0.7
n.v[2] = 0

times = [5,7,9,15,17]

vh = []
x = []

for t in 1:30
    spiked = step_LIF!(n, [(1,0.4)], false)
    push!(x,t)
    push!(vh, n.v[1])
end

vh2 = []
n.v[1] = 0

for t in 1:30
    spiked = step_LIF!(n, [(1,0.8)], false)
    push!(vh2, n.v[1])
end

plot(x,vh, line = :blue, label = "Membrane Potential I=0.4")
plot!(x,vh2, line = :red, label = "Membrane Potential I=0.8")
hline!([1.0], line = (3,:dash, :green), label = "Threshold Potential")
xlabel!("Time (ms)")
ylabel!("Potential")
