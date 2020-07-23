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

include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")

params = YAML.load(open("Julia/CartPole/LIF_base/cfglif.yml"))
n = Network([4,8,2], params)

n.v[1] = -70
vh = []
sh = []
push!(vh, n.v[1])
# I_in = 40.0
spiked = step_LIF!(n,[(1,rand(1.0:50.0))], false)
push!(vh, n.v[1])
push!(sh, spiked)
spiked = step_LIF!(n,[(1,rand(1.0:50.0))], false)
push!(vh, n.v[1])
push!(sh, spiked)

for i in 1:10
    spiked = step_LIF!(n,[(1,rand(1.0:50.0))], false)
    push!(vh, n.v[1])
    push!(sh, spiked)
    spiked = step_LIF!(n,[(1,rand(1.0:50.0))], false)
    push!(vh, n.v[1])
    push!(sh, spiked)
end

plot(vh)
@show sh
