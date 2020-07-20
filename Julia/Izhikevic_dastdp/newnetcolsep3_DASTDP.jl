
include("../Modules/new_net_col_sep3.jl")

# %%

n = Network(1000, 0.1, 800)

# %%

function post_exc(neuron,connection_exc)
    return findall(x->x==1,connection_exc[neuron,:])
end

const n1 = 2
const n2 = post_exc(n1,n.connection_ee)[1]
const interval = 20 #msec

n.s_ee[n1,n2] = 0.0
n1f = [-100]
n2f = Int64[]
rew = Int64[]

function reward!(n1f::Array{Int64},n2f::Array{Int64},rew::Array{Int64},spiked::Array{Int64},time::Int64) # Module DA_STDP
    if n1 in spiked
        append!(n1f,time)
    end
    if n2 in spiked
        append!(n2f,time)
        if (time-last(n1f)<interval) && (last(n2f)>last(n1f))
            append!(rew,time+1000+rand(1:2000))
        end
    end
end

shist = zeros(10*3600,2)



# %%



@inbounds for sec in 0:3599
    @show sec
    @time @inbounds for msec in 1:1000
        time = 1000*sec+msec
        train = false
        if msec%10 == 0
            train = true
        end
        spiked = step!(n,train)
        # reward!(n1f,n2f,rew,spiked,time)
        # if time in rew
        #     n.da += 0.5
        # end
        if msec%100==0
            shist[div(time,100),:] .= n.s_ee[n1,n2],n.sd_ee[n1,n2]
        end
    end
end


# %% Plot learning of the targeted synapse
using Plots
gr()
x1 = 0.1.*collect(1:length(shist[:,1]))
y1 = shist[:,1]
x2 = x1
y2 = shist[:,2]
fig = plot()
plot!(x1,y1,color="blue",label="synapse weight", legend = true)
plot!(x2,y2,color="green",label="eligibilty trace", legend = true)
xlabel!("Time (sec)")
