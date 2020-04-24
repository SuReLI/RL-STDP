module DASTDP


##### Exports

export STDP_fire
export LTP
export LTD
export DA_STDP_step
export synweight_step
export reward_fire
export DA_inc
export time_reset
export M
export D
export Ne
export N
export n1
export syn

##### Constants


const M = 100
const D = 1
const Ne = 800
const Ni = 200
const N = Ne+Ni
const sm = 4
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


##### Functions

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

end #end of module
