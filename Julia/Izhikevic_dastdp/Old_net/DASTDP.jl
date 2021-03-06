module DASTDP


##### Exports

export STDP_fire
export LTP
export LTD
export DA_STDP_step
export synweight_step
export DA_inc
export time_reset
export M
export D
export Ni
export Ne
export N
export post
export pre

##### Constants


const M = 100
const D = 1
const Ne = 800
const Ni = 200
const N = Ne+Ni



post_tmp = vcat(rand(1:N,Ne,M),rand(1:Ne,Ni,M))


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

function STDP_fire(STDP::Array{Float64,2},fired::Array{Int64},msec::Int64,STDPinc::Float64=0.1) # Module DA_STDP
    if length(fired) != 0
        STDP[fired,msec+D] .= STDPinc
    end
    return STDP
end


function LTP(STDP::Array{Float64,2},sd::Array{Float64,2},fired::Array{Int64},msec::Int64,LTPinc::Float64=1.0)  # Module DA_STDP
    for index in eachindex(fired)
        pre_neurons = [pre[fired[index]][i][1] for i in eachindex(pre[fired[index]])]
        sd[pre[fired[index]]] .= sd[pre[fired[index]]] .+ LTPinc .* STDP[pre_neurons,msec]
    end
    return sd
end

function LTD(STDP::Array{Float64,2},sd::Array{Float64,2},s::Array{Float64,2},firings::Array{Int64,2},I::Array{Float64},msec::Int64,LTDinc::Float64=1.5) # Module DA_STDP
    last_ = length(firings[:,1])
    while firings[last_,1]>msec-D
        del = delays[firings[last_,2]]
        ind = post[firings[last_,2], del]
        I[ind] = I[ind] .+ s[firings[last_,2],del]
        sd[firings[last_,2],del] = sd[firings[last_,2],del] .- (LTDinc .* STDP[ind,msec+D])
        last_ = last_ - 1
    end
    return I,sd
end

function DA_STDP_step(STDP::Array{Float64,2},DA::Float64,msec::Int64) # Module DA_STDP
    STDP[:,msec+D+1] .= 0.95 .* STDP[:,msec+D]
    DA = DA*0.995
    return STDP,DA
end

function synweight_step(sd::Array{Float64,2},s::Array{Float64,2},DA::Float64,sm::Int64=4) # Module DA_STDP
    s[1:Ne,:] .= clamp.(s[1:Ne,:] .+ ((0.002+DA) .* sd[1:Ne,:]),0,sm)
    sd .= 0.99.*sd
    return s,sd
end

function DA_inc(rew::Array{Int64},DA::Float64,time::Int64,DAinc::Float64=0.5) # Module DA_STDP
    if time in rew
        DA += DAinc
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
