# %% imports

include("../../Modules/save_param.jl")
include("../../Modules/Networks/net_arch.jl")

# %% usefull functions

function borne(x,a,b)
    y = a + 0.5*(b-a)*(1+cos(pi*x/10))
    return y
end



# %% function that creates the network

function set_weight!(net::Network, genes_w::Array{Float64})
    w = net.s.e
    for mat_idx in 1:length(w)
        for idx in eachindex(w[mat_idx])
            offset = sum([length(mat) for mat in w[1:mat_idx-1]])
            net.s.e[mat_idx][idx] = borne(genes_w[idx+offset], 0.0, 1.1)
        end
    end
end

function set_weight!(net::Network, genes_w::Array{Float64}, exc::Array{Int64}, inh::Array{Int64})
    w = net.s.e
    inh_count = 0
    for mat_idx in 1:length(w)
        for idx in eachindex(w[mat_idx])
            offset = sum([length(mat) for mat in w[1:mat_idx-1]]) - inh_count
            offset_neuron = sum(net.arch[1:mat_idx-1])
            (idx%size(w[mat_idx])[1]==0) ? line = size(w[mat_idx])[1] : line = idx%size(w[mat_idx])[1]
            current_neuron = line+offset_neuron
            if current_neuron in exc
                # net.s.e[mat_idx][idx] = abs(genes_w[idx+offset])
                net.s.e[mat_idx][idx] = borne(genes_w[idx+offset], 0.0, 1.1)
            elseif current_neuron in inh
                # net.s.e[mat_idx][idx] = - abs(genes_w[end])
                net.s.e[mat_idx][idx] = borne(genes_w[end], -2.0, 0.0)
                inh_count += 1
            end
        end
    end
end

function ei_idx(arch::Array{Int64})
    inh = Int64[]
    exc = Int64[]
    n_inh = [div(arch[i],4) for i in 1:length(arch)-1]
    push!(n_inh, 0)
    for l in 1:length(arch)
        for neuron in 1:arch[l]
            offset = sum(arch[1:l-1])
            if neuron <= n_inh[l]
                push!(inh, neuron+offset)
            else
                push!(exc, neuron+offset)
            end
        end
    end
    return exc, inh
end

function init_teacher_student(filepath::String, indiv::Int64)
    optim_param = YAML.load(open("Julia/CartPole/LIF_base/cfg_cmaes.yml"))
    arch = optim_param["arch"]
    results = load_param(filepath)
    genes = results["weights"][indiv]
    genes_alpha = map(x->borne(x,-1.0, 1.0),genes[1:arch[1]*4])
    matalpha = reshape(genes_alpha, (arch[1],4))
    genes_w = genes[arch[1]*4+1:end]
    params = YAML.load(open("Julia/Cartpole/Teacher/cfg_default.yml"))

    n_stud = Network(arch,params)
    n_stud.param["matalpha"] = matalpha
    n_teach = deepcopy(n_stud)


    # copy the weight of best indiv
    exc, inh = ei_idx(arch)
    set_weight!(n_teach,genes_w, exc, inh)
    return n_teach, n_stud
end

# # %% tests
#
#
# filepath = "Julia/CartPole/LIF_base/results/results_EI888.jld"
#
# n_teach, n_stud = init_teacher_student(filepath, 19)
#
# @show n_teach.param["matalpha"]

# # %% tests viz
#
# weight_viz(n_stud)
# weight_viz(n_teach)
