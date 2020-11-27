# %% imports

include("../../Modules/save_param.jl")
include("critic_sim.jl")
using YAML


# %% get the results

results = load_param("Julia/Cartpole/Criticv2/results002.jld")
function results_to_genes(results::Dict{String,Any}, indiv::Int64)
    genes_hyp = Float64[]
    hyp = String[]
    dic_keys = collect(keys(results))
    filter!(x-> x!="weights", dic_keys)
    filter!(x-> x!="fit", dic_keys)
    for key in dic_keys
        push!(hyp, key)
        push!(genes_hyp, results[key][indiv])
    end
    genes_alpha = map(x->borne(x, 0.0, 10.0),results["weights"][indiv])
    matalpha = reshape(genes_alpha, (4,4))
    return matalpha, genes_hyp, hyp
end


# %% create the networks

indiv = 20


#actor
filepath = "Julia/CartPole/LIF_base/results/results_EI888.jld"
n_actor, _ = init_teacher_student(filepath, 19)

#critic default
arch_critic = [8,8,2]
params = YAML.load(open("Julia/Cartpole/Criticv2/cfg_default_critic.yml"))
n_critic = Network(arch_critic, params)

#tune the critic with results
matalpha, genes_hyp,hyp = results_to_genes(results, indiv)
assert_genes!(n_critic,genes_hyp, hyp)

td_errors = Float64[]

for epoch in 1:20
    @show epoch
    td_error, td_errors = play_episode(n_actor, n_critic, matalpha)
    @show maximum(td_errors)
    display(plot(td_errors))
end
