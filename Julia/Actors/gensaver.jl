# %% imports

import Cambrian: save_gen
export save_gen

using Cambrian
using JLD

# %% resuts class

mutable struct Results

    n_indiv::Int64
    kwargs::Array{String}
    values::Dict{String,Any}

    function Results(n_indiv::Int64, kwargs::Array{String})
        ar = [Any[] for i in 1:length(kwargs)]
        values = Dict(kwargs .=> ar)
        new(n_indiv, kwargs, values)
    end
end

function get_indiv(res::Results, ind::Int64)
    indiv_param = Dict(kwarg => res.values[kwarg][ind] for kwarg in res.kwargs)
    return indiv_param
end

function update_results!(res::Results, fit::Float64, genes::Array{Float64})
    if length(res.values["fit"]) < res.n_indiv
        push!(res.values["fit"], fit)
        push!(res.values["genes"], genes)
        idx = sortperm(res.values["fit"])
        res.values["fit"] = res.values["fit"][idx]
        res.values["genes"] = res.values["genes"][idx]
    else
        if fit > res.values["fit"][1]
            res.values["fit"][1] = fit
            res.values["genes"][1] = genes
            idx = sortperm(res.values["fit"])
            res.values["fit"] = res.values["fit"][idx]
            res.values["genes"] = res.values["genes"][idx]
            found = findall(x->x==genes,res.values["genes"])
            if length(found)==0
                println("GENES NOT FOUND")
            else
                println("genes found")
            end
        end
    end
end


# %% save generation's best individual

function save_gen(e::AbstractEvolution, res::Results = results)
    # save best element of each gen
    fits = Float64[]
    for i in eachindex(e.population)
        push!(fits, e.population[i].fitness[1])
    end
    bestind = argmax(fits)
    bestfit = fits[bestind]
    best_genes = e.population[bestind].genes
    update_results!(res, fits[bestind], best_genes)
    save("Julia/Actors/results_sNES_cartpole.jld", "results", res)
end
