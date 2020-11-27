using JLD


function save_param(best_fit::Array{Float64}, best::Array{Array{Float64}}, tunekeys::Array{String})
    L = length(tunekeys)
    ar = [Float64[] for i in 1:L ]
    results = Dict{String,Any}(tunekeys .=> ar)
    results["weights"] = Array{Float64}[]
    results["fit"] = Float64[]
    for i in 1:length(best)
        genes_p = best[i][1:L]
        genes_w = best[i][L+1:end]
        push!(results["weights"], genes_w)
        push!(results["fit"], best_fit[i])
        for j in 1:L
            push!(results[tunekeys[j]], genes_p[j])
        end
    end
    save("results.jld", "results", results)
end

function save_param(best_fit::Array{Float64}, best::Array{Array{Float64}})
    results = Dict()
    results["weights"] = Array{Float64}[]
    results["fit"] = Float64[]
    for i in 1:length(best)
        genes_w = best[i]
        push!(results["weights"], genes_w)
        push!(results["fit"], best_fit[i])
    end
    save("results.jld", "results", results)
end

function load_param(filepath::String)
    results = load(filepath, "results")
    parameters = Dict()
    L = length(results["fit"])
    for i in 1:L
        for key in collect(keys(results))
            if key != "weights"
                parameters[key] = results[key][i]
            end
        end
        #println("Set $(i) : ", parameters)
    end
    return results
end

# %% test

#results = load_param("Julia/CartPole/CMAES/results_decision2.jld")
