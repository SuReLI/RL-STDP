# %% imports

using Cambrian
using EvolutionaryStrategies

include("../Actors/cartpole_tryout.jl")



# %% build a class with all the matrices, the fitness associated ( mean reward from cartpole tryout )

mutable struct MatrixClust

    fitness::Array{Float64}
    matrices::Array{Array{Float64,2}}
    s_to_m::Array{Int64}
    segs::Array{Array{Float64}}

    function MatrixClust(elites::Array{AbstractESIndividual,1}, mean_score::Array{Float64})
        fitness_tmp = [elites[i].fitness[1] for i in 1:length(elites)]
        ar = [map(x->borne(x,-1.0, 1.0),elites[i].genes[1:8*4]) for i in 1:length(elites)]
        matrices_tmp = [reshape(ar[i], (8,4)) for i in 1:length(ar)]

        idx = findall(x -> x > 0., mean_score .- 195.)
        fitness = fitness_tmp[idx]
        matrices = matrices_tmp[idx]
        segs = Array{Float64}[]
        s_to_m = Int64[]
        for i in eachindex(matrices)
            m = matrices[i]
            for j in 1:size(m)[1]
                push!(segs, m[j,:])
                push!(s_to_m, i)
            end
        end
        new(fitness, matrices, s_to_m, segs)
    end

    function MatrixClust()
        fitness = Float64[]
        matrices = Array{Float64,2}[]
        s_to_m = Int64[]
        segs = Array{Float64}[]
        new(fitness, matrices, s_to_m, segs)
    end
end

function join_matclust(mc1::MatrixClust, mc2::MatrixClust)
    l = length(mc1.matrices)
    mc1.fitness = [mc1.fitness ; mc2.fitness]
    mc1.matrices = [mc1.matrices ; mc2.matrices]
    mc1.s_to_m = [mc1.s_to_m ; mc2.s_to_m .+ l]
    mc1.segs = [mc1.segs ; mc2.segs]
    return mc1
end

# %% main function


# Builds the class MatrixClust with all the best individual from cartpole sNES
# saves it as mc.jl
function build_matclust(n::Int64)
    mc = MatrixClust()
    for i in 1:n
        elites = load("Julia/Actors/cartpoleElites/sNESelites_cartpole_$(i).jld", "elites")
        means = test_random(elites)
        mc_tmp = MatrixClust(elites, means)
        mc = join_matclust(mc, mc_tmp)
    end
    save("Julia/MatrixCluster/mc.jld", "mc", mc)
end
