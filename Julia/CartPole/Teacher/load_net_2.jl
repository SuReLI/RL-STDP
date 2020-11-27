# %% imports
using Cambrian
using EvolutionaryStrategies
using JLD
using PyCall
using Statistics
gym = pyimport("gym")
include("../../Modules/Networks/net_arch.jl")
include("../../Modules/Poisson/PoissonStateSpike.jl")
include("../../Actors/cartpole_fitness.jl")

function create_net(elites::Array{AbstractESIndividual,1}, indiv::Int64)
    params = YAML.load(open("Julia/Actors/cartpole_cfglif.yml"))
    arch = params["arch"]

    genes = elites[indiv].genes

    # Input linear combination
    genes_alpha = map(x->borne(x,-1.0, 1.0),genes[1:arch[1]*4])
    matalpha = reshape(genes_alpha, (arch[1],4))

    # Weights of the network
    genes_w = genes[arch[1]*4+1:end]

    # Initialize network
    n = Network(arch, params)
    exc, inh = ei_idx(arch)
    set_weight!(n,genes_w, exc, inh)

    return n, matalpha
end

function init_teacher_student(filepath::String, indiv::Int64)
    elites = load(filepath, "elites")
    n_teach, matalpha = create_net(elites, indiv)
    n_stud = deepcopy(n_teach)
    n_stud.s = Weights(n_stud.arch, n_stud.param)

    return n_teach, n_stud, matalpha
end
