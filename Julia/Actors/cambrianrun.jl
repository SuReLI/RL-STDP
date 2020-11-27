# %% imports


import Cambrian: run!
export run!

using Cambrian
using JLD


# %% save generation's best individual
function step!(e::AbstractEvolution)
    e.gen += 1
    if e.gen > 1
        populate(e)
    end
    evaluate(e)
    generation(e)
    if ((e.config.log_gen > 0) && mod(e.gen, e.config.log_gen) == 0)
        log_gen(e)
    end
    if ((e.config.save_gen > 0) && mod(e.gen, e.config.save_gen) == 0)
        save_gen(e)
    end
end

#if sNES
# function run!(e::AbstractEvolution, opt::Int64)
#     for i in (e.gen+1):e.config.n_gen
#         step!(e)
#         for idx in eachindex(e.population)
#             if e.population[idx].fitness[1] > 195.
#                 f = jldopen("Julia/Actors/topgensNES.jld", "r+")
#                 write(f, "$(opt)_run", e.gen)
#                 close(f)
#                 return 0
#             end
#         end
#     end
# end

#if xNES
function run!(e::AbstractEvolution, opt::Int64)
    for i in (e.gen+1):e.config.n_gen
        step!(e)
        for idx in eachindex(e.population)
            if e.population[idx].fitness[1] > 195.
                f = jldopen("Julia/Actors/topgenxNES.jld", "r+")
                write(f, "$(opt)_run", e.gen)
                close(f)
                return 0
            end
        end
    end
end
