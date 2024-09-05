st = time()

using Base.Threads
# using LinearAlgebra
# using InteractiveUtils
# using BenchmarkTools
import JSON
using UnicodePlots
using ErrorTypes
# using HypothesisTests
using Revise
using Dates
using Infiltrator
using Debugger
using UTCGP
using UTCGP: jsonTracker, save_json_tracker, repeatJsonTracker, jsonTestTracker
using UTCGP: SN_writer, sn_strictphenotype_hasher
using UTCGP: get_image2D_factory_bundles, SImageND, DataFrames
# import SearchNetworks as sn
using DataStructures
using UUIDs
# using Images, FileIO
using ImageCore
# using ImageBinarization
# using CSV
# using DataFrames
using Statistics
using StatsBase
using UTCGP: SImage2D, BatchEndpoint
using Random
# using ImageView
using StatsBase: kurtosis, variation, sample
# using MLDatasets: CIFAR10
using UTCGP: AbstractCallable, Population, RunConfGA, PopulationPrograms, IndividualPrograms, get_float_bundles, replace_shared_inputs!, evaluate_individual_programs, reset_programs!
# using JuliaInterpreter
# using Flux
using Logging
# using HDF5
# using DataFlowTasks
# using ArgParse
# import PNGFiles
# using ThreadPinning
using ImageView
using ArcadeLearningEnvironment
using MAGE_ATARI
using IOCapture

@inline function relu(x)
    x > 0 ? x : 0
end

dir = @__DIR__
pwd_dir = pwd()

file = @__FILE__
home = dirname(dirname(file))

PROB_ACTION = true
FIX_STRUCTURE = true
# GAME 
GAME = "pong"
_g = Game(GAME, 1)
DOWNSCALE = true
GRAYSCALE = true
s = get_state_buffer(_g, GRAYSCALE)
o = get_observation_buffer(_g, GRAYSCALE, DOWNSCALE)
IMG_SIZE = o[1] |> size
println("SCALE ($DOWNSCALE) SIZE : $(IMG_SIZE)")

# o = get_observation_buffer(_g, GRAYSCALE, true)
# IMG_SIZE_ = o[1] |> size
# println("DOWNSCALED SIZE : $(IMG_SIZE_ )")


IMAGE_TYPE = SImage2D{IMG_SIZE[1],IMG_SIZE[2],N0f8,Matrix{N0f8}}
TRUE_ACTIONS = getMinimalActionSet(_g.ale)

ATARI_LOCK = ReentrantLock()

abstract type AbstractProb end
struct prob <: AbstractProb end
struct notprob <: AbstractProb end

function pong_action_mapping(outputs::Vector, p::Type{notprob})
    global ACTIONS
    # if outputs[1] > 0.1
    #     return 4
    # elseif outputs[1] < -0.1
    #     return 3
    # end
    # 2
    ACTIONS[argmax(outputs)]
end
function pong_action_mapping(outputs::Vector, p::Type{prob}, mt)
    global ACTIONS
    os = relu.(outputs)
    w = Weights(os)
    action = sample(mt, ACTIONS, w)
    action
end

action_mapping_dict = Dict(
    "pong" => [3, pong_action_mapping]
)

NACTIONS = action_mapping_dict[GAME][1]
ACTIONS = TRUE_ACTIONS[2:NACTIONS+1]
ACTION_MAPPER = action_mapping_dict[GAME][2]
@show ACTIONS
# SEED 
seed_ = 1
@warn "Seed : $seed_"
Random.seed!(seed_)

# Initialize the project
disable_logging(Logging.Debug)

# HASH 
hash = UUIDs.uuid4() |> string

function evaluate_individual_programs!(
    ind_prog::IndividualPrograms,
    buffer,
    model_arch::modelArchitecture,
    meta_library::MetaLibrary
)
    in_types = model_arch.inputs_types_idx
    UTCGP.reset_programs!(ind_prog)
    input_nodes = [
        InputNode(value, pos, pos, in_types[pos]) for
        (pos, value) in enumerate(buffer)
    ]
    replace_shared_inputs!(ind_prog, input_nodes)
    outputs = UTCGP.evaluate_individual_programs(
        ind_prog,
        model_arch.chromosomes_types,
        meta_library,
    )
    # @show outputs
    outputs
end

# PARAMS --- --- 
function atari_fitness(ind::IndividualPrograms, seed, model_arch::modelArchitecture, meta_library::MetaLibrary)
    global GAME
    game = nothing
    # IOCapture.capture() do
    game = Game(GAME, 1, lck=ATARI_LOCK)
    # end
    Random.seed!(seed)
    mt = MersenneTwister(seed)
    #game = Game(rom, seed, lck=lck)
    MAGE_ATARI.reset!(game)
    # reset!(reducer) # zero buffers
    grayscale = true
    downscale = true
    max_frames = 2000
    stickiness = 0.25
    s = get_state_buffer(game, grayscale)
    o = get_observation_buffer(game, grayscale, downscale)
    reward = 0.0
    frames = 0
    prev_action = Int32(0)
    q = Queue{IMAGE_TYPE}()
    while ~game_over(game.ale)
        # if rand(mt) > stickiness || frames == 0
        if rand(mt) > stickiness || frames == 0
            get_state!(s, game, grayscale)
            get_observation!(o, s, game, grayscale, downscale)
            cur_frame = SImageND(N0f8.(o[1] ./ 256.0))
            # if rand() > 0.999
            # imshow(cur_frame)
            # end
            if isempty(q)
                enqueue!(q, cur_frame)
                enqueue!(q, cur_frame)
                enqueue!(q, cur_frame)
                enqueue!(q, cur_frame)
            end
            dequeue!(q) # removes the first
            enqueue!(q, cur_frame) # adds to last
            v = [i for i in q]
            o_copy = [v[1],
                v[2],
                v[3],
                v[4],
                0.1, -0.1, 2.0, -2.0]
            outputs = evaluate_individual_programs!(ind, o_copy, model_arch, meta_library)
            # @show outputs

            if PROB_ACTION
                action = ACTION_MAPPER(outputs, prob, mt)
            else
                action = ACTION_MAPPER(outputs, notprob)
            end
            # @show action
            # if rand(mt) < 0.01
            #     action = Int32(1)
            # end
        else
            action = prev_action
        end
        reward += act(game.ale, action)
        frames += 1
        prev_action = action
        if frames > max_frames
            @info "Game finished because of max_frames"
            break
        end
    end
    # close!(game)

    # if reward < 0.0
    #     r = reward
    # else
    #     r = reward * -1.0
    # end
    @show reward
    r = reward * -1.0
    # @show r
    r
end

struct AtariEndpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Float64}
    function AtariEndpoint(
        pop::PopulationPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary
    )
        n = length(pop)
        nt = nthreads()
        indices = collect(1:n)
        BS = ceil(Int, n / nt)
        pop_res = Vector{Float64}(undef, n)
        tasks = []
        for ith_x in Iterators.partition(indices, BS)
            t = @spawn begin
                @info "Spawn the task of $ith_x to thread $(threadid())"
                for i in ith_x
                    pop_res[i] = atari_fitness(pop[i], 1, model_arch, meta_library)

                end
            end
            push!(tasks, t)
        end
        results = fetch.(tasks)
        @show results
        @show pop_res
        return new(pop_res)
    end
end


### RUN CONF ###

n_elite = 5
n_new = 20
gens = 100
tour_size = 3
mut_rate = 1.1

run_conf = RunConfGA(
    n_elite, n_new, tour_size, mut_rate, 0.1, gens
)

# Bundles Integer
fallback() = SImageND(ones(N0f8, (IMG_SIZE[1], IMG_SIZE[2])))
image2D = UTCGP.get_image2D_factory_bundles_atari()
for factory_bundle in image2D
    for (i, wrapper) in enumerate(factory_bundle)
        try
            fn = wrapper.fn(IMAGE_TYPE) # specialize
            wrapper.fallback = fallback
            # create a new wrapper in order to change the type
            factory_bundle.functions[i] =
                UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)

        catch
        end
    end
end

float_bundles = UTCGP.get_float_bundles_atari()
# vector_float_bundles = UTCGP.get_listfloat_bundles()
# int_bundles = UTCGP.get_integer_bundles()

# Libraries
lib_image2D = Library(image2D)
lib_float = Library(float_bundles)
# lib_int = Library(int_bundles)
# lib_vecfloat = Library(vector_float_bundles)

# MetaLibrarylibfloat# ml = MetaLibrary([lib_image2D, lib_float, lib_vecfloat, lib_int])
ml = MetaLibrary([lib_image2D, lib_float])

offset_by = 8 # 0.1 0.2 1. -1.

### Model Architecture ###
model_arch = modelArchitecture(
    [IMAGE_TYPE, IMAGE_TYPE, IMAGE_TYPE, IMAGE_TYPE, Float64, Float64, Float64, Float64],
    [1, 1, 1, 1, 2, 2, 2, 2],
    # [IMAGE_TYPE, Float64, Vector{Float64}, Int], # genome
    [IMAGE_TYPE, Float64], # genome
    [Float64 for i in 1:NACTIONS], # outputs
    [2 for i in 1:NACTIONS]
)

### Node Config ###
N_nodes = 20
node_config = nodeConfig(N_nodes, 1, 3, offset_by)

### Make UT GENOME ###
shared_inputs, ut_genome = make_evolvable_utgenome(
    model_arch, ml, node_config
)
initialize_genome!(ut_genome)
correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
function fix_all_output_nodes!(ut_genome::UTGenome)
    for (ith_out_node, output_node) in enumerate(ut_genome.output_nodes)
        to_node = output_node[2].highest_bound + 1 - ith_out_node
        set_node_element_value!(output_node[2],
            to_node)
        set_node_freeze_state(output_node[2])
        set_node_freeze_state(output_node[1])
        set_node_freeze_state(output_node[3])
        println("Output node at $ith_out_node: $(output_node.id) pointing to $to_node")
        println("Output Node material : $(node_to_vector(output_node))")
    end
end
fix_all_output_nodes!(ut_genome)


###########
# METRICS #
###########

# struct jsonTrackerGA <: AbstractCallable
#     tracker::jsonTracker
#     acc_callback::acc_callback
#     label::String
#     test_lossses::Union{Nothing,Vector}
# end
# f = open(home * "/metrics_irace_ga_noracing_radiomics0/" * string(hash) * ".json", "a", lock=true)
# metric_tracker = jsonTracker(h_params, f)

# TRACK NLL & ACCURACY  => ReRuns for one individual
# function (jtga::jsonTrackerGA)(
#     ind_performances,
#     population,
#     iteration,
#     run_config,
#     model_architecture,
#     node_config,
#     meta_library,
#     shared_inputs,
#     population_programs,
#     best_losses,
#     best_programs,
#     elite_idx,
#     batch
# )
#     losses_acc, losses_error, losses_nll, losses_error, final_acc, final_error, final_nll, final_loss = jtga.acc_callback(ind_performances,
#         population,
#         iteration,
#         run_config,
#         model_architecture,
#         node_config,
#         meta_library,
#         shared_inputs,
#         population_programs,
#         best_losses,
#         best_programs,
#         elite_idx,
#         batch)
#     @warn "JTT $(jtga.label) Fitness : $final_acc"
#     s = Dict("data" => jtga.label, "iteration" => iteration,
#         "accuracy" => final_acc, "error" => final_error, "nll" => final_nll, "loss" => final_loss)
#     if !isnothing(jtga.test_lossses)
#         push!(jtga.test_lossses, final_acc)
#     end
#     write(jtga.tracker.file, JSON.json(s), "\n")
# end

if FIX_STRUCTURE
    N_IMG = 10

end
##########
#   FIT  #
##########
budget_stop = UTCGP.eval_budget_early_stop(10_000)

function fit_ga_atari(
    shared_inputs::SharedInput,
    genome::UTGenome,
    model_architecture::modelArchitecture,
    node_config::nodeConfig,
    run_config::RunConfGA,
    meta_library::MetaLibrary,
    # Callbacks before training
    pre_callbacks::UTCGP.Optional_FN,
    # Callbacks before step (before looping through data)
    population_callbacks::UTCGP.Mandatory_FN,
    mutation_callbacks::UTCGP.Mandatory_FN,
    output_mutation_callbacks::UTCGP.Mandatory_FN,
    decoding_callbacks::UTCGP.Mandatory_FN,
    # Callbacks per step (while looping through data)
    endpoint_callback::Type{<:BatchEndpoint},
    final_step_callbacks::UTCGP.Optional_FN,
    # Callbacks after step ::
    elite_selection_callbacks::UTCGP.Mandatory_FN,
    epoch_callbacks::UTCGP.Optional_FN,
    early_stop_callbacks::UTCGP.Optional_FN,
    last_callback::UTCGP.Optional_FN,
) # Tuple{UTGenome, IndividualPrograms, GenerationLossTracker}::

    local early_stop, best_programs, elite_idx, population, ind_performances, =
        UTCGP._ga_init_params(genome, run_config)

    # PRE CALLBACKS
    UTCGP._make_pre_callbacks_calls(pre_callbacks)
    M_gen_loss_tracker = UTCGP.GenerationLossTracker()

    for iteration = 1:run_config.generations
        early_stop ? break : nothing
        @warn "Iteration : $iteration"
        M_individual_loss_tracker = IndividualLossTracker() # size of []
        # Population
        ga_pop_args = UTCGP.GA_POP_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            ind_performances,
        )
        population, time_pop =
            @unwrap_or UTCGP._make_ga_population(ga_pop_args, population_callbacks) throw(
                "Could not unwrap make_population",
            )

        # Program mutations ---
        ga_mutation_args = GA_MUTATION_ARGS(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
        )
        population, time_mut =
            @unwrap_or UTCGP._make_ga_mutations!(ga_mutation_args, mutation_callbacks) throw(
                "Could not unwrap make_ga_mutations",
            )

        # Output mutations ---
        population, time_out_mut = @unwrap_or UTCGP._make_ga_output_mutations!(
            ga_mutation_args,
            output_mutation_callbacks,
        ) throw("Could not unwrap make_ga_output_mutations")

        # Genotype to Phenotype mapping --- 
        population_programs, time_pop_prog = UTCGP._make_decoding(
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            shared_inputs,
            decoding_callbacks,
        )

        @warn "Graphs evals"
        fitness = endpoint_callback(
            population_programs, model_arch, meta_library
        )
        fitness_values = get_endpoint_results(fitness)
        UTCGP.add_pop_loss_to_ind_tracker!(M_individual_loss_tracker, fitness_values)  # appends the loss for the ith x sample to the

        # Resetting the population (removes node values)
        [reset_genome!(g) for g in population]

        # final step call...
        if !isnothing(final_step_callbacks)
            for final_step_callback in final_step_callbacks
                UTCGP.get_fn_from_symbol(final_step_callback)()
            end
        end

        # DUPLICATES # TODO

        # Selection
        ind_performances = UTCGP.resolve_ind_loss_tracker(M_individual_loss_tracker)

        # Elite selection callbacks
        @warn "Selection"
        ga_selection_args = GA_SELECTION_ARGS(
            ind_performances,
            population,
            iteration,
            run_config,
            model_architecture,
            node_config,
            meta_library,
            population_programs,
        )
        elite_idx, time_elite = @unwrap_or UTCGP._make_ga_elite_selection(
            ga_selection_args,
            elite_selection_callbacks,
        ) throw("Could not unwrap make_ga_selection")

        elite_fitnesses = ind_performances[elite_idx]
        elite_best_fitness = minimum(skipmissing(elite_fitnesses))
        elite_best_fitness_idx = elite_fitnesses[1]
        elite_avg_fitness = mean(skipmissing(elite_fitnesses))
        elite_std_fitness = std(filter(!isnan, ind_performances))
        best_programs = population_programs[elite_idx]

        reset_programs!.(best_programs)
        # empty!(shared_inputs.inputs)
        for p in best_programs
            println("Best Prog")
            replace_shared_inputs!(p, [0 for i in 1:offset_by])
            println(p)
        end

        # genome = deepcopy(population[elite_idx])
        try
            histogram(ind_performances) |> println
        catch e
            @error "Could not drawn histogram"
        end

        # Subset Based on Elite IDX---
        old_pop = deepcopy(population.pop[elite_idx])
        empty!(population.pop)
        push!(population.pop, old_pop...)
        ind_performances = ind_performances[elite_idx]

        # EPOCH CALLBACK
        batch = []
        if !isnothing(epoch_callbacks)
            UTCGP._make_epoch_callbacks_calls(
                ind_performances,
                population,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                population_programs,
                elite_fitnesses,
                best_programs,
                elite_idx,
                view(batch, :),
                epoch_callbacks,)
        end
        # MU CALLBACKS # TODO

        # LAMBDA CALLBACKS # TODO

        # GENOME SIZE CALLBACKS # TODO

        # store iteration loss/fitness
        UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, iteration, elite_best_fitness)
        println(
            "Iteration $iteration. 
            Best fitness: $(round(elite_best_fitness, digits = 10)) at index $elite_best_fitness_idx 
            Elite mean fitness : $(round(elite_avg_fitness, digits = 10)). Std: $(round(elite_std_fitness)) at indices : $(elite_idx)",
        )

        # EARLY STOP CALLBACK # TODO
        if !isnothing(early_stop_callbacks) && length(early_stop_callbacks) != 0
            early_stop_args = UTCGP.GA_EARLYSTOP_ARGS(
                M_gen_loss_tracker,
                M_individual_loss_tracker,
                ind_performances,
                population,
                iteration,
                run_config,
                model_architecture,
                node_config,
                meta_library,
                shared_inputs,
                population_programs,
                elite_fitnesses,
                best_programs,
                elite_idx,
            )

            early_stop =
                UTCGP._make_ga_early_stop_callbacks_calls(early_stop_args, early_stop_callbacks) # true if any
        end

        if early_stop
            g = run_config.generations
            @warn "Early returning at iteration : $iteration from $g total iterations"
            if !isnothing(last_callback)
                last_callback(
                    ind_performances,
                    population,
                    iteration,
                    run_config,
                    model_architecture,
                    node_config,
                    meta_library,
                    population_programs,
                    elite_fitnesses,
                    best_programs,
                    elite_idx,
                )
            end
            # UTCGP.show_program(program)
            return tuple(genome, best_programs, M_gen_loss_tracker)
        end

    end
    return (genome, best_programs, M_gen_loss_tracker)
end

best_genome, best_programs, gen_trakcer = fit_ga_atari(
    shared_inputs,
    ut_genome,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    (:ga_population_callback,),
    (:ga_numbered_new_material_mutation_callback,),
    (:ga_output_mutation_callback,),
    (:default_decoding_callback,),
    # Endpoints
    AtariEndpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    (:ga_elite_selection_callback,),
    # Epoch Callback
    nothing, # (train_tracker, test_tracker), #[metric_tracker, test_tracker, sn_writer_callback],
    # Final callbacks ?
    (budget_stop,), #(:default_early_stop_callback,), # 
    nothing #repeat_metric_tracker # .. 
)

