st = time()

using Logging
using Base.Threads
import JSON
using UnicodePlots
using ErrorTypes
using Revise
using Dates
using Infiltrator
using Debugger
using UTCGP
using UTCGP: jsonTracker, save_json_tracker, repeatJsonTracker, jsonTestTracker
using UTCGP: SN_writer, sn_strictphenotype_hasher
using UTCGP: get_image2D_factory_bundles, SImageND, DataFrames
using DataStructures
using UUIDs
using ImageCore
using Statistics
using StatsBase
using UTCGP: SImage2D, BatchEndpoint
using StatsBase: kurtosis, variation, sample
using UTCGP: AbstractCallable, Population, RunConfNSGA2, PopulationPrograms, IndividualPrograms, get_float_bundles, replace_shared_inputs!, evaluate_individual_programs, reset_programs!
# using ImageView
using ArcadeLearningEnvironment
using MAGE_ATARI
using Random
using IOCapture
using LinearAlgebra

@inline function relu(x)
    x > 0 ? x : 0
end

dir = @__DIR__
pwd_dir = pwd()

file = @__FILE__
home = dirname(dirname(file))

REDUCED_ACTIONS = false
FIXED_THRESHOLDS = false
THRESHOLDS = [.1, .1]

PROB_ACTION = true
GAME = "pong"
_g = AtariEnv(GAME, 1)

# these two params do not count now
DOWNSCALE = false
GRAYSCALE = true

MAGE_ATARI.update_state(_g)
MAGE_ATARI.update_screen(_g)
_example = SImageND(N0f8.(Gray.(_g.screen)))
w, h = size(_g)
# IMAGE_TYPE = SImage2D{w,h,N0f8,Matrix{N0f8}}
IMAGE_TYPE = typeof(_example)
TRUE_ACTIONS = MAGE_ATARI.actions(_g, _g.state)
ATARI_LOCK = ReentrantLock()
IMG_SIZE = _example |> size

abstract type AbstractProb end
struct prob <: AbstractProb end
struct notprob <: AbstractProb end

function protected_norm(x::Vector{Float64})
    nrm = norm(x)
    if nrm > 0
        return nrm
    else
        return 1.
    end
end

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
    "pong" => [3, pong_action_mapping],
    "ms_pacman" => [5, pong_action_mapping]
)

if REDUCED_ACTIONS
    NACTIONS = MAGE_ATARI.atari_outputs_size(TRUE_ACTIONS)
    ACTIONS = TRUE_ACTIONS
    act_map = MAGE_ATARI.atari_outputs_mapping_fn(TRUE_ACTIONS)
    if FIXED_THRESHOLDS
        ACTION_MAPPER = o -> act_map(o, THRESHOLDS)
    else
        ACTION_MAPPER = act_map
    end
else
    NACTIONS = action_mapping_dict[GAME][1]
    if GAME == "pong"
        println("reducing action set")
        ACTIONS = TRUE_ACTIONS[2:NACTIONS+1]
    else    
        ACTIONS = TRUE_ACTIONS
    end
    ACTION_MAPPER = action_mapping_dict[GAME][2]
end

# @show ACTIONS
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
    # TODO remove this line
    seed = 1
    global GAME
    game = nothing
    # IOCapture.capture() do
    game = AtariEnv(GAME, seed, ATARI_LOCK)
    # end
    Random.seed!(seed)
    mt = MersenneTwister(seed)
    #game = Game(rom, seed, lck=lck)
    MAGE_ATARI.reset!(game)
    # reset!(reducer) # zero buffers
    # TODO just for debugging purposes, reset to 18_000
    max_frames = 10_000
    stickiness = 0.25
    reward = 0.0
    frames = 0
    prev_action = Int32(0)
    actions_counter = Dict(
        [action => 0 for action in ACTIONS]...
    )
    action_changes = 0
    output_sequence = Vector{Vector{Float64}}(undef, 0)
    q = Queue{IMAGE_TYPE}()
    while ~game_over(game.ale)
        if rand(mt) > stickiness || frames == 0
            MAGE_ATARI.update_screen(game)
            current_gray_screen = N0f8.(Gray.(game.screen))
            cur_frame = SImageND(current_gray_screen)
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
            push!(output_sequence, outputs)
            
            # action = ACTION_MAPPER(outputs)
            if PROB_ACTION
                action = ACTION_MAPPER(outputs, prob, mt)
            else
                action = ACTION_MAPPER(outputs, notprob)
            end

        else
            action = prev_action
        end
        # reward += act(game.ale, action)
        r, s = MAGE_ATARI.step!(game, game.state, action)
        if action != prev_action
            action_changes += 1
        end
        actions_counter[action] += 1
        reward += r
        frames += 1
        prev_action = action
        if frames > max_frames
            # @info "Game finished because of max_frames"
            break
        end
    end
    MAGE_ATARI.close(game)
    # calculate the descriptors
    tot = sum(values(actions_counter))
    action_tot = sum([actions_counter[a] for a in ACTIONS[2:end]])
    activity_descriptor = action_tot / tot
    action_changes_descriptor = action_changes / tot
    first_objective = reward * -1.0

    # TODO minimize the inverse of the average variance of image to float fns in the graph (variance over the episode = game) 
    # for now we maximize the variance of each output over the episode
    # note: we first divide the vector by its norm to ensure coherence across the values
    transposed_output_signals = [[v[i] for v in output_sequence] for i=1:length(output_sequence[1])]
    variances = [var(v/protected_norm(v)) for v in transposed_output_signals]
    second_objective = - mean(variances)
    return [first_objective, second_objective]
end

struct AtariNSGA2Endpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Vector{Float64}}
    function AtariNSGA2Endpoint(
        pop::PopulationPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary,
        generation::Int
    )
        n = length(pop)
        nt = nthreads()
        indices = collect(1:n)
        BS = ceil(Int, n / nt)
        pop_fits = Vector{Vector{Float64}}(undef, n)
        tasks = []
        for ith_x in Iterators.partition(indices, BS)
            t = @spawn begin
                @info "Spawn the task of $ith_x to thread $(threadid())"
                for i in ith_x
                    f = atari_fitness(deepcopy(pop[i]), generation, model_arch, meta_library)
                    # @show f
                    # @show d
                    pop_fits[i] = f
                end
            end
            push!(tasks, t)
        end
        results = fetch.(tasks)
        # @show results
        # @show pop_res
        return new(pop_fits)
    end
end

function UTCGP.get_endpoint_results(e::AtariNSGA2Endpoint)
    return e.fitness_results
end


### RUN CONF ###
pop_size = 48
tour_size = 3
gens = 4_000
mut_rate = 3.1
output_mut_rate = 0.1

run_conf = UTCGP.RunConfNSGA2(
    pop_size, tour_size, mut_rate, output_mut_rate, gens
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

offset_by = 8 # 4 inputs and 4 constants - 0.1 0.2 1. -1.

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
N_nodes = 15
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
        # println("Output node at $ith_out_node: $(output_node.id) pointing to $to_node")
        # println("Output Node material : $(node_to_vector(output_node))")
    end
end
fix_all_output_nodes!(ut_genome)


####################
#  EPOCH CALLBACK  #
####################

# DIRS to save metrics & models
scripts = "scripts"
folder = joinpath(scripts, GAME)
isdir(folder) || mkdir(folder)
folder = joinpath(folder, hash)
isdir(folder) || mkdir(folder)

# TODO possibly need to add some params here
h_params = Dict(
    "connection_temperature" => node_config.connection_temperature,
    "n_nodes" => node_config.n_nodes,
    "generations" => run_conf.generations,
    # "budget" => "budget"],
    "mutation_rate" => run_conf.mutation_rate,
    "output_mutation_rate" => run_conf.output_mutation_rate,
    "Correction" => "true",
    "output_node" => "fixed",
    # "n_train" => Parsed_args["n_samples"],
    # "n_test" => length(testx),
    # "seed" => Parsed_args["seed"],
    "mutation" => "ga_numbered_new_material_mutation_callback",
    # "n_elite" => Parsed_args["n_elite"],
    # "n_new" => Parsed_args["n_new"],
    # "tour_size" => Parsed_args["tour_size"],
    # "acc_weight" => ACC_WEIGHT,
    # "instance" => Parsed_args["instance"],
)

# TODO check these trackers
metrics_path = joinpath(folder, "metrics.json")
struct jsonTrackerNSGA2 <: UTCGP.AbstractCallable
    tracker::UTCGP.jsonTracker
    label::String
    test_losses::Vector
end

f = open(metrics_path, "a", lock=true)
metric_tracker = UTCGP.jsonTracker(h_params, f)
atari_tracker = jsonTrackerNSGA2(metric_tracker, "Test", [])

function (jtga::jsonTrackerNSGA2)(ind_performances,
    rep,
    population,
    generation,
    run_config,
    model_architecture,
    node_config,
    meta_library,
    shared_inputs,
    programs,
    best_loss,
    best_program,
    elite_idx,
    Batch)
    best_f = UTCGP.best_fitness(rep)
    @warn "JTT $(jtga.label) Fitness : $best_f"
    s = Dict("data" => jtga.label, "iteration" => generation,
        "coverage" => UTCGP.coverage(rep), "best_fitness" => best_f)

    push!(jtga.test_losses, best_f)
    write(jtga.tracker.file, JSON.json(s), "\n")
    flush(jtga.tracker.file)
end

# CHECKPOINT
function save_payload(best_genome,
    best_programs, gen_tracker,
    shared_inputs, ml, run_conf, node_config,
    name::String="best_genome.pickle")
    payload = Dict()
    payload["best_genome"] = deepcopy(best_genome)
    payload["best_program"] = deepcopy(best_programs)
    payload["gen_tracker"] = deepcopy(gen_tracker)
    payload["shared_inputs"] = deepcopy(shared_inputs)
    payload["ml"] = deepcopy(ml)
    payload["run_conf"] = deepcopy(run_conf)
    payload["node_config"] = deepcopy(node_config)

    genome_path = joinpath(folder, name)
    open(genome_path, "w") do io
        @info "Writing payload to $genome_path"
        write(io, UTCGP.general_serializer(deepcopy(payload)))
    end
end
struct checkpoint <: UTCGP.AbstractCallable
    every::Int
end

function (c::checkpoint)(ind_performances,
    rep,
    population,
    generation,
    run_config,
    model_architecture,
    node_config,
    meta_library,
    shared_inputs,
    programs,
    best_loss,
    best_program,
    elite_idx,
    Batch)

    if generation % c.every == 0
        best_individual = UTCGP.best_individual(rep)
        best_program = UTCGP.decode_with_output_nodes(
            best_individual,
            meta_library,
            model_architecture,
            shared_inputs,
        )
        save_payload(best_individual, best_program,
            nothing, shared_inputs,
            meta_library, run_config,
            node_config, "checkpoint_$generation.pickle")

    end
end
checkpoit_10 = checkpoint(10)
##########
#   FIT  #
##########

best_genome, best_programs, gen_tracker = UTCGP.fit_nsga2_atari_mt(
    shared_inputs,
    ut_genome,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    (UTCGP.nsga2_population_callback,),
    (UTCGP.nsga2_numbered_new_material_mutation_callback,),
    (:ga_output_mutation_callback,),
    (UTCGP.nsga2_decoding_callback,),
    # Endpoints
    AtariNSGA2Endpoint,
    # FINAL STEP CALLBACK
    nothing,
    # Callbacks after step (survival)
    (UTCGP.nsga2_survival_selection_callback,),
    # Epoch Callback
    # nothing, # (train_tracker, test_tracker), #[metric_tracker, test_tracker, sn_writer_callback],
    nothing,
    # (atari_tracker, checkpoit_10),
    # Final callbacks ?
    nothing, #(:default_early_stop_callback,), # 
    nothing #repeat_metric_tracker # .. 
)

# SAVE ALL 

# save_payload(best_genome, best_programs,
#     gen_tracker, shared_inputs,
#     ml, run_conf,
#     node_config)
# save_json_tracker(metric_tracker)
# close(metric_tracker.file)

# @show hash
# @show atari_tracker.test_losses[end]

# TO DESERIALIZE
# using Logging
# using Base.Threads
# import JSON
# using UnicodePlots
# using ErrorTypes
# using Revise
# using Dates
# using Infiltrator
# using Debugger
# using UTCGP
# using UTCGP: jsonTracker, save_json_tracker, repeatJsonTracker, jsonTestTracker
# using UTCGP: SN_writer, sn_strictphenotype_hasher
# using UTCGP: get_image2D_factory_bundles, SImageND, DataFrames
# using DataStructures
# using UUIDs
# using ImageCore
# using Statistics
# using StatsBase
# using UTCGP: SImage2D, BatchEndpoint
# using StatsBase: kurtosis, variation, sample
# using UTCGP: AbstractCallable, Population, RunConfGA, PopulationPrograms, IndividualPrograms, get_float_bundles, replace_shared_inputs!, evaluate_individual_programs, reset_programs!
# # using ImageView
# using ArcadeLearningEnvironment
# using MAGE_ATARI
# using Random
# using IOCapture
# GAME = "pong"
# _g = AtariEnv(GAME, 1)

# # these two params do not count now
# DOWNSCALE = false
# GRAYSCALE = true

# MAGE_ATARI.update_state(_g)
# MAGE_ATARI.update_screen(_g)
# _example = SImageND(N0f8.(Gray.(_g.screen)))
# w, h = size(_g)
# IMAGE_TYPE = typeof(_example)
# IMG_SIZE = _example |> size
# # Bundles Integer
# fallback() = SImageND(ones(N0f8, (IMG_SIZE[1], IMG_SIZE[2])))
# image2D = UTCGP.get_image2D_factory_bundles_atari()
# for factory_bundle in image2D
#     for (i, wrapper) in enumerate(factory_bundle)
#         try
#             fn = wrapper.fn(IMAGE_TYPE) # specialize
#             wrapper.fallback = fallback
#             # create a new wrapper in order to change the type
#             factory_bundle.functions[i] =
#                 UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)

#         catch
#         end
#     end
# end

# float_bundles = UTCGP.get_float_bundles_atari()

# # Libraries
# lib_image2D = Library(image2D)
# lib_float = Library(float_bundles)
# ml = MetaLibrary([lib_image2D, lib_float])
