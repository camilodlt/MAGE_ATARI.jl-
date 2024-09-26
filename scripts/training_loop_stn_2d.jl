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
using UTCGP: AbstractCallable, Population, RunConfGA, PopulationPrograms, IndividualPrograms, get_float_bundles, replace_shared_inputs!, evaluate_individual_programs, reset_programs!, ParametersStandardEpoch
# using ImageView
using ArcadeLearningEnvironment
using MAGE_ATARI
using Random
using IOCapture
using ThreadPinning
import SearchNetworks as sn

println(threadinfo(color=false, slurm=true))
pinthreads(:cores)
println(threadinfo(color=false, slurm=true))

@inline function relu(x)
    x > 0 ? x : 0
end

dir = @__DIR__
pwd_dir = pwd()

file = @__FILE__
home = dirname(dirname(file))


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

# Fixed Episode
function random_game(n::Int)
    frames = []
    _g = AtariEnv(GAME, 1)
    q = Queue{IMAGE_TYPE}()
    MAGE_ATARI.update_state(_g)
    MAGE_ATARI.update_screen(_g)
    for i in 1:n
        current_gray_screen = N0f8.(Gray.(_g.screen))
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

        action = rand(TRUE_ACTIONS)
        r, s = MAGE_ATARI.step!(_g, _g.state, action)
        MAGE_ATARI.update_state(_g)
        MAGE_ATARI.update_screen(_g)
        push!(frames, o_copy)
    end
    frames
end

random_frames = random_game(18_000)
random_frames_subset = sample(random_frames, 200)
# 

abstract type AbstractProb end
struct prob <: AbstractProb end
struct notprob <: AbstractProb end

function pong_deterministic_action_mapping(outputs::Vector, p::Type{notprob})
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
function atari_fitness(ind::IndividualPrograms, seed::Int, model_arch::modelArchitecture, meta_library::MetaLibrary)
    global GAME
    game = nothing
    # IOCapture.capture() do
    game = AtariEnv(GAME, 1, ATARI_LOCK)
    # end
    Random.seed!(1)
    mt = MersenneTwister(1)
    #game = Game(rom, seed, lck=lck)
    MAGE_ATARI.reset!(game)
    # reset!(reducer) # zero buffers
    max_frames = 5000
    stickiness = 0.25
    reward = 0.0
    frames = 0
    prev_action = Int32(0)
    actions_counter = Dict(
        [action => 0 for action in ACTIONS]...
    )
    action_changes = 0
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
    r = reward * -1.0
    r, [activity_descriptor, action_changes_descriptor]
end

struct AtariMEEndpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Float64}
    descriptor_results::Vector{Vector{Float64}}
    function AtariMEEndpoint(
        pop::PopulationPrograms,
        model_arch::modelArchitecture,
        meta_library::MetaLibrary,
        iteration::Int
    )
        n = length(pop)
        nt = nthreads()
        indices = collect(1:n)
        BS = ceil(Int, n / nt)
        pop_res = Vector{Float64}(undef, n)
        pop_descriptor = Vector{Vector{Float64}}(undef, n)
        tasks = []
        previous_rand = copy(Random.default_rng())
        for ith_x in Iterators.partition(indices, BS)
            t = @spawn begin
                @info "Spawn the task of $ith_x to thread $(threadid())"
                for i in ith_x
                    f, d = atari_fitness(deepcopy(pop[i]), iteration, model_arch, meta_library)
                    # @show f
                    # @show d
                    pop_res[i] = f
                    pop_descriptor[i] = d

                end
            end
            push!(tasks, t)
        end
        copy!(Random.default_rng(), previous_rand)
        results = fetch.(tasks)
        # @show results
        # @show pop_res
        return new(pop_res, pop_descriptor)
    end
end

function UTCGP.get_endpoint_results(e::AtariMEEndpoint)
    return e.fitness_results, e.descriptor_results
end


### RUN CONF ###
centroids = collect(0:0.05:0.99) .+ 0.05 / 2
centroids_grid = vec([[i, j] for i in centroids, j in centroids])
sample_size = 10
gens = 300
mut_rate = 3.0

run_conf = UTCGP.RunConfSTN(
    sample_size, "behavior_hash", "serialization_hash",
    mut_rate, 0.1, gens
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

metrics_path = joinpath(folder, "metrics.json")
struct jsonTrackerME <: UTCGP.AbstractCallable
    tracker::UTCGP.jsonTracker
    label::String
    test_losses::Vector
end

f = open(metrics_path, "a", lock=true)
metric_tracker = UTCGP.jsonTracker(h_params, f)
atari_tracker = jsonTrackerME(metric_tracker, "Test", [])

function (jtga::jsonTrackerME)(args::UTCGP.STN_EPOCH_ARGS)
    # best_f = UTCGP.best_fitness(rep)
    # @warn "JTT $(jtga.label) Fitness : $best_f"
    # s = Dict("data" => jtga.label, "iteration" => generation,
    #     "coverage" => UTCGP.coverage(rep), "best_fitness" => best_f)

    # push!(jtga.test_losses, best_f)
    # write(jtga.tracker.file, JSON.json(s), "\n")
    # flush(jtga.tracker.file)
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

function (c::checkpoint)(args::UTCGP.STN_EPOCH_ARGS)
    # if generation % c.every == 0
    #     best_individual = UTCGP.best_individual(rep)
    #     best_program = UTCGP.decode_with_output_nodes(
    #         best_individual,
    #         meta_library,
    #         model_architecture,
    #         shared_inputs,
    #     )
    #     save_payload(best_individual, best_program,
    #         nothing, shared_inputs,
    #         meta_library, run_config,
    #         node_config, "checkpoint_$generation.pickle")

    # end
end
checkpoit_10 = checkpoint(10)

##############
#    STN     #
##############
db_name = hash * ".db"
db_folder = joinpath(folder, "db")
db_path = joinpath(db_folder, db_name)
isdir(db_folder) || mkdir(db_folder)
con = sn.create_DB(db_path)
sn.create_SN_tables!(
    con,
    extra_nodes_cols=OrderedDict(
        "gen_hash" => sn.SN_col_type(string=true),
        "phen_hash" => sn.SN_col_type(string=true),
        "behavior_hash" => sn.SN_col_type(string=true),
        "serialization_hash" => sn.SN_col_type(string=true),
        "db_name" => sn.SN_col_type(string=true),
    ),
    extra_edges_cols=OrderedDict(
        "fitness" => sn.SN_col_type(float=true),
        # "is_elite" => sn.SN_col_type(float=true),
        "db_name" => sn.SN_col_type(string=true),
        "conf_name" => sn.SN_col_type(string=true),
    )
)

struct sn_atari_behavior_hasher <: UTCGP.Abstract_Node_Hash_Function
    semantic_examples::Vector
    seed::Int
end

"""
Plays a fixed game for each individual. 
The outputs from each individual are hashed.
"""
function (s::sn_atari_behavior_hasher)(args::ParametersStandardEpoch)
    pop = args.programs
    semantic_hashes = Vector{String}(undef, length(pop))
    previous_rand = copy(Random.default_rng())
    for (i, ind) in enumerate(pop)
        Random.seed!(s.seed)
        mt = MersenneTwister(s.seed)
        actions = []
        for semantic_example in s.semantic_examples
            outputs = evaluate_individual_programs!(ind, semantic_example, args.model_architecture, args.meta_library)
            if PROB_ACTION
                action = ACTION_MAPPER(outputs, prob, mt)
            else
                action = ACTION_MAPPER(outputs, notprob)
            end
            push!(actions, action)
        end
        ind_semantics = UTCGP.general_hasher_sha(identity.(actions))
        semantic_hashes[i] = ind_semantics
    end
    copy!(Random.default_rng(), previous_rand)
    semantic_hashes
end

"""
Serializes the genome to be able to recover it later.
"""
struct sn_serialization_hasher <: UTCGP.Abstract_Node_Hash_Function end
function (::sn_serialization_hasher)(p::ParametersStandardEpoch)
    [UTCGP.serialize_ind_to_string(ind) for ind in p.population]
end

sn_writer_callback = SN_writer(
    con,
    all_edges(),
    OrderedDict(
        "gen_hash" => sn_genotype_hasher(),
        "phen_hash" => sn_strictphenotype_hasher(),
        "behavior_hash" => sn_atari_behavior_hasher(random_frames_subset, 1),
        "serialization_hash" => sn_serialization_hasher(),
        "db_name" => sn_db_name_node(db_name)
    ),
    OrderedDict(
        "fitness" => sn_fitness_hasher(),
        # "is_elite" => sn_elite_hasher(),
        "db_name" => sn_db_name_edge(db_name),
        "conf_name" => sn_db_name_edge(GAME)
    )
)

"""
The instantiated SN_writer writes : 
- The nodes 
- The edges between the parent and all the children

Hashes will depend on which hashers where passed to the SN_writer during initialization. 

All individuals are hashed with those hashers. 

Edges happen between the `id_hash` of the parent and that of every child. 
Extra cols for the EDGE table depend on the `edges_prop_getters` of the struct. 

Note: `id_hash` is the hash of the union of all extra hashes. 
"""
function (writer::SN_writer)(
    epoch_args::UTCGP.STN_EPOCH_ARGS
    # ind_performances::Union{Vector{<:Number},Vector{Vector{<:Number}}},
    # population::Population,
    # generation::Int,
    # run_config::runConf,
    # model_architecture::modelArchitecture,
    # node_config::nodeConfig,
    # meta_library::MetaLibrary,
    # shared_inputs::SharedInput,
    # programs::PopulationPrograms,
    # edges_indices_fn::Abstract_Edge_Prop_Getter,
    # best_loss::Float64,
    # best_program::IndividualPrograms,
    # elite_idx::Int,
)
    epoch_params = UTCGP.ParametersStandardEpoch(
        epoch_args.ind_performances,
        epoch_args.population,
        epoch_args.generation,
        epoch_args.run_config,
        epoch_args.model_architecture,
        epoch_args.node_config,
        epoch_args.meta_library,
        epoch_args.shared_inputs,
        epoch_args.programs,
        minimum(epoch_args.ind_performances),#epoch_args.best_loss,
        epoch_args.programs[1],#epoch_args.best_program,
        1,#epoch_args.elite_idx,
        Dict("mappings" => epoch_args.mappings)
    )

    gen = epoch_args.generation
    # Hash all individuals (NODES)
    all_hash_rows = UTCGP._get_rows_by_running_all_fns(epoch_params, writer, sn.Abstract_Nodes)
    UTCGP._log_n_rows_view(all_hash_rows, sn.Abstract_Nodes)

    # Get Info for all edges (EDGES)
    indices_for_edges = writer.edges_indices_fn(epoch_params)
    all_edges_info = UTCGP._get_rows_by_running_all_fns(epoch_params, writer, sn.Abstract_Edges)
    UTCGP._log_n_rows_view(all_edges_info, sn.Abstract_Edges)
    @assert length(indices_for_edges) == length(all_edges_info) "Mismatch between the number of edges rows and the indices of those edges : $(all_edges_info) vs $(indices_for_edges )"

    # write Nodes to DB
    r = sn.write_only_new_to_nodes!(writer.con, identity.(all_hash_rows))
    @assert r == 0 "DB result is 1, so writing failed"

    # Write Edges if any
    # p_hash = all_hash_rows[end]["id_hash"] # pick the parent
    for (ith_child, child, mutable_map) in zip(indices_for_edges, all_edges_info, epoch_args.mappings)
        if epoch_args.generation == 1
            # The edge does not have a parent 
            @info "'First Iteration makes an edge between the individual and itself"
            @debug "Edge from parent => $ith_child"
            _to = all_hash_rows[ith_child]["id_hash"] # ⚠ The ith child has to match with the length of the hashed rows. # make a fn that verif this ? # TODO
            r = sn.write_to_edges!(
                writer.con,
                OrderedDict(
                    "_from" => _to,
                    "_to" => _to,
                    "iteration" => gen,
                    child...,
                ),
            )
            @assert r == 0 "DB result is 1, so writing edge failed"
        else
            p_hash = mutable_map.first # retrieved from DB so its correct
            @debug "Edge from parent => $ith_child"
            _to = all_hash_rows[ith_child]["id_hash"] # Calculated on the fly. ⚠ The ith child has to match with the length of the hashed rows. # make a fn that verif this ? # TODO
            mutable_map.second = _to # update the mappings

            r = sn.write_to_edges!(
                writer.con,
                OrderedDict(
                    "_from" => p_hash,
                    "_to" => _to,
                    "iteration" => gen,
                    child...,
                ),
            )
            @assert r == 0 "DB result is 1, so writing edge failed"
        end
    end

    # CHECKPOINT EVERY 10 its
    if gen % 10 == 0
        println("Manual DB Checkpoint at generation $gen")
        sn._execute_command(writer.con, "CHECKPOINT")
    end
    #
end

##########
#   FIT  #
##########

# best_genome, best_programs, gen_tracker = UTCGP.fit_stn_atari_mt(
UTCGP.fit_stn_atari_mt(
    con,
    shared_inputs,
    ut_genome,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    (UTCGP.stn_population_callback,),
    (UTCGP.stn_numbered_new_material_mutation_callback,),
    (:ga_output_mutation_callback,),
    (UTCGP.stn_decoding_callback,),
    # Endpoints
    AtariMEEndpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    nothing,
    # Epoch Callback
    # nothing, # (train_tracker, test_tracker), #[metric_tracker, test_tracker, sn_writer_callback],
    (atari_tracker, checkpoit_10, sn_writer_callback),
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
