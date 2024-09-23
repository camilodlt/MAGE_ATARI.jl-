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
using UTCGP: AbstractCallable, Population, RunConfGA, PopulationPrograms, IndividualPrograms, get_float_bundles, replace_shared_inputs!, evaluate_individual_programs, reset_programs!
# using ImageView
using ArcadeLearningEnvironment
using MAGE_ATARI
using Random
using IOCapture
# using Plots

# RUN_FILE = "results/skiing_me_1.csv"
RUN_FILE = "results/freeway_me_1.csv"
open(RUN_FILE, "a") do f
    write(f,join(["iteration", "best_fitness", "coverage"], ",")* "\n")  
end


@inline function relu(x)
    x > 0 ? x : 0
end

dir = @__DIR__
pwd_dir = pwd()

file = @__FILE__
home = dirname(dirname(file))


PROB_ACTION = true
# GAME = "skiing"
GAME = "freeway"
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
function skiing_action_mapping(outputs::Vector, p::Type{prob}, mt)
    global ACTIONS
    os = relu.(outputs)
    w = Weights(os)
    action = sample(mt, ACTIONS, w)
    action
end

action_mapping_dict = Dict(
    "pong" => [3, pong_action_mapping],
    "skiing" => [3, skiing_action_mapping],
    "freeway" => [3, skiing_action_mapping]
)

NACTIONS = action_mapping_dict[GAME][1]
ACTIONS = GAME == "pong" ? TRUE_ACTIONS[2:NACTIONS+1] : TRUE_ACTIONS
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

#
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
    game = AtariEnv(GAME, 1, ATARI_LOCK)
    # end
    Random.seed!(seed)
    mt = MersenneTwister(seed)
    #game = Game(rom, seed, lck=lck)
    MAGE_ATARI.reset!(game)
    # reset!(reducer) # zero buffers
    max_frames = 18_000
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
        meta_library::MetaLibrary
    )
        n = length(pop)
        nt = nthreads()
        indices = collect(1:n)
        BS = ceil(Int, n / nt)
        pop_res = Vector{Float64}(undef, n)
        pop_descriptor = Vector{Vector{Float64}}(undef, n)
        tasks = []
        for ith_x in Iterators.partition(indices, BS)
            t = @spawn begin
                @info "Spawn the task of $ith_x to thread $(threadid())"
                for i in ith_x
                    f, d = atari_fitness(deepcopy(pop[i]), 1, model_arch, meta_library)
                    # @show f
                    # @show d
                    pop_res[i] = f
                    pop_descriptor[i] = d

                end
            end
            push!(tasks, t)
        end
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
centroids_grid = vec([[i,j] for i in centroids, j in centroids])
sample_size = 50
gens = 3000
mut_rate = 2.1

run_conf = UTCGP.RunConfME(
    centroids_grid, sample_size, mut_rate, 0.1, gens
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
function me_epoch_callback_fn(inds, rep::UTCGP.MapelitesRepertoire, iteration)
    open(RUN_FILE, "a") do f
        write(f, join(string.([iteration, UTCGP.best_fitness(rep), UTCGP.coverage(rep)]), ",") * "\n")  
    end
end




##########
#   FIT  #
##########

best_genome, best_programs, gen_trakcer = UTCGP.fit_me_atari_mt(
    shared_inputs,
    ut_genome,
    model_arch,
    node_config,
    run_conf,
    ml,
    # Callbacks before training
    nothing,
    # Callbacks before step
    (UTCGP.me_population_callback,),
    (UTCGP.me_numbered_new_material_mutation_callback,),
    (:ga_output_mutation_callback,),
    (UTCGP.me_decoding_callback,),
    # Endpoints
    AtariMEEndpoint,
    # STEP CALLBACK
    nothing,
    # Callbacks after step
    nothing,
    # Epoch Callback
    # nothing, # (train_tracker, test_tracker), #[metric_tracker, test_tracker, sn_writer_callback],
    (me_epoch_callback_fn,),
    # Final callbacks ?
    nothing, #(:default_early_stop_callback,), # 
    nothing #repeat_metric_tracker # .. 
)
