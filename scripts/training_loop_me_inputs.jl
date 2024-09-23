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
using ImageView
using ArcadeLearningEnvironment
using MAGE_ATARI
using Random
using IOCapture
using Plots

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
MAGE_ATARI.update_state(_g)
MAGE_ATARI.update_screen(_g)
_example = SImageND(N0f8.(Gray.(_g.screen)))
w, h = size(_g)
IMAGE_TYPE = SImage2D{w,h,N0f8,Matrix{N0f8}}
TRUE_ACTIONS = MAGE_ATARI.actions(_g, _g.state)
ATARI_LOCK = ReentrantLock()

abstract type AbstractProb end
struct prob <: AbstractProb end
struct notprob <: AbstractProb end

function pong_action_mapping(outputs::Vector, p::Type{notprob})
    global ACTIONS
    out = outputs[1]
    out_i = trunc(Int, out)
    # @show out_i
    if out_i in ACTIONS
        return out_i
    else
        return ACTIONS[1]
    end
end
function pong_action_mapping(outputs::Vector, p::Type{prob}, mt)
    global ACTIONS
    out = outputs[1]
    out_i = trunc(Int, out)
    # @show out_i
    if out_i in ACTIONS
        return out_i
    else
        return ACTIONS[1]
    end
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
# hash = UUIDs.uuid4() |> string


# HARCODED FUNCTIONS
mask_between = UTCGP.experimental_image2D_mask.experimental_bundle_image2D_maskregion_factory[1].fn(typeof(_example))
mask_vertical = UTCGP.experimental_image2D_mask.experimental_bundle_image2D_maskregion_factory[2].fn(typeof(_example))
x_max = UTCGP.bundle_number_coordinatesFromImg[1].fn
y_max = UTCGP.bundle_number_coordinatesFromImg[2].fn

function top_mask(s)
    mask_vertical(s, 37, 193)
end

function center_mask(s)
    mask_between(s, 26, 139)
end
function right_mask(s)
    mask_between(s, 140, 1000)
end

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
    UTCGP.reset_programs!(ind_prog)
    # @show outputs
    outputs
end

# PARAMS --- --- 
function atari_fitness(ind::IndividualPrograms, seed, model_arch::modelArchitecture, meta_library::MetaLibrary)
    global GAME, ACTIONS
    game = nothing
    # IOCapture.capture() do
    game = AtariEnv(GAME, 1, ATARI_LOCK)
    # end
    Random.seed!(seed)
    mt = MersenneTwister(seed)
    #game = Game(rom, seed, lck=lck)
    MAGE_ATARI.reset!(game)
    # reset!(reducer) # zero buffers
    max_frames = 1000
    stickiness = 0.25
    reward = 0.0
    frames = 0
    prev_action = Int32(0)
    actions_counter = Dict(
        [action => 0 for action in ACTIONS]...
    )
    while ~game_over(game.ale)
        # if rand(mt) > stickiness || frames == 0
        if rand(mt) > stickiness || frames == 0
            MAGE_ATARI.update_screen(game)
            current_gray_screen = N0f8.(Gray.(game.screen))
            cur_frame = SImageND(current_gray_screen)
            c = top_mask(cur_frame)
            center = center_mask(c)
            right = right_mask(c)
            ball_x, ball_y = float_caster(x_max(center)), float_caster(y_max(center))
            right_x, right_y = float_caster(x_max(right)), float_caster(y_max(right))
            inputs = [ball_x, ball_y, right_x, right_y, 1.0, 10.0, 100.0, -1.0, ACTIONS...]
            outputs = evaluate_individual_programs!(ind, inputs, model_arch, meta_library)
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
        actions_counter[action] += 1
        reward += r
        frames += 1
        prev_action = action
        if frames > max_frames
            # @info "Game finished because of max_frames"
            break
        end
        # calculate the descriptor
    end
    MAGE_ATARI.close(game)
    tot = sum(values(actions_counter))
    action_tot = sum([actions_counter[a] for a in ACTIONS[2:end]])
    descriptor = action_tot / tot

    # movement_pct 
    left_counter = actions_counter[3]
    right_counter = actions_counter[4]
    movement_pct = 0.0
    if left_counter == right_counter
        movement_pct += 0.5 # no movement
    elseif left_counter == 0.0
        movement_pct += 1.0 # completely to the right
    elseif right_counter == 0.0
        movement_pct += 0.0 # completely to the left
    else # a combination of right and left
        movement_pct += right_counter / (left_counter + right_counter)
    end
    r = reward * -1.0
    r, [descriptor, movement_pct]
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
                    @show f
                    @show d
                    pop_res[i] = f
                    pop_descriptor[i] = d

                end
            end
            push!(tasks, t)
        end
        results = fetch.(tasks)
        @show results
        @show pop_res
        return new(pop_res, pop_descriptor)
    end
end

function UTCGP.get_endpoint_results(e::AtariMEEndpoint)
    println("I'm called")
    return e.fitness_results, e.descriptor_results
end


### RUN CONF ###
usage_of_action = collect(0:0.05:0.99) .+ 0.05 / 2
movement_pct = collect(0:0.05:0.99) .+ 0.05 / 2
centroids = [[i, j] for i in usage_of_action for j in movement_pct]
# centroids_v = map(i -> [i], centroids)
sample_size = 50
gens = 1000
mut_rate = 1.1

run_conf = UTCGP.RunConfME(
    centroids, sample_size, mut_rate, 0.1, gens
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
ml = MetaLibrary([lib_float])

offset_by = 11 # 0.1 0.2 1. -1.

### Model Architecture ###
model_arch = modelArchitecture(
    [Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int, Int, Int],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3],
    # [IMAGE_TYPE, Float64, Vector{Float64}, Int], # genome
    [Float64], # genome
    [Float64], # outputs
    [1]
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
    nothing, # (train_tracker, test_tracker), #[metric_tracker, test_tracker, sn_writer_callback],
    # Final callbacks ?
    nothing, #(:default_early_stop_callback,), # 
    nothing #repeat_metric_tracker # .. 
)

