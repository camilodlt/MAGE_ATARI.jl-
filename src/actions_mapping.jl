ATARI_ACTIONS = Dict(
    "NOOP" => Int32(0),
    "FIRE" => Int32(1),
    "UP" => Int32(2),
    "RIGHT" => Int32(3),
    "LEFT" => Int32(4),
    "DOWN" => Int32(5),
    "UPRIGHT" => Int32(6),
    "UPLEFT" => Int32(7),
    "DOWNRIGHT" => Int32(8),
    "DOWNLEFT" => Int32(9),
    "UP_FIRE" => Int32(10),
    "RIGHT_FIRE" => Int32(11),
    "LEFT_FIRE" => Int32(12),
    "DOWN_FIRE" => Int32(13),
    "UPRIGHT_FIRE" => Int32(14),
    "UPLEFT_FIRE" => Int32(15),
    "DOWNRIGHT_FIRE" => Int32(16),
    "DOWNLEFT_FIRE" => Int32(17),
)

function atari_outputs_size(game_actions::Vector{Int})
    fire = 1 in game_actions
    left_right = (3 in game_actions) && (4 in game_actions)
    up_down = (2 in game_actions) && (5 in game_actions)
    return fire + left_right + up_down
end

function atari_outputs_mapping_fn(
    game_actions::Vector{Int},
)
    fire = 1 in game_actions
    left_right = (3 in game_actions) && (4 in game_actions)
    up_down = (2 in game_actions) && (5 in game_actions)
    diagonals = (6 in game_actions) && (7 in game_actions) && (8 in game_actions) && (9 in game_actions)
    return diagonals ? (o, t) -> atari_full_mapping(o, t, up_down, left_right, fire) : (o, t) -> atari_no_diagonal_mapping(o, t, fire)
end

function atari_full_mapping(
    outputs::Vector{Float64}, 
    thresholds::Vector{Float64}, 
    up_down::Bool,
    left_right::Bool,
    fire_action::Bool
)
    up = (up_down && outputs[1] > thresholds[1]) ? true : false
    right = ((left_right && up_down && outputs[2] > thresholds[2]) || (left_right && outputs[1] > thresholds[1])) ? true : false
    left = ((left_right && up_down && outputs[2] < -thresholds[2]) || (left_right && outputs[1] > thresholds[1])) ? true : false
    down = (up_down && outputs[1] < -thresholds[1]) ? true : false
    fire = ((fire_action && up_down && left_right && outputs[3] > thresholds[3]) || (fire_action && outputs[2] > thresholds[2])) ? true : false
    fire_addition = fire ? 8 : 0
    if up
        return right ? Int32(6 + fire_addition) : (left ? Int32(7 + fire_addition) : Int32(2 + fire_addition))
    end
    if down
        return right ? Int32(8 + fire_addition) : (left ? Int32(9 + fire_addition) : Int32(5 + fire_addition))
    end
    return right ? Int32(3 + fire_addition) : (left ? Int32(4 + fire_addition) : Int32(fire))
end

function atari_no_diagonal_mapping(outputs::Vector{Float64}, thresholds::Vector{Float64}, fire_action::Bool)
    up = outputs[1] > thresholds[1] ? abs(outputs[1]) : 0.0
    right = outputs[2] > thresholds[2] ? abs(outputs[2]) : 0.0
    left = outputs[2] < -thresholds[2] ? abs(outputs[2]) : 0.0
    down = outputs[1] < -thresholds[1] ? abs(outputs[1]) : 0.0
    fire = (fire_action && outputs[3] > thresholds[3]) ? true : 0.0
    fire_addition = fire ? 8 : 0
    if up || right || left || down
        return Int32(collect(2:5)[argmax([up, right, left, down])])
    end
    return Int32(fire)
end