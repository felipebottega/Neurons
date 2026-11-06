class_name SimulationAgent
extends RefCounted


var state: Dictionary = {}
# Typed hot-path coordinate. set_position() keeps the original dictionary
# synchronized for UI and save/inspection compatibility.
var position: Vector2 = Vector2.ZERO
# Last pre-movement progress, reused if the agent does not move.
var pre_move_progress: float = 0.0
# Maze target and distance field are constant during one target test.
# Cache the field query for the current position across consecutive steps.
var maze_cached_distance: float = INF
var maze_distance_valid: bool = false
var input_signals: PackedFloat64Array = PackedFloat64Array()
var output_signals: PackedFloat64Array = PackedFloat64Array()

var reset_behavior: Callable
var input_behavior: Callable
var output_behavior: Callable


func _init(
		num_inputs: int,
		num_outputs: int,
		p_reset_behavior: Callable,
		p_input_behavior: Callable,
		p_output_behavior: Callable
	) -> void:
	input_signals.resize(num_inputs)
	output_signals.resize(num_outputs)
	reset_behavior = p_reset_behavior
	input_behavior = p_input_behavior
	output_behavior = p_output_behavior


func set_position(next_position: Vector2) -> void:
	if next_position != position:
		maze_distance_valid = false
	position = next_position
	state["position"] = next_position


func reset() -> void:
	position = Vector2.ZERO
	pre_move_progress = 0.0
	maze_distance_valid = false
	state.clear()
	_clear_signals()
	reset_behavior.call(self)


func clear() -> void:
	position = Vector2.ZERO
	pre_move_progress = 0.0
	maze_distance_valid = false
	state.clear()
	_clear_signals()


func update_inputs() -> void:
	input_behavior.call(self)


func apply_outputs(step_delta: float) -> void:
	output_behavior.call(self, step_delta)


func _clear_signals() -> void:
	for input_index in input_signals.size():
		input_signals[input_index] = 0.0

	for output_index in output_signals.size():
		output_signals[output_index] = 0.0
