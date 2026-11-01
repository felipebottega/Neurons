class_name SimulationAgent
extends RefCounted


var state: Dictionary = {}
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


func reset() -> void:
	state.clear()
	_clear_signals()
	reset_behavior.call(self)


func clear() -> void:
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
