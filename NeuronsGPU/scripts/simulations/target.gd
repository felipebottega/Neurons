class_name SimulationTarget
extends RefCounted


var position: Vector2 = Vector2.ZERO
var state: Dictionary = {}

var reset_behavior: Callable
var update_behavior: Callable


func _init(p_reset_behavior: Callable, p_update_behavior: Callable) -> void:
	reset_behavior = p_reset_behavior
	update_behavior = p_update_behavior


func reset() -> void:
	position = Vector2.ZERO
	state.clear()
	reset_behavior.call(self)


func clear() -> void:
	position = Vector2.ZERO
	state.clear()


func update(step_delta: float) -> void:
	update_behavior.call(self, step_delta)
