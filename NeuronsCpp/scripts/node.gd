extends Node

func _ready():
	var dynamics = BrainDynamicsNative.new()
	print(dynamics.advance(0.016))
