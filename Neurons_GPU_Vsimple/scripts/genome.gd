class_name Genome
extends Node

# Vsimple: the genome fixes architecture, polarity and the transmission scale.
# The network has no synaptic or structural updates during a lifetime.
var max_connections: Array[int]
var beta: Array[int]
var connections: Array[int]
var initial_search_region: Array[int]
var exponential_factor: Array[int]
var polarity_factor: Array[int]
var input_influence: Array[int]

func _init() -> void:
	max_connections = _random_gene(8)
	beta = _random_gene(8)
	connections = _random_gene(64)
	initial_search_region = _random_gene(64)
	exponential_factor = _random_gene(8)
	polarity_factor = _random_gene(64)
	input_influence = _random_gene(8)

func _random_gene(size: int) -> Array[int]:
	var gene: Array[int] = []
	for i in size:
		gene.append(randi_range(0, 3))
	return gene
