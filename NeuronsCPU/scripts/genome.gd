class_name Genome
extends Node


var max_connections: Array[int]
var beta: Array[int]
var connections: Array[int]
var activation_threshold: Array[int]
var refractory_strength: Array[int]
var exponential_factor: Array[int]
var decay_factor: Array[int]
var retention_factor: Array[int]
var polarity_factor: Array[int]
var hebbian_plasticity_rate: Array[int]
var modulatory_release_factor: Array[int]
var modulatory_sensitivity: Array[int]
var modulatory_dynamics: Array[int]
var structural_plasticity: Array[int]
var input_influence: Array[int]


## Creates a random genome.
func _init() -> void:
	max_connections = _random_gene(8)
	beta = _random_gene(8)
	connections = _random_gene(64)
	activation_threshold = _random_gene(64)
	refractory_strength = _random_gene(8)
	exponential_factor = _random_gene(8)
	decay_factor = _random_gene(64)
	retention_factor = _random_gene(64)
	polarity_factor = _random_gene(64)
	hebbian_plasticity_rate = _random_gene(64)
	modulatory_release_factor = _random_gene(64)
	modulatory_sensitivity = _random_gene(64)
	modulatory_dynamics = _random_gene(8)
	structural_plasticity = _random_gene(8)
	input_influence = _random_gene(8)

## Creates a random gene with values from 0 to 3.
## [param size] defines the number of digits in the gene.
## Returns the generated gene.
func _random_gene(size: int) -> Array[int]:
	var gene: Array[int] = []

	for i in size:
		gene.append(randi_range(0, 3))

	return gene
