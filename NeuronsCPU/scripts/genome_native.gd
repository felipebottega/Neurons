class_name GenomeNative
extends Genome

# GDScript analogue of the former C++ genome adapter.
const GENES := {
    "max_connections": 8, "beta": 8, "connections": 64,
    "initial_search_region": 64, "activation_threshold": 64,
    "refractory_strength": 8, "exponential_factor": 8, "decay_factor": 64,
    "retention_factor": 64, "polarity_factor": 64,
    "hebbian_plasticity_rate": 64, "modulatory_release_factor": 64,
    "modulatory_sensitivity": 64, "modulatory_dynamics": 8,
    "structural_plasticity": 8, "input_influence": 8
}

func get_gene(name: String) -> Array:
    return get(name).duplicate()

func to_dictionary() -> Dictionary:
    var result := {}
    for name in GENES:
        result[name] = get(name).duplicate()
    return result
