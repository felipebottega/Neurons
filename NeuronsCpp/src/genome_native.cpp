#include "genome_native.h"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void GenomeNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GenomeNative::to_dictionary);
	ClassDB::bind_method(D_METHOD("get_gene", "gene_name"), &GenomeNative::get_gene);
}

GenomeNative::GenomeNative() {
	std::random_device random_device;
	const uint64_t seed = (static_cast<uint64_t>(random_device()) << 32) ^ static_cast<uint64_t>(random_device());
	rng.seed(seed);

	_random_gene(max_connections, 8);
	_random_gene(beta, 8);
	_random_gene(connections, 64);
	_random_gene(activation_threshold, 64);
	_random_gene(refractory_strength, 8);
	_random_gene(exponential_factor, 8);
	_random_gene(decay_factor, 64);
	_random_gene(retention_factor, 64);
	_random_gene(polarity_factor, 64);
	_random_gene(hebbian_plasticity_rate, 64);
	_random_gene(modulatory_release_factor, 64);
	_random_gene(modulatory_sensitivity, 64);
	_random_gene(modulatory_dynamics, 8);
	_random_gene(structural_plasticity, 8);
	_random_gene(input_influence, 8);
}

GenomeNative::~GenomeNative() {
}

void GenomeNative::_random_gene(std::vector<int32_t> &gene, int32_t size) {
	gene.resize(static_cast<size_t>(size));
	std::uniform_int_distribution<int32_t> distribution(0, 3);

	for (int32_t i = 0; i < size; ++i) {
		gene[static_cast<size_t>(i)] = distribution(rng);
	}
}

TypedArray<int64_t> GenomeNative::_gene_to_array(const std::vector<int32_t> &gene) {
	TypedArray<int64_t> result;
	result.resize(static_cast<int64_t>(gene.size()));

	for (int32_t i = 0; i < static_cast<int32_t>(gene.size()); ++i) {
		result[i] = gene[static_cast<size_t>(i)];
	}

	return result;
}

Dictionary GenomeNative::to_dictionary() const {
	Dictionary data;
	data["max_connections"] = _gene_to_array(max_connections);
	data["beta"] = _gene_to_array(beta);
	data["connections"] = _gene_to_array(connections);
	data["activation_threshold"] = _gene_to_array(activation_threshold);
	data["refractory_strength"] = _gene_to_array(refractory_strength);
	data["exponential_factor"] = _gene_to_array(exponential_factor);
	data["decay_factor"] = _gene_to_array(decay_factor);
	data["retention_factor"] = _gene_to_array(retention_factor);
	data["polarity_factor"] = _gene_to_array(polarity_factor);
	data["hebbian_plasticity_rate"] = _gene_to_array(hebbian_plasticity_rate);
	data["modulatory_release_factor"] = _gene_to_array(modulatory_release_factor);
	data["modulatory_sensitivity"] = _gene_to_array(modulatory_sensitivity);
	data["modulatory_dynamics"] = _gene_to_array(modulatory_dynamics);
	data["structural_plasticity"] = _gene_to_array(structural_plasticity);
	data["input_influence"] = _gene_to_array(input_influence);
	return data;
}

Array GenomeNative::get_gene(const StringName &gene_name) const {
	if (gene_name == StringName("max_connections")) {
		return _gene_to_array(max_connections);
	}
	if (gene_name == StringName("beta")) {
		return _gene_to_array(beta);
	}
	if (gene_name == StringName("connections")) {
		return _gene_to_array(connections);
	}
	if (gene_name == StringName("activation_threshold")) {
		return _gene_to_array(activation_threshold);
	}
	if (gene_name == StringName("refractory_strength")) {
		return _gene_to_array(refractory_strength);
	}
	if (gene_name == StringName("exponential_factor")) {
		return _gene_to_array(exponential_factor);
	}
	if (gene_name == StringName("decay_factor")) {
		return _gene_to_array(decay_factor);
	}
	if (gene_name == StringName("retention_factor")) {
		return _gene_to_array(retention_factor);
	}
	if (gene_name == StringName("polarity_factor")) {
		return _gene_to_array(polarity_factor);
	}
	if (gene_name == StringName("hebbian_plasticity_rate")) {
		return _gene_to_array(hebbian_plasticity_rate);
	}
	if (gene_name == StringName("modulatory_release_factor")) {
		return _gene_to_array(modulatory_release_factor);
	}
	if (gene_name == StringName("modulatory_sensitivity")) {
		return _gene_to_array(modulatory_sensitivity);
	}
	if (gene_name == StringName("modulatory_dynamics")) {
		return _gene_to_array(modulatory_dynamics);
	}
	if (gene_name == StringName("structural_plasticity")) {
		return _gene_to_array(structural_plasticity);
	}
	if (gene_name == StringName("input_influence")) {
		return _gene_to_array(input_influence);
	}
	return Array();
}
