#include "offspring_native.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>

using namespace godot;

void OffspringNative::_bind_methods() {
	ClassDB::bind_method(
			D_METHOD("generate", "parent_a", "parent_b", "mutation_probability"),
			&OffspringNative::generate,
			DEFVAL(0.001));
	ClassDB::bind_method(
			D_METHOD("genome_from_dictionary", "data"),
			&OffspringNative::genome_from_dictionary);
}

OffspringNative::OffspringNative() {
	std::random_device random_device;
	const uint64_t seed = (static_cast<uint64_t>(random_device()) << 32) ^ static_cast<uint64_t>(random_device());
	rng.seed(seed);
}

OffspringNative::~OffspringNative() {
}

Ref<GenomeNative> OffspringNative::generate(
		const Ref<GenomeNative> &parent_a,
		const Ref<GenomeNative> &parent_b,
		double mutation_probability) {
	if (parent_a.is_null() || parent_b.is_null()) {
		return Ref<GenomeNative>();
	}
	if (mutation_probability < 0.0 || mutation_probability > 1.0) {
		return Ref<GenomeNative>();
	}

	Ref<GenomeNative> child;
	child.instantiate();

	_crossover_gene(parent_a->max_connections, parent_b->max_connections, child->max_connections);
	_crossover_gene(parent_a->beta, parent_b->beta, child->beta);
	_crossover_gene(parent_a->connections, parent_b->connections, child->connections);
	_crossover_gene(parent_a->activation_threshold, parent_b->activation_threshold, child->activation_threshold);
	_crossover_gene(parent_a->refractory_strength, parent_b->refractory_strength, child->refractory_strength);
	_crossover_gene(parent_a->exponential_factor, parent_b->exponential_factor, child->exponential_factor);
	_crossover_gene(parent_a->decay_factor, parent_b->decay_factor, child->decay_factor);
	_crossover_gene(parent_a->retention_factor, parent_b->retention_factor, child->retention_factor);
	_crossover_gene(parent_a->polarity_factor, parent_b->polarity_factor, child->polarity_factor);
	_crossover_gene(parent_a->hebbian_plasticity_rate, parent_b->hebbian_plasticity_rate, child->hebbian_plasticity_rate);
	_crossover_gene(parent_a->modulatory_release_factor, parent_b->modulatory_release_factor, child->modulatory_release_factor);
	_crossover_gene(parent_a->modulatory_sensitivity, parent_b->modulatory_sensitivity, child->modulatory_sensitivity);
	_crossover_gene(parent_a->modulatory_dynamics, parent_b->modulatory_dynamics, child->modulatory_dynamics);
	_crossover_gene(parent_a->structural_plasticity, parent_b->structural_plasticity, child->structural_plasticity);
	_crossover_gene(parent_a->input_influence, parent_b->input_influence, child->input_influence);

	_mutate_gene(child->max_connections, mutation_probability);
	_mutate_gene(child->beta, mutation_probability);
	_mutate_gene(child->connections, mutation_probability);
	_mutate_gene(child->activation_threshold, mutation_probability);
	_mutate_gene(child->refractory_strength, mutation_probability);
	_mutate_gene(child->exponential_factor, mutation_probability);
	_mutate_gene(child->decay_factor, mutation_probability);
	_mutate_gene(child->retention_factor, mutation_probability);
	_mutate_gene(child->polarity_factor, mutation_probability);
	_mutate_gene(child->hebbian_plasticity_rate, mutation_probability);
	_mutate_gene(child->modulatory_release_factor, mutation_probability);
	_mutate_gene(child->modulatory_sensitivity, mutation_probability);
	_mutate_gene(child->modulatory_dynamics, mutation_probability);
	_mutate_gene(child->structural_plasticity, mutation_probability);
	_mutate_gene(child->input_influence, mutation_probability);

	return child;
}

Ref<GenomeNative> OffspringNative::genome_from_dictionary(const Dictionary &data) {
	Ref<GenomeNative> genome;
	genome.instantiate();

	if (
			!_read_gene(data, "max_connections", 8, genome->max_connections) ||
			!_read_gene(data, "beta", 8, genome->beta) ||
			!_read_gene(data, "connections", 64, genome->connections) ||
			!_read_gene(data, "activation_threshold", 64, genome->activation_threshold) ||
			!_read_gene(data, "refractory_strength", 8, genome->refractory_strength) ||
			!_read_gene(data, "exponential_factor", 8, genome->exponential_factor) ||
			!_read_gene(data, "decay_factor", 64, genome->decay_factor) ||
			!_read_gene(data, "retention_factor", 64, genome->retention_factor) ||
			!_read_gene(data, "polarity_factor", 64, genome->polarity_factor) ||
			!_read_gene(data, "hebbian_plasticity_rate", 64, genome->hebbian_plasticity_rate) ||
			!_read_gene(data, "modulatory_release_factor", 64, genome->modulatory_release_factor) ||
			!_read_gene(data, "modulatory_sensitivity", 64, genome->modulatory_sensitivity) ||
			!_read_gene(data, "modulatory_dynamics", 8, genome->modulatory_dynamics) ||
			!_read_gene(data, "structural_plasticity", 8, genome->structural_plasticity) ||
			!_read_gene(data, "input_influence", 8, genome->input_influence)) {
		return Ref<GenomeNative>();
	}

	return genome;
}

void OffspringNative::_crossover_gene(
		const std::vector<int32_t> &gene_a,
		const std::vector<int32_t> &gene_b,
		std::vector<int32_t> &child_gene) {
	if (gene_a.size() != gene_b.size() || gene_a.size() % BLOCK_SIZE != 0) {
		child_gene.clear();
		return;
	}

	child_gene.resize(gene_a.size());
	std::uniform_int_distribution<int32_t> parent_distribution(0, 1);

	for (int32_t block_start = 0; block_start < static_cast<int32_t>(gene_a.size()); block_start += BLOCK_SIZE) {
		const std::vector<int32_t> &source_gene = parent_distribution(rng) == 0 ? gene_a : gene_b;
		for (int32_t index = block_start; index < block_start + BLOCK_SIZE; ++index) {
			child_gene[static_cast<size_t>(index)] = source_gene[static_cast<size_t>(index)];
		}
	}
}

void OffspringNative::_mutate_gene(std::vector<int32_t> &gene, double mutation_probability) {
	std::uniform_real_distribution<double> mutation_distribution(0.0, 1.0);
	std::uniform_int_distribution<int32_t> digit_distribution(0, 3);

	for (int32_t &digit : gene) {
		if (mutation_distribution(rng) < mutation_probability) {
			digit = digit_distribution(rng);
		}
	}
}

bool OffspringNative::_read_gene(
		const Dictionary &data,
		const char *gene_name,
		int32_t expected_size,
		std::vector<int32_t> &destination) const {
	const String key(gene_name);
	if (!data.has(key)) {
		return false;
	}

	const Array source = data[key];
	if (source.size() != expected_size) {
		return false;
	}

	destination.resize(static_cast<size_t>(expected_size));
	for (int32_t index = 0; index < expected_size; ++index) {
		const int64_t digit = static_cast<int64_t>(source[index]);
		if (digit < 0 || digit > 3) {
			return false;
		}
		destination[static_cast<size_t>(index)] = static_cast<int32_t>(digit);
	}

	return true;
}
