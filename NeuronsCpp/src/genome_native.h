#ifndef GENOME_NATIVE_H
#define GENOME_NATIVE_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace godot {

class BrainBuilderNative;
class OffspringNative;

class GenomeNative : public RefCounted {
	GDCLASS(GenomeNative, RefCounted);

protected:
	static void _bind_methods();

public:
	GenomeNative();
	~GenomeNative();

	Dictionary to_dictionary() const;
	Array get_gene(const StringName &gene_name) const;

private:
	friend class BrainBuilderNative;
	friend class OffspringNative;

	std::vector<int32_t> max_connections;
	std::vector<int32_t> beta;
	std::vector<int32_t> connections;
	std::vector<int32_t> activation_threshold;
	std::vector<int32_t> refractory_strength;
	std::vector<int32_t> exponential_factor;
	std::vector<int32_t> decay_factor;
	std::vector<int32_t> retention_factor;
	std::vector<int32_t> polarity_factor;
	std::vector<int32_t> hebbian_plasticity_rate;
	std::vector<int32_t> modulatory_release_factor;
	std::vector<int32_t> modulatory_sensitivity;
	std::vector<int32_t> modulatory_dynamics;
	std::vector<int32_t> structural_plasticity;
	std::vector<int32_t> input_influence;

	std::mt19937_64 rng;

	void _random_gene(std::vector<int32_t> &gene, int32_t size);
	static TypedArray<int64_t> _gene_to_array(const std::vector<int32_t> &gene);
};

}

#endif
