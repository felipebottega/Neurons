#ifndef OFFSPRING_NATIVE_H
#define OFFSPRING_NATIVE_H

#include "genome_native.h"

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace godot {

class OffspringNative : public RefCounted {
	GDCLASS(OffspringNative, RefCounted);

protected:
	static void _bind_methods();

public:
	OffspringNative();
	~OffspringNative();

	Ref<GenomeNative> generate(
			const Ref<GenomeNative> &parent_a,
			const Ref<GenomeNative> &parent_b,
			double mutation_probability = 0.001);
	Ref<GenomeNative> genome_from_dictionary(const Dictionary &data);

private:
	static constexpr int32_t BLOCK_SIZE = 4;

	std::mt19937_64 rng;

	void _crossover_gene(
			const std::vector<int32_t> &gene_a,
			const std::vector<int32_t> &gene_b,
			std::vector<int32_t> &child_gene);
	void _mutate_gene(std::vector<int32_t> &gene, double mutation_probability);
	bool _read_gene(
			const Dictionary &data,
			const char *gene_name,
			int32_t expected_size,
			std::vector<int32_t> &destination) const;
};

}

#endif
