#ifndef BRAIN_BUILDER_NATIVE_H
#define BRAIN_BUILDER_NATIVE_H

#include "genome_native.h"

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <cstdint>
#include <vector>

namespace godot {

class BrainBuilderNative : public RefCounted {
	GDCLASS(BrainBuilderNative, RefCounted);

protected:
	static void _bind_methods();

public:
	BrainBuilderNative();
	~BrainBuilderNative();

	Dictionary build(
			const Ref<GenomeNative> &genome,
			const Vector3i &size,
			int64_t num_inputs,
			int64_t num_outputs,
			const Callable &progress_callback = Callable(),
			double connections_density = 1.0);

private:
	struct NativeConnection {
		int32_t target_spatial_index = 0;
		float weight = 0.0f;
		float eligibility_trace = 0.0f;
		uint8_t construction_pass = 0; // 0=local, 1=complementary, 2=rescue
	};

	struct NativeNeuron {
		Vector3i position;
		bool input = false;
		bool output = false;
		float activation_threshold = 0.0f;
		float decay_factor = 0.0f;
		float retention_factor = 0.0f;
		int32_t polarity_factor = 1;
		float hebbian_plasticity_rate = 0.0f;
		float modulatory_release_factor = 0.0f;
		float modulatory_sensitivity = 0.0f;
		float modulatory_field = 0.0f;
		float connections_value = 0.0f;
		std::vector<NativeConnection> connections;
	};

	struct IsolationCounts {
		int32_t internal = 0;
		int32_t output = 0;
		int32_t input = 0;
	};

	struct RescueFailureCauses {
		int32_t no_structural_candidate = 0;
		int32_t no_positive_h_candidate = 0;
		int32_t positive_h_candidates_saturated = 0;
		int32_t input_source_saturated = 0;
	};

	struct NativeBrain {
		Vector3i size;
		double beta = 0.0;
		int32_t max_connections = 0;
		double connections_density = 1.0;
		float initial_search_extent = 0.0f;
		int32_t initial_search_radius_i = 0;
		int32_t initial_search_radius_j = 0;
		int32_t initial_search_radius_k = 0;
		float refractory_strength = 0.0f;
		float exponential_factor = 0.0f;
		float structural_plasticity = 0.0f;
		float lambda = 0.0f;
		float nu = 0.0f;
		float mu = 0.0f;
		float input_gain = 0.0f;
		float mean_initial_incoming = 0.0f;
		int32_t role_selection_gap = 0;
		bool role_selection_valid = false;
		bool role_cleanup_valid = false;
		bool valid = false;
		int64_t first_pass_connections = 0;
		int64_t second_pass_connections = 0;
		IsolationCounts isolated_before_rescue;
		IsolationCounts isolated_after_rescue;
		RescueFailureCauses rescue_failure_causes;
		int64_t rescue_candidates_receiver = 0;
		int64_t rescue_candidates_input = 0;
		std::vector<NativeNeuron> neurons;
	};

	void _create_neurons(NativeBrain &brain, const GenomeNative &genome, const Callable &progress_callback) const;
	bool _set_input_output_neurons(NativeBrain &brain, int32_t num_inputs, int32_t num_outputs, const Callable &progress_callback) const;
	void _create_connections(NativeBrain &brain, const std::vector<int32_t> &gene, const std::vector<int32_t> &search_gene, const Callable &progress_callback) const;
	bool _clear_invalid_connections(NativeBrain &brain, const std::vector<int32_t> &gene, const std::vector<int32_t> &input_influence_gene, const Callable &progress_callback) const;
	bool _rescue_isolated_neurons(NativeBrain &brain, const std::vector<int32_t> &gene, const std::vector<int32_t> &neuron_by_spatial_index, std::vector<int32_t> &incoming_connections, const Callable &progress_callback) const;
	Vector3 _modulatory_dynamics(const std::vector<int32_t> &gene) const;
	Dictionary _to_dictionary(const NativeBrain &brain, const Callable &progress_callback) const;
	void _report_progress(const Callable &progress_callback, int32_t percent) const;

	static int32_t _spatial_index(const Vector3i &position, int32_t I, int32_t IJ);
	static Vector3i _position_from_spatial_index(int32_t spatial_index, int32_t I, int32_t J);
};

}

#endif
