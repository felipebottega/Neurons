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
			const Callable &progress_callback = Callable());

private:
	struct NativeConnection {
		int32_t target_spatial_index = 0;
		double weight = 0.0;
		double eligibility_trace = 0.0;
	};

	struct NativeNeuron {
		Vector3i position;
		bool input = false;
		bool output = false;
		double activation_threshold = 0.0;
		double decay_factor = 0.0;
		double retention_factor = 0.0;
		int32_t polarity_factor = 1;
		double hebbian_plasticity_rate = 0.0;
		double modulatory_release_factor = 0.0;
		double modulatory_sensitivity = 0.0;
		double modulatory_field = 0.0;
		double connections_value = 0.0;
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
		double refractory_strength = 0.0;
		double exponential_factor = 0.0;
		double structural_plasticity = 0.0;
		double lambda = 0.0;
		double nu = 0.0;
		double mu = 0.0;
		double input_gain = 0.0;
		bool valid = false;
		IsolationCounts isolated_before_rescue;
		IsolationCounts isolated_after_rescue;
		RescueFailureCauses rescue_failure_causes;
		std::vector<NativeNeuron> neurons;
	};

	void _create_neurons(NativeBrain &brain, const GenomeNative &genome, const Callable &progress_callback) const;
	void _set_input_output_neurons(NativeBrain &brain, int32_t num_inputs, int32_t num_outputs, const Callable &progress_callback) const;
	void _create_connections(NativeBrain &brain, const std::vector<int32_t> &gene, const Callable &progress_callback) const;
	bool _clear_invalid_connections(NativeBrain &brain, const std::vector<int32_t> &gene, const std::vector<int32_t> &input_influence_gene, const Callable &progress_callback) const;
	bool _rescue_isolated_neurons(NativeBrain &brain, const std::vector<int32_t> &gene, const std::vector<int32_t> &neuron_by_spatial_index, std::vector<int32_t> &incoming_connections, const Callable &progress_callback) const;
	Vector3 _modulatory_dynamics(const std::vector<int32_t> &gene) const;
	Dictionary _to_dictionary(const NativeBrain &brain) const;
	void _report_progress(const Callable &progress_callback, int32_t percent) const;

	static int32_t _spatial_index(const Vector3i &position, int32_t I, int32_t IJ);
	static Vector3i _position_from_spatial_index(int32_t spatial_index, int32_t I, int32_t J);
};

}

#endif
