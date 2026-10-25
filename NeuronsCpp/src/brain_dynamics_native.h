#ifndef BRAIN_DYNAMICS_NATIVE_H
#define BRAIN_DYNAMICS_NATIVE_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace godot {

class BrainDynamicsNative : public RefCounted {
	GDCLASS(BrainDynamicsNative, RefCounted);

protected:
	static void _bind_methods();

public:
	BrainDynamicsNative();
	~BrainDynamicsNative();

	void set_brain(const Dictionary &brain);
	void reset();
	void clear();
	void add_input_signal(int64_t input_index, double input_signal);
	int64_t advance(double delta_seconds);
	Dictionary get_brain_snapshot() const;
	Dictionary get_viewer_state() const;

	void set_profiling_enabled(bool enabled);
	bool get_profiling_enabled() const;

	double get_output_signal(int64_t output_index) const;
	int64_t get_total_connections() const;
	Dictionary get_neuron_statistics(const String &parameter) const;
	double get_mean_weight() const;
	int64_t get_total_connections_formed() const;
	int64_t get_total_connections_destroyed() const;

	int64_t get_profile_total_us() const;
	int64_t get_profile_prepare_and_firing_us() const;
	int64_t get_profile_modulatory_field_us() const;
	int64_t get_profile_connections_us() const;
	int64_t get_profile_homeostatic_us() const;
	int64_t get_profile_propagation_statistics_us() const;
	int64_t get_profile_state_update_us() const;
	int64_t get_profile_structural_plasticity_us() const;

private:
	Dictionary brain_data;

	int64_t iteration_count = 0;
	float elapsed_time_seconds = 0.0f;
	float delta_time_seconds = 0.0f;
	float mean_weight = 0.0f;
	float mean_signal = 0.0f;
	int64_t total_connections = 0;
	int64_t mean_outgoing_connections = 0;
	int32_t min_out_degree = 0;
	int32_t max_out_degree = 0;
	int64_t added_connections_last_iteration = 0;
	int64_t total_connections_formed = 0;
	int64_t total_structural_plasticity_connections_formed = 0;
	int64_t total_homeostatic_connections_formed = 0;
	int64_t total_connections_destroyed = 0;
	float mean_modulatory_release = 0.0f;
	float mean_modulatory_field = 0.0f;
	float mean_effective_modulation = 0.0f;
	float mean_eligibility_trace = 0.0f;

	int32_t neuron_count = 0;
	int32_t size_i = 0;
	int32_t size_j = 0;
	int32_t size_k = 0;
	int32_t plane_size = 0;
	float structural_plasticity = 0.0f;
	float structural_new_weight = 0.0f;
	float modulatory_persistence = 0.0f;
	float modulatory_spread = 0.0f;
	float one_minus_modulatory_spread = 1.0f;
	float trace_persistence = 0.0f;
	float one_minus_trace_persistence = 1.0f;
	float refractory_strength = 0.0f;
	float inverse_exponential_factor = 1.0f;
	float homeostatic_low_state = 0.0f;
	float homeostatic_high_state = 0.0f;
	float input_gain = 1.0f;
	float input_min_weight = 0.5f;

	std::vector<uint8_t> input_flags;
	std::vector<uint8_t> output_flags;
	std::vector<uint8_t> internal_flags;
	std::vector<float> activation_thresholds;
	std::vector<float> polarity_factors;
	std::vector<float> hebbian_rates;
	std::vector<float> decay_rates;
	std::vector<float> retention_factors;
	std::vector<float> modulatory_release_factors;
	std::vector<float> modulatory_sensitivities;
	std::vector<int32_t> spatial_indices;
	std::vector<int32_t> z_indices;

	std::vector<float> fatigues;
	std::vector<float> fatigues_used;
	std::vector<float> effective_thresholds_used;
	std::vector<float> neuron_states;
	std::vector<float> neuron_states_used;
	std::vector<float> normalized_signals;
	std::vector<uint8_t> firing_states;
	std::vector<float> activity_traces;
	std::vector<float> incoming_signals;
	std::vector<int32_t> incoming_connection_counts;
	std::vector<float> pending_input_signals;
	std::vector<float> input_signals_used;
	std::vector<float> output_signals;
	std::vector<float> modulatory_releases;
	std::vector<float> modulatory_fields;
	std::vector<float> next_modulatory_fields;
	std::vector<float> effective_modulations;

	std::vector<int32_t> input_neuron_indices;
	std::vector<int32_t> output_slot_by_neuron_index;
	std::vector<int32_t> neuron_index_by_spatial_index;
	std::vector<int32_t> modulatory_neighbor_indices;
	std::vector<int32_t> modulatory_neighbor_counts;

	struct Connection {
		int32_t target_index = 0;
		float weight = 0.0f;
		float eligibility_trace = 0.0f;
	};

	std::vector<std::vector<Connection>> connections;
	std::vector<std::vector<Connection>> initial_connections;
	std::vector<float> initial_modulatory_fields;

	int32_t outgoing_neuron_count = 0;
	std::vector<int32_t> out_degree_histogram;
	std::vector<int32_t> orphaned_targets;

	std::mt19937_64 structural_rng;
	uint64_t initial_structural_rng_seed = 0;

	bool profiling_enabled = false;
	int64_t profile_total_us = 0;
	int64_t profile_propagate_us = 0;
	int64_t profile_prepare_and_firing_us = 0;
	int64_t profile_modulatory_field_us = 0;
	int64_t profile_connections_us = 0;
	int64_t profile_homeostatic_us = 0;
	int64_t profile_propagation_statistics_us = 0;
	int64_t profile_state_update_us = 0;
	int64_t profile_structural_plasticity_us = 0;

	Dictionary _calculate_statistics(const std::vector<float> &values) const;
	void _resize_neuron_arrays(int32_t count);
	void _cache_modulatory_neighbors();
	void _setup_input_output_indices(int32_t input_count, int32_t output_count);
	void _build_runtime_connections(const Array &neurons);
	void _store_initial_runtime_state();
	int32_t _propagate_signals();
	bool _replace_last_incoming_connection(int32_t target_index, float new_weight);
	void _update_modulatory_field();
	void _apply_neuron_state_update();
	int32_t _apply_structural_plasticity();
	int32_t _random_structural_target_index(int32_t source_index);
	void _append_connection(int32_t source_index, int32_t target_index, float weight, float eligibility);
	void _remove_connection_at(int32_t source_index, int32_t connection_index, int32_t target_index);
	void _update_initial_statistics();
	void _update_modulatory_statistics();
	void _record_out_degree_change(int32_t source_index, int32_t old_degree, int32_t new_degree);
	void _ensure_degree_histogram_size(int32_t degree);
	void _rebuild_degree_extremes();
	void _refresh_mean_outgoing_connections();
	static int64_t _ticks_usec();
};

}

#endif
