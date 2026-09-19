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
	double elapsed_time_seconds = 0.0;
	double delta_time_seconds = 0.0;
	double mean_weight = 0.0;
	double mean_signal = 0.0;
	int64_t total_connections = 0;
	int64_t mean_outgoing_connections = 0;
	int32_t min_out_degree = 0;
	int32_t max_out_degree = 0;
	int64_t added_connections_last_iteration = 0;
	int64_t total_connections_formed = 0;
	int64_t total_structural_plasticity_connections_formed = 0;
	int64_t total_homeostatic_connections_formed = 0;
	int64_t total_connections_destroyed = 0;
	double mean_modulatory_release = 0.0;
	double mean_modulatory_field = 0.0;
	double mean_effective_modulation = 0.0;
	double mean_eligibility_trace = 0.0;

	int32_t neuron_count = 0;
	int32_t size_i = 0;
	int32_t size_j = 0;
	int32_t size_k = 0;
	int32_t plane_size = 0;
	double structural_plasticity = 0.0;
	double structural_new_weight = 0.0;
	double modulatory_persistence = 0.0;
	double modulatory_spread = 0.0;
	double one_minus_modulatory_spread = 1.0;
	double trace_persistence = 0.0;
	double one_minus_trace_persistence = 1.0;
	double refractory_strength = 0.0;
	double inverse_exponential_factor = 1.0;
	double homeostatic_low_state = 0.0;
	double homeostatic_high_state = 0.0;
	double input_gain = 1.0;
	double input_min_weight = 0.5;

	std::vector<uint8_t> input_flags;
	std::vector<uint8_t> output_flags;
	std::vector<uint8_t> internal_flags;
	std::vector<double> activation_thresholds;
	std::vector<double> polarity_factors;
	std::vector<double> hebbian_rates;
	std::vector<double> decay_rates;
	std::vector<double> retention_factors;
	std::vector<double> modulatory_release_factors;
	std::vector<double> modulatory_sensitivities;
	std::vector<int32_t> spatial_indices;
	std::vector<int32_t> z_indices;

	std::vector<double> fatigues;
	std::vector<double> fatigues_used;
	std::vector<double> effective_thresholds_used;
	std::vector<double> neuron_states;
	std::vector<double> neuron_states_used;
	std::vector<double> normalized_signals;
	std::vector<uint8_t> firing_states;
	std::vector<double> activity_traces;
	std::vector<double> incoming_signals;
	std::vector<int32_t> incoming_connection_counts;
	std::vector<double> pending_input_signals;
	std::vector<double> input_signals_used;
	std::vector<double> output_signals;
	std::vector<double> modulatory_releases;
	std::vector<double> modulatory_fields;
	std::vector<double> next_modulatory_fields;
	std::vector<double> effective_modulations;

	std::vector<int32_t> input_neuron_indices;
	std::vector<int32_t> output_slot_by_neuron_index;
	std::vector<int32_t> neuron_index_by_spatial_index;
	std::vector<int32_t> modulatory_neighbor_indices;
	std::vector<int32_t> modulatory_neighbor_counts;

	struct Connection {
		int32_t target_index = 0;
		double weight = 0.0;
		double eligibility_trace = 0.0;
	};

	std::vector<std::vector<Connection>> connections;
	std::vector<std::vector<Connection>> initial_connections;
	std::vector<double> initial_modulatory_fields;

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

	Dictionary _calculate_statistics(const std::vector<double> &values) const;
	void _resize_neuron_arrays(int32_t count);
	void _cache_modulatory_neighbors();
	void _setup_input_output_indices(int32_t input_count, int32_t output_count);
	void _build_runtime_connections(const Array &neurons);
	void _store_initial_runtime_state();
	int32_t _propagate_signals();
	bool _replace_last_incoming_connection(int32_t target_index, double new_weight);
	void _update_modulatory_field();
	void _apply_neuron_state_update();
	int32_t _apply_structural_plasticity();
	int32_t _random_structural_target_index(int32_t source_index);
	void _append_connection(int32_t source_index, int32_t target_index, double weight, double eligibility);
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
