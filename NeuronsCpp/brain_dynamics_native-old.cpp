#include "brain_dynamics_native.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/vector3i.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

using namespace godot;

void BrainDynamicsNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_brain", "brain"), &BrainDynamicsNative::set_brain);
	ClassDB::bind_method(D_METHOD("reset"), &BrainDynamicsNative::reset);
	ClassDB::bind_method(D_METHOD("clear"), &BrainDynamicsNative::clear);
	ClassDB::bind_method(D_METHOD("add_input_signal", "input_index", "input_signal"), &BrainDynamicsNative::add_input_signal);
	ClassDB::bind_method(D_METHOD("advance", "delta_seconds"), &BrainDynamicsNative::advance);
	ClassDB::bind_method(D_METHOD("get_brain_snapshot"), &BrainDynamicsNative::get_brain_snapshot);

	ClassDB::bind_method(D_METHOD("set_profiling_enabled", "enabled"), &BrainDynamicsNative::set_profiling_enabled);
	ClassDB::bind_method(D_METHOD("get_profiling_enabled"), &BrainDynamicsNative::get_profiling_enabled);

	ClassDB::bind_method(D_METHOD("get_output_signal", "output_index"), &BrainDynamicsNative::get_output_signal);
	ClassDB::bind_method(D_METHOD("get_total_connections"), &BrainDynamicsNative::get_total_connections);
	ClassDB::bind_method(D_METHOD("get_mean_weight"), &BrainDynamicsNative::get_mean_weight);
	ClassDB::bind_method(D_METHOD("get_total_connections_formed"), &BrainDynamicsNative::get_total_connections_formed);
	ClassDB::bind_method(D_METHOD("get_total_connections_destroyed"), &BrainDynamicsNative::get_total_connections_destroyed);

	ClassDB::bind_method(D_METHOD("get_profile_total_us"), &BrainDynamicsNative::get_profile_total_us);
	ClassDB::bind_method(D_METHOD("get_profile_prepare_and_firing_us"), &BrainDynamicsNative::get_profile_prepare_and_firing_us);
	ClassDB::bind_method(D_METHOD("get_profile_modulatory_field_us"), &BrainDynamicsNative::get_profile_modulatory_field_us);
	ClassDB::bind_method(D_METHOD("get_profile_connections_us"), &BrainDynamicsNative::get_profile_connections_us);
	ClassDB::bind_method(D_METHOD("get_profile_homeostatic_us"), &BrainDynamicsNative::get_profile_homeostatic_us);
	ClassDB::bind_method(D_METHOD("get_profile_propagation_statistics_us"), &BrainDynamicsNative::get_profile_propagation_statistics_us);
	ClassDB::bind_method(D_METHOD("get_profile_state_update_us"), &BrainDynamicsNative::get_profile_state_update_us);
	ClassDB::bind_method(D_METHOD("get_profile_structural_plasticity_us"), &BrainDynamicsNative::get_profile_structural_plasticity_us);
}

BrainDynamicsNative::BrainDynamicsNative() {
}

BrainDynamicsNative::~BrainDynamicsNative() {
}

void BrainDynamicsNative::set_brain(const Dictionary &brain) {
	brain_data = brain;
	iteration_count = 0;
	elapsed_time_seconds = 0.0;
	delta_time_seconds = 0.0;
	mean_signal = 0.0;
	mean_modulatory_release = 0.0;
	mean_modulatory_field = 0.0;
	mean_effective_modulation = 0.0;
	mean_eligibility_trace = 0.0;

	Array neurons = brain_data["neurons"];
	Vector3i brain_size = brain_data["size"];
	neuron_count = static_cast<int32_t>(neurons.size());
	size_i = brain_size.x;
	size_j = brain_size.y;
	size_k = brain_size.z;
	plane_size = size_i * size_j;
	structural_plasticity = static_cast<double>(brain_data["structural_plasticity"]);
	structural_new_weight = structural_plasticity * 0.25;
	modulatory_persistence = static_cast<double>(brain_data["lambda"]);
	modulatory_spread = static_cast<double>(brain_data["nu"]);
	one_minus_modulatory_spread = 1.0 - modulatory_spread;
	trace_persistence = static_cast<double>(brain_data["mu"]);
	one_minus_trace_persistence = 1.0 - trace_persistence;
	refractory_strength = static_cast<double>(brain_data["refractory_strength"]);
	inverse_exponential_factor = 1.0 / static_cast<double>(brain_data["exponential_factor"]);
	input_gain = static_cast<double>(brain_data["input_gain"]);
	input_min_weight = 1.0 / (1.0 + input_gain);

	_resize_neuron_arrays(neuron_count);

	int32_t input_count = 0;
	int32_t output_count = 0;

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		Dictionary neuron = neurons[neuron_index];
		Vector3i neuron_position = neuron["position"];
		const int32_t spatial_index = neuron_position.x + size_i * neuron_position.y + plane_size * neuron_position.z;
		const bool is_input = static_cast<bool>(neuron["input"]);
		const bool is_output = static_cast<bool>(neuron["output"]);

		spatial_indices[neuron_index] = spatial_index;
		z_indices[neuron_index] = neuron_position.z;
		neuron_index_by_spatial_index[spatial_index] = neuron_index;
		input_flags[neuron_index] = is_input ? 1 : 0;
		output_flags[neuron_index] = is_output ? 1 : 0;
		internal_flags[neuron_index] = (!is_input && !is_output) ? 1 : 0;
		activation_thresholds[neuron_index] = is_input ? 0.0 : static_cast<double>(neuron["activation_threshold"]);
		polarity_factors[neuron_index] = static_cast<double>(neuron["polarity_factor"]);
		hebbian_rates[neuron_index] = static_cast<double>(neuron["hebbian_plasticity_rate"]);
		decay_rates[neuron_index] = is_input ? 0.0 : std::exp(-12.0 / static_cast<double>(neuron["decay_factor"]));
		modulatory_release_factors[neuron_index] = static_cast<double>(neuron["modulatory_release_factor"]);
		modulatory_sensitivities[neuron_index] = static_cast<double>(neuron["modulatory_sensitivity"]);
		modulatory_fields[neuron_index] = static_cast<double>(neuron["modulatory_field"]);
		effective_modulations[neuron_index] = modulatory_sensitivities[neuron_index] * (1.0 - std::exp(-modulatory_fields[neuron_index]));

		if (is_input) {
			neuron["retention_factor"] = 0.0;
			neurons[neuron_index] = neuron;
			retention_factors[neuron_index] = 0.0;
			++input_count;
		} else {
			retention_factors[neuron_index] = static_cast<double>(neuron["retention_factor"]);
			if (is_output) {
				++output_count;
			}
		}
	}

	brain_data["neurons"] = neurons;
	_cache_modulatory_neighbors();
	_setup_input_output_indices(input_count, output_count);
	_build_runtime_connections(neurons);
	_store_initial_runtime_state();

	std::random_device random_device;
	initial_structural_rng_seed = (static_cast<uint64_t>(random_device()) << 32) ^ static_cast<uint64_t>(random_device());
	structural_rng.seed(initial_structural_rng_seed);
	added_connections_last_iteration = 0;
	total_connections_formed = 0;
	total_structural_plasticity_connections_formed = 0;
	total_homeostatic_connections_formed = 0;
	total_connections_destroyed = 0;

	_update_modulatory_statistics();
	_update_initial_statistics();
}

void BrainDynamicsNative::_resize_neuron_arrays(int32_t count) {
	input_flags.assign(count, 0);
	output_flags.assign(count, 0);
	internal_flags.assign(count, 0);
	activation_thresholds.assign(count, 0.0);
	polarity_factors.assign(count, 0.0);
	hebbian_rates.assign(count, 0.0);
	decay_rates.assign(count, 0.0);
	retention_factors.assign(count, 0.0);
	modulatory_release_factors.assign(count, 0.0);
	modulatory_sensitivities.assign(count, 0.0);
	spatial_indices.assign(count, 0);
	z_indices.assign(count, 0);

	fatigues.assign(count, 0.0);
	fatigues_used.assign(count, 0.0);
	effective_thresholds_used.assign(count, 0.0);
	neuron_states.assign(count, 0.0);
	neuron_states_used.assign(count, 0.0);
	normalized_signals.assign(count, 0.0);
	firing_states.assign(count, 0);
	activity_traces.assign(count, 0.0);
	incoming_signals.assign(count, 0.0);
	incoming_connection_counts.assign(count, 0);
	modulatory_releases.assign(count, 0.0);
	modulatory_fields.assign(count, 0.0);
	next_modulatory_fields.assign(count, 0.0);
	effective_modulations.assign(count, 0.0);

	output_slot_by_neuron_index.assign(count, -1);
	neuron_index_by_spatial_index.assign(count, 0);
	modulatory_neighbor_indices.assign(static_cast<size_t>(count) * 6, 0);
	modulatory_neighbor_counts.assign(count, 0);

	connections.assign(count, {});

	orphaned_targets.clear();
}

void BrainDynamicsNative::_cache_modulatory_neighbors() {
	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		const int32_t spatial_index = spatial_indices[neuron_index];
		const int32_t neuron_z = z_indices[neuron_index];
		const int32_t within_plane = spatial_index - plane_size * neuron_z;
		const int32_t neuron_y = within_plane / size_i;
		const int32_t neuron_x = within_plane - size_i * neuron_y;
		const int32_t neighbor_offset = neuron_index * 6;
		int32_t neighbor_count = 0;

		if (neuron_x > 0) {
			modulatory_neighbor_indices[neighbor_offset + neighbor_count++] = neuron_index_by_spatial_index[spatial_index - 1];
		}
		if (neuron_x + 1 < size_i) {
			modulatory_neighbor_indices[neighbor_offset + neighbor_count++] = neuron_index_by_spatial_index[spatial_index + 1];
		}
		if (neuron_y > 0) {
			modulatory_neighbor_indices[neighbor_offset + neighbor_count++] = neuron_index_by_spatial_index[spatial_index - size_i];
		}
		if (neuron_y + 1 < size_j) {
			modulatory_neighbor_indices[neighbor_offset + neighbor_count++] = neuron_index_by_spatial_index[spatial_index + size_i];
		}
		if (neuron_z > 0) {
			modulatory_neighbor_indices[neighbor_offset + neighbor_count++] = neuron_index_by_spatial_index[spatial_index - plane_size];
		}
		if (neuron_z + 1 < size_k) {
			modulatory_neighbor_indices[neighbor_offset + neighbor_count++] = neuron_index_by_spatial_index[spatial_index + plane_size];
		}

		modulatory_neighbor_counts[neuron_index] = neighbor_count;
	}
}

void BrainDynamicsNative::_setup_input_output_indices(int32_t input_count, int32_t output_count) {
	input_neuron_indices.assign(input_count, 0);
	pending_input_signals.assign(input_count, 0.0);
	input_signals_used.assign(input_count, 0.0);
	output_signals.assign(output_count, 0.0);

	int32_t input_slot = 0;
	int32_t output_slot = 0;

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		if (input_flags[neuron_index] != 0) {
			input_neuron_indices[input_slot++] = neuron_index;
		} else if (output_flags[neuron_index] != 0) {
			output_slot_by_neuron_index[neuron_index] = output_slot++;
		}
	}
}

void BrainDynamicsNative::_build_runtime_connections(const Array &neurons) {
	std::fill(incoming_connection_counts.begin(), incoming_connection_counts.end(), 0);
	connections.assign(neuron_count, {});

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		Dictionary source_neuron = neurons[source_index];
		Array source_connections = source_neuron["connections"];
		auto &native_connections = connections[source_index];
		native_connections.reserve(static_cast<size_t>(source_connections.size()));

		for (int32_t connection_index = 0; connection_index < source_connections.size(); ++connection_index) {
			Dictionary connection = source_connections[connection_index];
			Vector3i target_position = connection["target"];
			const int32_t target_spatial_index = target_position.x + size_i * target_position.y + plane_size * target_position.z;
			const int32_t target_index = neuron_index_by_spatial_index[target_spatial_index];

			native_connections.push_back({
				target_index,
				static_cast<double>(connection["weight"]),
				static_cast<double>(connection["eligibility_trace"])
			});
			++incoming_connection_counts[target_index];
		}
	}
}

void BrainDynamicsNative::_store_initial_runtime_state() {
	initial_connections = connections;
	initial_modulatory_fields = modulatory_fields;
}

void BrainDynamicsNative::reset() {
	if (brain_data.is_empty()) {
		return;
	}

	std::fill(incoming_connection_counts.begin(), incoming_connection_counts.end(), 0);
	connections = initial_connections;

	for (const auto &source_connections : connections) {
		for (const Connection &connection : source_connections) {
			++incoming_connection_counts[connection.target_index];
		}
	}

	iteration_count = 0;
	elapsed_time_seconds = 0.0;
	delta_time_seconds = 0.0;
	mean_signal = 0.0;
	mean_modulatory_release = 0.0;
	mean_modulatory_field = 0.0;
	mean_effective_modulation = 0.0;
	mean_eligibility_trace = 0.0;
	added_connections_last_iteration = 0;
	total_connections_formed = 0;
	total_structural_plasticity_connections_formed = 0;
	total_homeostatic_connections_formed = 0;
	total_connections_destroyed = 0;
	structural_rng.seed(initial_structural_rng_seed);

	std::fill(fatigues.begin(), fatigues.end(), 0.0);
	std::fill(fatigues_used.begin(), fatigues_used.end(), 0.0);
	std::fill(neuron_states.begin(), neuron_states.end(), 0.0);
	std::fill(neuron_states_used.begin(), neuron_states_used.end(), 0.0);
	std::fill(normalized_signals.begin(), normalized_signals.end(), 0.0);
	std::fill(firing_states.begin(), firing_states.end(), 0);
	std::fill(activity_traces.begin(), activity_traces.end(), 0.0);
	std::fill(incoming_signals.begin(), incoming_signals.end(), 0.0);
	std::fill(pending_input_signals.begin(), pending_input_signals.end(), 0.0);
	std::fill(input_signals_used.begin(), input_signals_used.end(), 0.0);
	std::fill(output_signals.begin(), output_signals.end(), 0.0);
	std::fill(modulatory_releases.begin(), modulatory_releases.end(), 0.0);
	std::fill(next_modulatory_fields.begin(), next_modulatory_fields.end(), 0.0);

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		effective_thresholds_used[neuron_index] = activation_thresholds[neuron_index];
		modulatory_fields[neuron_index] = initial_modulatory_fields[neuron_index];
		effective_modulations[neuron_index] = modulatory_sensitivities[neuron_index] * (1.0 - std::exp(-modulatory_fields[neuron_index]));
	}

	_update_modulatory_statistics();
	_update_initial_statistics();
}

void BrainDynamicsNative::add_input_signal(int64_t input_index, double input_signal) {
	pending_input_signals[static_cast<size_t>(input_index)] = input_signal;
}

int64_t BrainDynamicsNative::advance(double delta_seconds) {
	delta_time_seconds = delta_seconds;
	elapsed_time_seconds += delta_seconds;
	++iteration_count;
	added_connections_last_iteration = 0;

	if (!profiling_enabled) {
		const int32_t removed_connections = _propagate_signals();
		_apply_neuron_state_update();
		added_connections_last_iteration += _apply_structural_plasticity();
		return removed_connections;
	}

	const int64_t total_start_us = _ticks_usec();
	int64_t stage_start_us = total_start_us;
	const int32_t removed_connections = _propagate_signals();
	int64_t current_us = _ticks_usec();
	profile_propagate_us = current_us - stage_start_us;
	stage_start_us = current_us;

	_apply_neuron_state_update();
	current_us = _ticks_usec();
	profile_state_update_us = current_us - stage_start_us;
	stage_start_us = current_us;

	added_connections_last_iteration += _apply_structural_plasticity();
	current_us = _ticks_usec();
	profile_structural_plasticity_us = current_us - stage_start_us;
	profile_total_us = current_us - total_start_us;

	return removed_connections;
}

int32_t BrainDynamicsNative::_propagate_signals() {
	double signal_sum = 0.0;
	double modulatory_release_sum = 0.0;
	int32_t removed_connections = 0;
	int32_t replacement_connections = 0;
	int64_t profile_stage_start_us = profiling_enabled ? _ticks_usec() : 0;

	std::fill(incoming_signals.begin(), incoming_signals.end(), 0.0);
	std::fill(normalized_signals.begin(), normalized_signals.end(), 0.0);
	std::fill(firing_states.begin(), firing_states.end(), 0);
	std::fill(output_signals.begin(), output_signals.end(), 0.0);
	std::fill(modulatory_releases.begin(), modulatory_releases.end(), 0.0);
	std::fill(input_signals_used.begin(), input_signals_used.end(), 0.0);
	orphaned_targets.clear();

	for (int32_t input_index = 0; input_index < static_cast<int32_t>(input_neuron_indices.size()); ++input_index) {
		const double input_signal = pending_input_signals[input_index];
		input_signals_used[input_index] = input_signal;
		neuron_states[input_neuron_indices[input_index]] = input_signal;
		pending_input_signals[input_index] = 0.0;
	}

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		const bool is_input = input_flags[neuron_index] != 0;

		if (!is_input) {
			const double current_fatigue = fatigues[neuron_index];
			fatigues_used[neuron_index] = current_fatigue;
			effective_thresholds_used[neuron_index] = activation_thresholds[neuron_index] * (1.0 + current_fatigue);
			activity_traces[neuron_index] *= trace_persistence;
		}

		const double current_state = neuron_states[neuron_index];
		neuron_states_used[neuron_index] = current_state;

		if (current_state <= 0.0) {
			if (!is_input) {
				fatigues[neuron_index] *= 0.5;
			}
			continue;
		}

		const double normalized_signal = 1.0 - std::exp(-current_state * inverse_exponential_factor);
		normalized_signals[neuron_index] = normalized_signal;
		signal_sum += normalized_signal;

		if (!is_input) {
			activity_traces[neuron_index] += one_minus_trace_persistence * normalized_signal;
		}

		if (is_input) {
			firing_states[neuron_index] = 1;
			const double input_release = modulatory_release_factors[neuron_index] * normalized_signal;
			modulatory_releases[neuron_index] = input_release;
			modulatory_release_sum += input_release;
			continue;
		}

		if (normalized_signal <= effective_thresholds_used[neuron_index]) {
			fatigues[neuron_index] *= 0.5;
			continue;
		}

		firing_states[neuron_index] = 1;
		const double release = modulatory_release_factors[neuron_index] * normalized_signal;
		modulatory_releases[neuron_index] = release;
		modulatory_release_sum += release;
		fatigues[neuron_index] = refractory_strength;

		const int32_t output_slot = output_slot_by_neuron_index[neuron_index];
		if (output_slot >= 0) {
			output_signals[output_slot] = normalized_signal;
		}
	}

	if (profiling_enabled) {
		const int64_t profile_now_us = _ticks_usec();
		profile_prepare_and_firing_us = profile_now_us - profile_stage_start_us;
		profile_stage_start_us = profile_now_us;
	}

	_update_modulatory_field();

	if (profiling_enabled) {
		const int64_t profile_now_us = _ticks_usec();
		profile_modulatory_field_us = profile_now_us - profile_stage_start_us;
		profile_stage_start_us = profile_now_us;
	}

	const double trace_keep = trace_persistence;
	const double trace_add = one_minus_trace_persistence;
	const double cached_input_gain = input_gain;
	const double cached_input_min_weight = input_min_weight;

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		auto &source_connections = connections[source_index];
		int32_t connection_index = static_cast<int32_t>(source_connections.size()) - 1;

		if (connection_index < 0) {
			continue;
		}

		const bool source_fired = firing_states[source_index] != 0;
		const double source_normalized_signal = normalized_signals[source_index];
		const bool source_is_input = input_flags[source_index] != 0;
		const double hebbian_rate = hebbian_rates[source_index];

		if (source_is_input) {
			const double source_signal = cached_input_gain * neuron_states[source_index];
			const double transmission_scale = polarity_factors[source_index] * source_signal;

			if (source_fired) {
				const double eligibility_source_scale = trace_add * source_normalized_signal;
				while (connection_index >= 0) {
					Connection &connection = source_connections[connection_index];
					const int32_t target_index = connection.target_index;
					double weight = connection.weight;
					incoming_signals[target_index] += transmission_scale * weight;
					weight += hebbian_rate * (1.0 - weight);
					const double eligibility = trace_keep * connection.eligibility_trace + eligibility_source_scale * normalized_signals[target_index];
					connection.eligibility_trace = eligibility;

					const double modulation = effective_modulations[target_index];
					if (modulation >= 0.0) {
						weight += modulation * eligibility * (1.0 - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					weight = std::max(weight, cached_input_min_weight);
					connection.weight = weight;
					--connection_index;
				}
			} else {
				while (connection_index >= 0) {
					Connection &connection = source_connections[connection_index];
					const int32_t target_index = connection.target_index;
					double weight = connection.weight;
					const double eligibility = trace_keep * connection.eligibility_trace;
					connection.eligibility_trace = eligibility;

					const double modulation = effective_modulations[target_index];
					if (modulation >= 0.0) {
						weight += modulation * eligibility * (1.0 - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					weight = std::max(weight, cached_input_min_weight);
					connection.weight = weight;
					--connection_index;
				}
			}
		} else {
			const double decrement = decay_rates[source_index];
			const double transmission_scale = polarity_factors[source_index] * source_normalized_signal;

			if (source_fired) {
				const double eligibility_source_scale = trace_add * source_normalized_signal;
				while (connection_index >= 0) {
					Connection &connection = source_connections[connection_index];
					const int32_t target_index = connection.target_index;
					double weight = connection.weight;
					incoming_signals[target_index] += transmission_scale * weight;
					weight += hebbian_rate * (1.0 - weight);
					const double eligibility = trace_keep * connection.eligibility_trace + eligibility_source_scale * normalized_signals[target_index];
					connection.eligibility_trace = eligibility;

					const double modulation = effective_modulations[target_index];
					if (modulation >= 0.0) {
						weight += modulation * eligibility * (1.0 - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					double connection_decrement = decrement;
					if (output_flags[target_index] != 0) {
						connection_decrement *= activity_traces[target_index];
					}
					weight = std::max(weight - connection_decrement, 0.0);

					if (weight <= 0.0) {
						_remove_connection_at(source_index, connection_index, target_index);
						++removed_connections;
						if (incoming_connection_counts[target_index] == 0 && input_flags[target_index] == 0) {
							orphaned_targets.push_back(target_index);
						}
					} else {
						connection.weight = weight;
					}

					--connection_index;
				}
			} else {
				while (connection_index >= 0) {
					Connection &connection = source_connections[connection_index];
					const int32_t target_index = connection.target_index;
					double weight = connection.weight;
					const double eligibility = trace_keep * connection.eligibility_trace;
					connection.eligibility_trace = eligibility;

					const double modulation = effective_modulations[target_index];
					if (modulation >= 0.0) {
						weight += modulation * eligibility * (1.0 - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					double connection_decrement = decrement;
					if (output_flags[target_index] != 0) {
						connection_decrement *= activity_traces[target_index];
					}
					weight = std::max(weight - connection_decrement, 0.0);

					if (weight <= 0.0) {
						_remove_connection_at(source_index, connection_index, target_index);
						++removed_connections;
						if (incoming_connection_counts[target_index] == 0 && input_flags[target_index] == 0) {
							orphaned_targets.push_back(target_index);
						}
					} else {
						connection.weight = weight;
					}

					--connection_index;
				}
			}
		}
	}

	if (profiling_enabled) {
		const int64_t profile_now_us = _ticks_usec();
		profile_connections_us = profile_now_us - profile_stage_start_us;
		profile_stage_start_us = profile_now_us;
	}

	for (int32_t target_index : orphaned_targets) {
		if (_replace_last_incoming_connection(target_index, structural_new_weight)) {
			++replacement_connections;
		}
	}

	if (profiling_enabled) {
		const int64_t profile_now_us = _ticks_usec();
		profile_homeostatic_us = profile_now_us - profile_stage_start_us;
		profile_stage_start_us = profile_now_us;
	}

	total_connections += replacement_connections - removed_connections;
	if (removed_connections > 0) {
		total_connections_destroyed += removed_connections;
	}

	if (replacement_connections > 0) {
		total_connections_formed += replacement_connections;
		total_homeostatic_connections_formed += replacement_connections;
		added_connections_last_iteration += replacement_connections;
	}

	if (removed_connections > 0 || replacement_connections > 0) {
		_refresh_mean_outgoing_connections();
	}

	mean_signal = neuron_count == 0 ? 0.0 : signal_sum / static_cast<double>(neuron_count);
	mean_modulatory_release = neuron_count == 0 ? 0.0 : modulatory_release_sum / static_cast<double>(neuron_count);

	if (profiling_enabled) {
		profile_propagation_statistics_us = _ticks_usec() - profile_stage_start_us;
	}

	return removed_connections;
}

bool BrainDynamicsNative::_replace_last_incoming_connection(int32_t target_index, double new_weight) {
	const int32_t target_z = z_indices[target_index];
	const int32_t source_k_min = std::max(0, target_z - 2);
	const int32_t source_k_max = std::min(size_k - 1, target_z + 2);
	int32_t best_source_index = -1;
	double best_activity = -1.0;
	int32_t tie_count = 0;

	for (int32_t source_k = source_k_min; source_k <= source_k_max; ++source_k) {
		const int32_t slice_start = plane_size * source_k;
		const int32_t slice_end = slice_start + plane_size;

		for (int32_t source_spatial_index = slice_start; source_spatial_index < slice_end; ++source_spatial_index) {
			const int32_t source_index = neuron_index_by_spatial_index[source_spatial_index];

			if (source_index == target_index || internal_flags[source_index] == 0) {
				continue;
			}

			const double activity = activity_traces[source_index];

			if (activity > best_activity) {
				best_activity = activity;
				best_source_index = source_index;
				tie_count = 1;
			} else if (activity == best_activity) {
				++tie_count;
				std::uniform_int_distribution<int32_t> tie_distribution(1, tie_count);
				if (tie_distribution(structural_rng) == 1) {
					best_source_index = source_index;
				}
			}
		}
	}

	if (best_source_index < 0) {
		return false;
	}

	_append_connection(best_source_index, target_index, new_weight, 0.0);
	++incoming_connection_counts[target_index];
	return true;
}

void BrainDynamicsNative::_update_modulatory_field() {
	double field_sum = 0.0;
	double effective_modulation_sum = 0.0;

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		const int32_t neighbor_offset = neuron_index * 6;
		const int32_t neighbor_count = modulatory_neighbor_counts[neuron_index];
		double neighbor_sum = 0.0;

		for (int32_t neighbor_slot = 0; neighbor_slot < neighbor_count; ++neighbor_slot) {
			neighbor_sum += modulatory_fields[modulatory_neighbor_indices[neighbor_offset + neighbor_slot]];
		}

		const double neighbor_average = neighbor_sum / static_cast<double>(neighbor_count);
		const double new_field = modulatory_persistence * (one_minus_modulatory_spread * modulatory_fields[neuron_index] + modulatory_spread * neighbor_average) + modulatory_releases[neuron_index];
		const double effective_modulation = modulatory_sensitivities[neuron_index] * (1.0 - std::exp(-new_field));

		next_modulatory_fields[neuron_index] = new_field;
		effective_modulations[neuron_index] = effective_modulation;
		field_sum += new_field;
		effective_modulation_sum += effective_modulation;
	}

	modulatory_fields.swap(next_modulatory_fields);
	mean_modulatory_field = neuron_count == 0 ? 0.0 : field_sum / static_cast<double>(neuron_count);
	mean_effective_modulation = neuron_count == 0 ? 0.0 : effective_modulation_sum / static_cast<double>(neuron_count);
}

void BrainDynamicsNative::_apply_neuron_state_update() {
	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		neuron_states[neuron_index] = std::max(0.0, retention_factors[neuron_index] * neuron_states[neuron_index] + incoming_signals[neuron_index]);
	}
}

int32_t BrainDynamicsNative::_apply_structural_plasticity() {
	int32_t added_connections = 0;

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		if (internal_flags[source_index] == 0 || firing_states[source_index] == 0) {
			continue;
		}

		const int32_t target_index = _random_structural_target_index(source_index);
		if (target_index < 0 || input_flags[target_index] != 0) {
			continue;
		}

		bool already_connected = false;
		const auto &source_connections = connections[source_index];

		for (const Connection &connection : source_connections) {
			if (connection.target_index == target_index) {
				already_connected = true;
				break;
			}
		}

		if (already_connected) {
			continue;
		}

		if (activity_traces[source_index] + activity_traces[target_index] <= structural_plasticity) {
			continue;
		}

		_append_connection(source_index, target_index, structural_new_weight, 0.0);
		++incoming_connection_counts[target_index];
		++added_connections;
	}

	if (added_connections > 0) {
		total_connections += added_connections;
		total_connections_formed += added_connections;
		total_structural_plasticity_connections_formed += added_connections;
		_refresh_mean_outgoing_connections();
	}

	return added_connections;
}

int32_t BrainDynamicsNative::_random_structural_target_index(int32_t source_index) {
	const int32_t source_z = z_indices[source_index];
	const int32_t target_k_min = std::max(0, source_z - 2);
	const int32_t target_k_max = std::min(size_k - 1, source_z + 2);
	const int32_t slice_count = target_k_max - target_k_min + 1;
	const int32_t candidate_count = plane_size * slice_count - 1;

	if (candidate_count <= 0) {
		return -1;
	}

	const int32_t source_slot = spatial_indices[source_index] - plane_size * target_k_min;
	std::uniform_int_distribution<int32_t> slot_distribution(0, candidate_count - 1);
	int32_t random_slot = slot_distribution(structural_rng);

	if (random_slot >= source_slot) {
		++random_slot;
	}

	const int32_t local_k = random_slot / plane_size;
	const int32_t within_slice = random_slot - local_k * plane_size;
	const int32_t target_spatial_index = within_slice + plane_size * (target_k_min + local_k);
	return neuron_index_by_spatial_index[target_spatial_index];
}

void BrainDynamicsNative::_append_connection(int32_t source_index, int32_t target_index, double weight, double eligibility) {
	auto &source_connections = connections[source_index];
	const int32_t old_degree = static_cast<int32_t>(source_connections.size());

	source_connections.push_back({ target_index, weight, eligibility });
	_record_out_degree_change(source_index, old_degree, old_degree + 1);
}

void BrainDynamicsNative::_remove_connection_at(int32_t source_index, int32_t connection_index, int32_t target_index) {
	auto &source_connections = connections[source_index];
	const int32_t old_degree = static_cast<int32_t>(source_connections.size());

	source_connections.erase(source_connections.begin() + connection_index);
	--incoming_connection_counts[target_index];
	_record_out_degree_change(source_index, old_degree, old_degree - 1);
}

void BrainDynamicsNative::_update_initial_statistics() {
	double weight_sum = 0.0;
	double eligibility_sum = 0.0;
	total_connections = 0;
	outgoing_neuron_count = 0;
	out_degree_histogram.clear();
	min_out_degree = 0;
	max_out_degree = 0;

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		const auto &source_connections = connections[source_index];
		total_connections += static_cast<int64_t>(source_connections.size());

		for (const Connection &connection : source_connections) {
			weight_sum += connection.weight;
			eligibility_sum += connection.eligibility_trace;
		}

		if (output_flags[source_index] == 0) {
			const int32_t degree = static_cast<int32_t>(source_connections.size());
			++outgoing_neuron_count;
			_ensure_degree_histogram_size(degree);
			++out_degree_histogram[degree];
		}
	}

	mean_weight = total_connections == 0 ? 0.0 : weight_sum / static_cast<double>(total_connections);
	mean_eligibility_trace = total_connections == 0 ? 0.0 : eligibility_sum / static_cast<double>(total_connections);
	_rebuild_degree_extremes();
	_refresh_mean_outgoing_connections();
}

void BrainDynamicsNative::_update_modulatory_statistics() {
	double field_sum = 0.0;
	double effective_modulation_sum = 0.0;

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		field_sum += modulatory_fields[neuron_index];
		effective_modulation_sum += effective_modulations[neuron_index];
	}

	mean_modulatory_field = neuron_count == 0 ? 0.0 : field_sum / static_cast<double>(neuron_count);
	mean_effective_modulation = neuron_count == 0 ? 0.0 : effective_modulation_sum / static_cast<double>(neuron_count);
}

void BrainDynamicsNative::_record_out_degree_change(int32_t source_index, int32_t old_degree, int32_t new_degree) {
	if (output_flags[source_index] != 0) {
		return;
	}

	_ensure_degree_histogram_size(std::max(old_degree, new_degree));
	--out_degree_histogram[old_degree];
	++out_degree_histogram[new_degree];

	if (new_degree < min_out_degree) {
		min_out_degree = new_degree;
	}
	if (new_degree > max_out_degree) {
		max_out_degree = new_degree;
	}

	if (old_degree == min_out_degree && out_degree_histogram[old_degree] == 0) {
		while (min_out_degree < static_cast<int32_t>(out_degree_histogram.size()) - 1 && out_degree_histogram[min_out_degree] == 0) {
			++min_out_degree;
		}
	}

	if (old_degree == max_out_degree && out_degree_histogram[old_degree] == 0) {
		while (max_out_degree > 0 && out_degree_histogram[max_out_degree] == 0) {
			--max_out_degree;
		}
	}
}

void BrainDynamicsNative::_ensure_degree_histogram_size(int32_t degree) {
	if (degree < static_cast<int32_t>(out_degree_histogram.size())) {
		return;
	}
	out_degree_histogram.resize(static_cast<size_t>(degree) + 1, 0);
}

void BrainDynamicsNative::_rebuild_degree_extremes() {
	if (outgoing_neuron_count == 0) {
		min_out_degree = 0;
		max_out_degree = 0;
		return;
	}

	min_out_degree = 0;
	while (min_out_degree < static_cast<int32_t>(out_degree_histogram.size()) && out_degree_histogram[min_out_degree] == 0) {
		++min_out_degree;
	}

	max_out_degree = static_cast<int32_t>(out_degree_histogram.size()) - 1;
	while (max_out_degree > 0 && out_degree_histogram[max_out_degree] == 0) {
		--max_out_degree;
	}
}

void BrainDynamicsNative::_refresh_mean_outgoing_connections() {
	mean_outgoing_connections = outgoing_neuron_count == 0 ? 0 : static_cast<int64_t>(static_cast<double>(total_connections) / static_cast<double>(outgoing_neuron_count));
}

Dictionary BrainDynamicsNative::get_brain_snapshot() const {
	Dictionary snapshot = static_cast<Dictionary>(Variant(brain_data).duplicate(true));
	Array neurons = snapshot["neurons"];

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		const auto &source_connections = connections[source_index];
		const int32_t connection_count = static_cast<int32_t>(source_connections.size());
		Array snapshot_connections;
		snapshot_connections.resize(connection_count);
		Dictionary source_neuron = neurons[source_index];
		source_neuron["modulatory_field"] = modulatory_fields[source_index];

		for (int32_t connection_index = 0; connection_index < connection_count; ++connection_index) {
			const Connection &native_connection = source_connections[connection_index];
			const int32_t target_index = native_connection.target_index;
			Dictionary target_neuron = neurons[target_index];
			Dictionary connection;
			connection["target"] = target_neuron["position"];
			connection["target_index"] = target_index;
			connection["weight"] = native_connection.weight;
			connection["eligibility_trace"] = native_connection.eligibility_trace;
			snapshot_connections[connection_index] = connection;
		}

		source_neuron["connections"] = snapshot_connections;
		neurons[source_index] = source_neuron;
	}

	snapshot["neurons"] = neurons;
	return snapshot;
}

void BrainDynamicsNative::clear() {
	brain_data = Dictionary();
	iteration_count = 0;
	elapsed_time_seconds = 0.0;
	delta_time_seconds = 0.0;
	mean_weight = 0.0;
	mean_signal = 0.0;
	total_connections = 0;
	mean_outgoing_connections = 0;
	min_out_degree = 0;
	max_out_degree = 0;
	added_connections_last_iteration = 0;
	total_connections_formed = 0;
	total_structural_plasticity_connections_formed = 0;
	total_homeostatic_connections_formed = 0;
	total_connections_destroyed = 0;
	mean_modulatory_release = 0.0;
	mean_modulatory_field = 0.0;
	mean_effective_modulation = 0.0;
	mean_eligibility_trace = 0.0;

	neuron_count = 0;
	size_i = 0;
	size_j = 0;
	size_k = 0;
	plane_size = 0;
	structural_plasticity = 0.0;
	structural_new_weight = 0.0;
	modulatory_persistence = 0.0;
	modulatory_spread = 0.0;
	one_minus_modulatory_spread = 1.0;
	trace_persistence = 0.0;
	one_minus_trace_persistence = 1.0;
	refractory_strength = 0.0;
	inverse_exponential_factor = 1.0;
	input_gain = 1.0;
	input_min_weight = 0.5;

	input_flags.clear();
	output_flags.clear();
	internal_flags.clear();
	activation_thresholds.clear();
	polarity_factors.clear();
	hebbian_rates.clear();
	decay_rates.clear();
	retention_factors.clear();
	modulatory_release_factors.clear();
	modulatory_sensitivities.clear();
	spatial_indices.clear();
	z_indices.clear();
	fatigues.clear();
	fatigues_used.clear();
	effective_thresholds_used.clear();
	neuron_states.clear();
	neuron_states_used.clear();
	normalized_signals.clear();
	firing_states.clear();
	activity_traces.clear();
	incoming_signals.clear();
	incoming_connection_counts.clear();
	pending_input_signals.clear();
	input_signals_used.clear();
	output_signals.clear();
	modulatory_releases.clear();
	modulatory_fields.clear();
	next_modulatory_fields.clear();
	effective_modulations.clear();
	input_neuron_indices.clear();
	output_slot_by_neuron_index.clear();
	neuron_index_by_spatial_index.clear();
	modulatory_neighbor_indices.clear();
	modulatory_neighbor_counts.clear();
	connections.clear();
	initial_connections.clear();
	initial_modulatory_fields.clear();
	outgoing_neuron_count = 0;
	out_degree_histogram.clear();
	orphaned_targets.clear();
	initial_structural_rng_seed = 0;
}

void BrainDynamicsNative::set_profiling_enabled(bool enabled) {
	profiling_enabled = enabled;
}

bool BrainDynamicsNative::get_profiling_enabled() const {
	return profiling_enabled;
}

double BrainDynamicsNative::get_output_signal(int64_t output_index) const {
	return output_signals[static_cast<size_t>(output_index)];
}

int64_t BrainDynamicsNative::get_total_connections() const {
	return total_connections;
}

double BrainDynamicsNative::get_mean_weight() const {
	if (total_connections == 0) {
		return 0.0;
	}

	double weight_sum = 0.0;

	// Match the runtime connection traversal order: sources ascending, connections descending.
	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		const auto &source_connections = connections[source_index];

		for (int32_t connection_index = static_cast<int32_t>(source_connections.size()) - 1; connection_index >= 0; --connection_index) {
			weight_sum += source_connections[connection_index].weight;
		}
	}

	return weight_sum / static_cast<double>(total_connections);
}

int64_t BrainDynamicsNative::get_total_connections_formed() const {
	return total_connections_formed;
}

int64_t BrainDynamicsNative::get_total_connections_destroyed() const {
	return total_connections_destroyed;
}

int64_t BrainDynamicsNative::get_profile_total_us() const {
	return profile_total_us;
}

int64_t BrainDynamicsNative::get_profile_prepare_and_firing_us() const {
	return profile_prepare_and_firing_us;
}

int64_t BrainDynamicsNative::get_profile_modulatory_field_us() const {
	return profile_modulatory_field_us;
}

int64_t BrainDynamicsNative::get_profile_connections_us() const {
	return profile_connections_us;
}

int64_t BrainDynamicsNative::get_profile_homeostatic_us() const {
	return profile_homeostatic_us;
}

int64_t BrainDynamicsNative::get_profile_propagation_statistics_us() const {
	return profile_propagation_statistics_us;
}

int64_t BrainDynamicsNative::get_profile_state_update_us() const {
	return profile_state_update_us;
}

int64_t BrainDynamicsNative::get_profile_structural_plasticity_us() const {
	return profile_structural_plasticity_us;
}

int64_t BrainDynamicsNative::_ticks_usec() {
	const auto now = std::chrono::steady_clock::now().time_since_epoch();
	return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}
