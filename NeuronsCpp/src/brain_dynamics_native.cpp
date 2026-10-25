// OPTIMIZED INITIALIZATION VERSION - 2026-09-13
#include "brain_dynamics_native.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
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
	ClassDB::bind_method(D_METHOD("get_viewer_state"), &BrainDynamicsNative::get_viewer_state);

	ClassDB::bind_method(D_METHOD("set_profiling_enabled", "enabled"), &BrainDynamicsNative::set_profiling_enabled);
	ClassDB::bind_method(D_METHOD("get_profiling_enabled"), &BrainDynamicsNative::get_profiling_enabled);

	ClassDB::bind_method(D_METHOD("get_output_signal", "output_index"), &BrainDynamicsNative::get_output_signal);
	ClassDB::bind_method(D_METHOD("get_total_connections"), &BrainDynamicsNative::get_total_connections);
	ClassDB::bind_method(D_METHOD("get_neuron_statistics", "parameter"), &BrainDynamicsNative::get_neuron_statistics);

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
	elapsed_time_seconds = 0.0f;
	delta_time_seconds = 0.0f;
	added_connections_last_iteration = 0;
	total_connections_formed = 0;
	total_structural_plasticity_connections_formed = 0;
	total_homeostatic_connections_formed = 0;
	total_connections_destroyed = 0;

	Array neurons = brain_data["neurons"];
	Vector3i brain_size = brain_data["size"];
	neuron_count = static_cast<int32_t>(neurons.size());
	size_i = brain_size.x;
	size_j = brain_size.y;
	size_k = brain_size.z;
	plane_size = size_i * size_j;
	structural_plasticity = static_cast<float>(brain_data["structural_plasticity"]);
	structural_new_weight = structural_plasticity * 0.25f;
	modulatory_persistence = static_cast<float>(brain_data["lambda"]);
	modulatory_spread = static_cast<float>(brain_data["nu"]);
	one_minus_modulatory_spread = 1.0f - modulatory_spread;
	trace_persistence = static_cast<float>(brain_data["mu"]);
	one_minus_trace_persistence = 1.0f - trace_persistence;
	refractory_strength = static_cast<float>(brain_data["refractory_strength"]);
	const float exponential_factor = static_cast<float>(brain_data["exponential_factor"]);
	inverse_exponential_factor = 1.0f / exponential_factor;
	homeostatic_low_state = -exponential_factor * std::log(1.0f - 0.05f);
	homeostatic_high_state = -exponential_factor * std::log(1.0f - 0.95f);
	input_gain = static_cast<float>(brain_data["input_gain"]);
	input_min_weight = 1.0f / (1.0f + input_gain);

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
		activation_thresholds[neuron_index] = is_input ? 0.0f : static_cast<float>(neuron["activation_threshold"]);
		polarity_factors[neuron_index] = static_cast<float>(neuron["polarity_factor"]);
		hebbian_rates[neuron_index] = static_cast<float>(neuron["hebbian_plasticity_rate"]);
		decay_rates[neuron_index] = is_input ? 0.0f : std::exp(-12.0f / static_cast<float>(neuron["decay_factor"]));
		modulatory_release_factors[neuron_index] = static_cast<float>(neuron["modulatory_release_factor"]);
		modulatory_sensitivities[neuron_index] = static_cast<float>(neuron["modulatory_sensitivity"]);
		modulatory_fields[neuron_index] = 0.0f;
		effective_modulations[neuron_index] = 0.0f;

		if (is_input) {
			retention_factors[neuron_index] = 0.0f;
			++input_count;
		} else {
			retention_factors[neuron_index] = static_cast<float>(neuron["retention_factor"]);
			if (is_output) {
				++output_count;
			}
		}
	}

	_cache_modulatory_neighbors();
	_setup_input_output_indices(input_count, output_count);
	_build_runtime_connections(neurons);
	_store_initial_runtime_state();

	std::random_device random_device;
	initial_structural_rng_seed = (static_cast<uint64_t>(random_device()) << 32) ^ static_cast<uint64_t>(random_device());
	structural_rng.seed(initial_structural_rng_seed);
}

void BrainDynamicsNative::_resize_neuron_arrays(int32_t count) {
	input_flags.assign(count, 0);
	output_flags.assign(count, 0);
	internal_flags.assign(count, 0);
	activation_thresholds.assign(count, 0.0f);
	polarity_factors.assign(count, 0.0f);
	hebbian_rates.assign(count, 0.0f);
	decay_rates.assign(count, 0.0f);
	retention_factors.assign(count, 0.0f);
	modulatory_release_factors.assign(count, 0.0f);
	modulatory_sensitivities.assign(count, 0.0f);
	spatial_indices.assign(count, 0);
	z_indices.assign(count, 0);

	fatigues.assign(count, 0.0f);
	fatigues_used.assign(count, 0.0f);
	neuron_states_used.assign(count, 0.0f);
	effective_thresholds_used.assign(count, 0.0f);
	neuron_states.assign(count, 0.0f);
	normalized_signals.assign(count, 0.0f);
	firing_states.assign(count, 0);
	activity_traces.assign(count, 0.0f);
	incoming_signals.assign(count, 0.0f);
	incoming_connection_counts.assign(count, 0);
	modulatory_releases.assign(count, 0.0f);
	modulatory_fields.assign(count, 0.0f);
	next_modulatory_fields.assign(count, 0.0f);
	effective_modulations.assign(count, 0.0f);

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
	input_signals_used.assign(input_count, 0.0f);
	pending_input_signals.assign(input_count, 0.0f);
	output_signals.assign(output_count, 0.0f);

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
	total_connections = 0;

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		Dictionary source_neuron = neurons[source_index];
		Array source_connections = source_neuron["connections"];
		auto &native_connections = connections[source_index];
		native_connections.reserve(static_cast<size_t>(source_connections.size()));
		total_connections += static_cast<int64_t>(source_connections.size());

		for (int32_t connection_index = 0; connection_index < source_connections.size(); ++connection_index) {
			Dictionary connection = source_connections[connection_index];
			Vector3i target_position = connection["target"];
			const int32_t target_spatial_index = target_position.x + size_i * target_position.y + plane_size * target_position.z;
			const int32_t target_index = neuron_index_by_spatial_index[target_spatial_index];

			native_connections.push_back({
				target_index,
				static_cast<float>(connection["weight"]),
				0.0f
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
	total_connections = 0;

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		const auto &source_connections = connections[source_index];
		total_connections += static_cast<int64_t>(source_connections.size());

		for (const Connection &connection : source_connections) {
			++incoming_connection_counts[connection.target_index];
		}
	}

	iteration_count = 0;
	elapsed_time_seconds = 0.0f;
	delta_time_seconds = 0.0f;
	added_connections_last_iteration = 0;
	total_connections_formed = 0;
	total_structural_plasticity_connections_formed = 0;
	total_homeostatic_connections_formed = 0;
	total_connections_destroyed = 0;
	std::random_device random_device;
	initial_structural_rng_seed = (static_cast<uint64_t>(random_device()) << 32) ^ static_cast<uint64_t>(random_device());
	structural_rng.seed(initial_structural_rng_seed);

	std::fill(fatigues.begin(), fatigues.end(), 0.0f);
	std::fill(fatigues_used.begin(), fatigues_used.end(), 0.0f);
	std::fill(neuron_states_used.begin(), neuron_states_used.end(), 0.0f);
	std::fill(input_signals_used.begin(), input_signals_used.end(), 0.0f);
	std::fill(neuron_states.begin(), neuron_states.end(), 0.0f);
	std::fill(normalized_signals.begin(), normalized_signals.end(), 0.0f);
	std::fill(firing_states.begin(), firing_states.end(), 0);
	std::fill(activity_traces.begin(), activity_traces.end(), 0.0f);
	std::fill(incoming_signals.begin(), incoming_signals.end(), 0.0f);
	std::fill(pending_input_signals.begin(), pending_input_signals.end(), 0.0f);
	std::fill(output_signals.begin(), output_signals.end(), 0.0f);
	std::fill(modulatory_releases.begin(), modulatory_releases.end(), 0.0f);
	std::fill(next_modulatory_fields.begin(), next_modulatory_fields.end(), 0.0f);

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		effective_thresholds_used[neuron_index] = activation_thresholds[neuron_index];
		modulatory_fields[neuron_index] = initial_modulatory_fields[neuron_index];
		effective_modulations[neuron_index] = 0.0f;
	}
}

void BrainDynamicsNative::add_input_signal(int64_t input_index, double input_signal) {
	pending_input_signals[static_cast<size_t>(input_index)] = static_cast<float>(input_signal);
}

int64_t BrainDynamicsNative::advance(double delta_seconds) {
	const float delta_seconds_f32 = static_cast<float>(delta_seconds);
	delta_time_seconds = delta_seconds_f32;
	elapsed_time_seconds += delta_seconds_f32;
	++iteration_count;
	added_connections_last_iteration = 0;

	if (!profiling_enabled) {
		const int32_t removed_connections = _propagate_signals();
		_apply_neuron_state_update();
		const int32_t added_structural = _apply_structural_plasticity();
		added_connections_last_iteration += added_structural;
		total_structural_plasticity_connections_formed += added_structural;
		total_connections_formed += added_structural;
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

	const int32_t added_structural = _apply_structural_plasticity();
	added_connections_last_iteration += added_structural;
	total_structural_plasticity_connections_formed += added_structural;
	total_connections_formed += added_structural;
	current_us = _ticks_usec();
	profile_structural_plasticity_us = current_us - stage_start_us;
	profile_total_us = current_us - total_start_us;

	return removed_connections;
}

int32_t BrainDynamicsNative::_propagate_signals() {
	int32_t removed_connections = 0;
	int32_t replacement_connections = 0;
	int64_t profile_stage_start_us = profiling_enabled ? _ticks_usec() : 0;

	std::fill(incoming_signals.begin(), incoming_signals.end(), 0.0f);
	std::fill(normalized_signals.begin(), normalized_signals.end(), 0.0f);
	std::fill(firing_states.begin(), firing_states.end(), 0);
	std::fill(output_signals.begin(), output_signals.end(), 0.0f);
	std::fill(modulatory_releases.begin(), modulatory_releases.end(), 0.0f);
	std::fill(input_signals_used.begin(), input_signals_used.end(), 0.0f);
	orphaned_targets.clear();

	for (int32_t input_index = 0; input_index < static_cast<int32_t>(input_neuron_indices.size()); ++input_index) {
		const float input_signal = pending_input_signals[input_index];
		input_signals_used[input_index] = input_signal;
		neuron_states[input_neuron_indices[input_index]] = input_signal;
		pending_input_signals[input_index] = 0.0f;
	}

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		const bool is_input = input_flags[neuron_index] != 0;
		neuron_states_used[neuron_index] = neuron_states[neuron_index];
		fatigues_used[neuron_index] = fatigues[neuron_index];

		if (!is_input) {
			const float current_fatigue = fatigues[neuron_index];
			effective_thresholds_used[neuron_index] = activation_thresholds[neuron_index] * (1.0f + current_fatigue);
			activity_traces[neuron_index] *= trace_persistence;
		}

		const float current_state = neuron_states[neuron_index];

		if (current_state <= 0.0f) {
			if (!is_input) {
				fatigues[neuron_index] *= 0.5f;
			}
			continue;
		}

		const float normalized_signal = 1.0f - std::exp(-current_state * inverse_exponential_factor);
		normalized_signals[neuron_index] = normalized_signal;

		if (!is_input) {
			activity_traces[neuron_index] += one_minus_trace_persistence * normalized_signal;
		}

		if (is_input) {
			firing_states[neuron_index] = 1;
			modulatory_releases[neuron_index] = modulatory_release_factors[neuron_index] * normalized_signal;
			continue;
		}

		if (normalized_signal <= effective_thresholds_used[neuron_index]) {
			fatigues[neuron_index] *= 0.5f;
			continue;
		}

		firing_states[neuron_index] = 1;
		modulatory_releases[neuron_index] = modulatory_release_factors[neuron_index] * normalized_signal;
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

	const float trace_keep = trace_persistence;
	const float trace_add = one_minus_trace_persistence;
	const float cached_input_gain = input_gain;
	const float cached_input_min_weight = input_min_weight;

	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		auto &source_connections = connections[source_index];
		int32_t connection_index = static_cast<int32_t>(source_connections.size()) - 1;

		if (connection_index < 0) {
			continue;
		}

		const bool source_fired = firing_states[source_index] != 0;
		const float source_normalized_signal = normalized_signals[source_index];
		const bool source_is_input = input_flags[source_index] != 0;
		const float hebbian_rate = hebbian_rates[source_index];

		if (source_is_input) {
			const float source_signal = cached_input_gain * neuron_states[source_index];
			const float transmission_scale = polarity_factors[source_index] * source_signal;

			if (source_fired) {
				const float eligibility_source_scale = trace_add * source_normalized_signal;
				while (connection_index >= 0) {
					Connection &connection = source_connections[connection_index];
					const int32_t target_index = connection.target_index;
					float weight = connection.weight;
					incoming_signals[target_index] += transmission_scale * weight;
					weight += hebbian_rate * (1.0f - weight);
					const float eligibility = trace_keep * connection.eligibility_trace + eligibility_source_scale * normalized_signals[target_index];
					connection.eligibility_trace = eligibility;

					const float modulation = effective_modulations[target_index];
					if (modulation >= 0.0f) {
						weight += modulation * eligibility * (1.0f - weight);
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
					float weight = connection.weight;
					const float eligibility = trace_keep * connection.eligibility_trace;
					connection.eligibility_trace = eligibility;

					const float modulation = effective_modulations[target_index];
					if (modulation >= 0.0f) {
						weight += modulation * eligibility * (1.0f - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					weight = std::max(weight, cached_input_min_weight);
					connection.weight = weight;
					--connection_index;
				}
			}
		} else {
			const float decrement = decay_rates[source_index];
			const float transmission_scale = polarity_factors[source_index] * source_normalized_signal;

			if (source_fired) {
				const float eligibility_source_scale = trace_add * source_normalized_signal;
				while (connection_index >= 0) {
					Connection &connection = source_connections[connection_index];
					const int32_t target_index = connection.target_index;
					float weight = connection.weight;
					incoming_signals[target_index] += transmission_scale * weight;
					weight += hebbian_rate * (1.0f - weight);
					const float eligibility = trace_keep * connection.eligibility_trace + eligibility_source_scale * normalized_signals[target_index];
					connection.eligibility_trace = eligibility;

					const float modulation = effective_modulations[target_index];
					if (modulation >= 0.0f) {
						weight += modulation * eligibility * (1.0f - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					float connection_decrement = decrement;
					if (output_flags[target_index] != 0) {
						connection_decrement *= activity_traces[target_index];
					}
					weight = std::max(weight - connection_decrement, 0.0f);

					if (weight <= 0.0f) {
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
					float weight = connection.weight;
					const float eligibility = trace_keep * connection.eligibility_trace;
					connection.eligibility_trace = eligibility;

					const float modulation = effective_modulations[target_index];
					if (modulation >= 0.0f) {
						weight += modulation * eligibility * (1.0f - weight);
					} else {
						weight += modulation * eligibility * weight;
					}

					float connection_decrement = decrement;
					if (output_flags[target_index] != 0) {
						connection_decrement *= activity_traces[target_index];
					}
					weight = std::max(weight - connection_decrement, 0.0f);

					if (weight <= 0.0f) {
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
	total_connections_destroyed += removed_connections;
	total_homeostatic_connections_formed += replacement_connections;
	total_connections_formed += replacement_connections;
	added_connections_last_iteration += replacement_connections;

	if (profiling_enabled) {
		profile_propagation_statistics_us = _ticks_usec() - profile_stage_start_us;
	}

	return removed_connections;
}

bool BrainDynamicsNative::_replace_last_incoming_connection(int32_t target_index, float new_weight) {
	const int32_t target_z = z_indices[target_index];
	const int32_t source_k_min = std::max(0, target_z - 2);
	const int32_t source_k_max = std::min(size_k - 1, target_z + 2);
	int32_t best_source_index = -1;
	float best_activity = -1.0f;
	int32_t tie_count = 0;

	for (int32_t source_k = source_k_min; source_k <= source_k_max; ++source_k) {
		const int32_t slice_start = plane_size * source_k;
		const int32_t slice_end = slice_start + plane_size;

		for (int32_t source_spatial_index = slice_start; source_spatial_index < slice_end; ++source_spatial_index) {
			const int32_t source_index = neuron_index_by_spatial_index[source_spatial_index];

			if (source_index == target_index || internal_flags[source_index] == 0) {
				continue;
			}

			const float activity = activity_traces[source_index];

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

	_append_connection(best_source_index, target_index, new_weight, 0.0f);
	++incoming_connection_counts[target_index];
	return true;
}

void BrainDynamicsNative::_update_modulatory_field() {
	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		const int32_t neighbor_offset = neuron_index * 6;
		const int32_t neighbor_count = modulatory_neighbor_counts[neuron_index];
		float neighbor_sum = 0.0f;

		for (int32_t neighbor_slot = 0; neighbor_slot < neighbor_count; ++neighbor_slot) {
			neighbor_sum += modulatory_fields[modulatory_neighbor_indices[neighbor_offset + neighbor_slot]];
		}

		const float neighbor_average = neighbor_sum / static_cast<float>(neighbor_count);
		const float new_field = modulatory_persistence * (one_minus_modulatory_spread * modulatory_fields[neuron_index] + modulatory_spread * neighbor_average) + modulatory_releases[neuron_index];
		const float effective_modulation = modulatory_sensitivities[neuron_index] * (1.0f - std::exp(-new_field));

		next_modulatory_fields[neuron_index] = new_field;
		effective_modulations[neuron_index] = effective_modulation;
	}

	modulatory_fields.swap(next_modulatory_fields);
}

void BrainDynamicsNative::_apply_neuron_state_update() {
	constexpr float homeostatic_low_activity = 0.05f;
	constexpr float homeostatic_high_activity = 0.95f;

	for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
		float state = std::max(0.0f, retention_factors[neuron_index] * neuron_states[neuron_index] + incoming_signals[neuron_index]);

		if (input_flags[neuron_index] == 0) {
			const float activity = activity_traces[neuron_index];

			if (activity < homeostatic_low_activity && state < homeostatic_low_state) {
				state += (homeostatic_low_activity - activity) / homeostatic_low_activity * (homeostatic_low_state - state);
			} else if (activity > homeostatic_high_activity && state > homeostatic_high_state) {
				state -= (activity - homeostatic_high_activity) / (1.0f - homeostatic_high_activity) * (state - homeostatic_high_state);
			}
		}

		neuron_states[neuron_index] = state;
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

		_append_connection(source_index, target_index, structural_new_weight, 0.0f);
		++incoming_connection_counts[target_index];
		++added_connections;
	}

	if (added_connections > 0) {
		total_connections += added_connections;
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

void BrainDynamicsNative::_append_connection(int32_t source_index, int32_t target_index, float weight, float eligibility) {
	connections[source_index].push_back({ target_index, weight, eligibility });
}

void BrainDynamicsNative::_remove_connection_at(int32_t source_index, int32_t connection_index, int32_t target_index) {
	connections[source_index].erase(connections[source_index].begin() + connection_index);
	--incoming_connection_counts[target_index];
}

void BrainDynamicsNative::_update_initial_statistics() {
	// Legacy compatibility hook. Initial statistics are no longer maintained.
}

void BrainDynamicsNative::_update_modulatory_statistics() {
	// Legacy compatibility hook. Modulatory means are no longer maintained.
}

void BrainDynamicsNative::_record_out_degree_change(int32_t source_index, int32_t old_degree, int32_t new_degree) {
	(void)source_index;
	(void)old_degree;
	(void)new_degree;
}

void BrainDynamicsNative::_ensure_degree_histogram_size(int32_t degree) {
	(void)degree;
}

void BrainDynamicsNative::_rebuild_degree_extremes() {
}

void BrainDynamicsNative::_refresh_mean_outgoing_connections() {
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
		if (input_flags[source_index] != 0) {
			source_neuron["retention_factor"] = 0.0f;
		}

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

// Read-only UI snapshot. The core neural dynamics never depends on this method.
// Packed arrays keep the transfer to GDScript compact, one per rendered frame.
Dictionary BrainDynamicsNative::get_viewer_state() const {
	Dictionary result;
	auto floats = [](const std::vector<float> &data) -> PackedFloat32Array {
		PackedFloat32Array packed;
		packed.resize(static_cast<int32_t>(data.size()));
		for (int32_t i = 0; i < static_cast<int32_t>(data.size()); ++i) {
			packed.set(i, data[static_cast<size_t>(i)]);
		}
		return packed;
	};
	auto ints = [](const std::vector<int32_t> &data) -> PackedInt32Array {
		PackedInt32Array packed;
		packed.resize(static_cast<int32_t>(data.size()));
		for (int32_t i = 0; i < static_cast<int32_t>(data.size()); ++i) {
			packed.set(i, data[static_cast<size_t>(i)]);
		}
		return packed;
	};

	result["normalized_signals"] = floats(normalized_signals);
	result["neuron_states_used"] = floats(neuron_states_used);
	result["incoming_signals"] = floats(incoming_signals);
	result["fatigues"] = floats(fatigues);
	result["fatigues_used"] = floats(fatigues_used);
	result["activity_traces"] = floats(activity_traces);
	result["effective_thresholds_used"] = floats(effective_thresholds_used);
	result["decay_rates"] = floats(decay_rates);
	result["modulatory_releases"] = floats(modulatory_releases);
	result["modulatory_fields"] = floats(modulatory_fields);
	result["effective_modulations"] = floats(effective_modulations);
	result["input_signals_used"] = floats(input_signals_used);
	PackedFloat64Array packed_outputs;
	packed_outputs.resize(static_cast<int32_t>(output_signals.size()));
	for (int32_t i = 0; i < static_cast<int32_t>(output_signals.size()); ++i) {
		packed_outputs.set(i, static_cast<double>(output_signals[static_cast<size_t>(i)]));
	}
	result["output_signals"] = packed_outputs;
	result["incoming_connection_counts"] = ints(incoming_connection_counts);
	result["input_neuron_indices"] = ints(input_neuron_indices);
	result["output_slot_by_neuron_index"] = ints(output_slot_by_neuron_index);
	PackedFloat32Array firing;
	firing.resize(neuron_count);
	for (int32_t i = 0; i < neuron_count; ++i) {
		firing.set(i, static_cast<float>(firing_states[static_cast<size_t>(i)]));
	}
	result["firing_states"] = firing;

	result["iteration_count"] = iteration_count;
	result["elapsed_time_seconds"] = elapsed_time_seconds;
	result["total_connections"] = total_connections;
	result["added_connections_last_iteration"] = added_connections_last_iteration;
	result["total_connections_destroyed"] = total_connections_destroyed;
	result["total_structural_plasticity_connections_formed"] = total_structural_plasticity_connections_formed;
	result["total_homeostatic_connections_formed"] = total_homeostatic_connections_formed;
	result["input_gain"] = input_gain;
	result["input_min_weight"] = input_min_weight;

	double signal_sum = 0.0;
	double release_sum = 0.0;
	double field_sum = 0.0;
	double modulation_sum = 0.0;
	for (int32_t i = 0; i < neuron_count; ++i) {
		signal_sum += normalized_signals[i];
		release_sum += modulatory_releases[i];
		field_sum += modulatory_fields[i];
		modulation_sum += effective_modulations[i];
	}
	const double n = neuron_count > 0 ? static_cast<double>(neuron_count) : 1.0;
	result["mean_signal"] = signal_sum / n;
	result["mean_modulatory_release"] = release_sum / n;
	result["mean_modulatory_field"] = field_sum / n;
	result["mean_effective_modulation"] = modulation_sum / n;

	double weight_sum = 0.0;
	double trace_sum = 0.0;
	int32_t outgoing_count = 0;
	int64_t outgoing_edges = 0;
	int32_t min_degree = 0;
	int32_t max_degree = 0;
	bool first_degree = true;
	for (int32_t source = 0; source < neuron_count; ++source) {
		const auto &edges = connections[source];
		for (const Connection &edge : edges) {
			weight_sum += edge.weight;
			trace_sum += edge.eligibility_trace;
		}
		if (output_flags[source] == 0) {
			const int32_t degree = static_cast<int32_t>(edges.size());
			++outgoing_count;
			outgoing_edges += degree;
			if (first_degree) {
				min_degree = max_degree = degree;
				first_degree = false;
			} else {
				min_degree = std::min(min_degree, degree);
				max_degree = std::max(max_degree, degree);
			}
		}
	}
	const double edge_count = total_connections > 0 ? static_cast<double>(total_connections) : 1.0;
	result["mean_weight"] = weight_sum / edge_count;
	result["mean_eligibility_trace"] = trace_sum / edge_count;
	result["mean_outgoing_connections"] = outgoing_count > 0 ? outgoing_edges / outgoing_count : 0;
	result["min_out_degree"] = min_degree;
	result["max_out_degree"] = max_degree;
	return result;
}

void BrainDynamicsNative::clear() {
	brain_data = Dictionary();
	iteration_count = 0;
	elapsed_time_seconds = 0.0f;
	delta_time_seconds = 0.0f;
	total_connections = 0;

	neuron_count = 0;
	size_i = 0;
	size_j = 0;
	size_k = 0;
	plane_size = 0;
	structural_plasticity = 0.0f;
	structural_new_weight = 0.0f;
	modulatory_persistence = 0.0f;
	modulatory_spread = 0.0f;
	one_minus_modulatory_spread = 1.0f;
	trace_persistence = 0.0f;
	one_minus_trace_persistence = 1.0f;
	refractory_strength = 0.0f;
	inverse_exponential_factor = 1.0f;
	homeostatic_low_state = 0.0f;
	homeostatic_high_state = 0.0f;
	input_gain = 1.0f;
	input_min_weight = 0.5f;

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
	effective_thresholds_used.clear();
	neuron_states.clear();
	normalized_signals.clear();
	firing_states.clear();
	activity_traces.clear();
	incoming_signals.clear();
	incoming_connection_counts.clear();
	pending_input_signals.clear();
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
	orphaned_targets.clear();
	initial_structural_rng_seed = 0;
	added_connections_last_iteration = 0;
	total_connections_formed = 0;
	total_structural_plasticity_connections_formed = 0;
	total_homeostatic_connections_formed = 0;
	total_connections_destroyed = 0;
	fatigues_used.clear();
	neuron_states_used.clear();
	input_signals_used.clear();
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

Dictionary BrainDynamicsNative::get_neuron_statistics(const String &parameter) const {
	const std::vector<float> *values = nullptr;
	std::vector<float> non_input_values;

	if (parameter == "state" || parameter == "signal") {
		const std::vector<float> &source_values = parameter == "state" ? neuron_states : normalized_signals;
		non_input_values.reserve(static_cast<size_t>(neuron_count - input_neuron_indices.size()));

		for (int32_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
			if (input_flags[neuron_index] == 0) {
				non_input_values.push_back(source_values[neuron_index]);
			}
		}

		values = &non_input_values;
	} else if (parameter == "fatigue") {
		values = &fatigues;
	} else if (parameter == "effective_threshold") {
		values = &effective_thresholds_used;
	} else if (parameter == "activity_trace") {
		values = &activity_traces;
	} else if (parameter == "incoming_signal") {
		values = &incoming_signals;
	} else if (parameter == "modulatory_release") {
		values = &modulatory_releases;
	} else if (parameter == "modulatory_field") {
		values = &modulatory_fields;
	} else if (parameter == "effective_modulation") {
		values = &effective_modulations;
	} else if (parameter == "activation_threshold") {
		values = &activation_thresholds;
	} else if (parameter == "polarity_factor") {
		values = &polarity_factors;
	} else if (parameter == "hebbian_plasticity_rate") {
		values = &hebbian_rates;
	} else if (parameter == "decay_rate") {
		values = &decay_rates;
	} else if (parameter == "retention_factor") {
		values = &retention_factors;
	} else if (parameter == "modulatory_release_factor") {
		values = &modulatory_release_factors;
	} else if (parameter == "modulatory_sensitivity") {
		values = &modulatory_sensitivities;
	}

	if (values == nullptr) {
		return Dictionary();
	}

	return _calculate_statistics(*values);
}

Dictionary BrainDynamicsNative::_calculate_statistics(const std::vector<float> &values) const {
	Dictionary statistics;
	const int64_t count = static_cast<int64_t>(values.size());

	statistics["count"] = count;
	if (count == 0) {
		statistics["min"] = 0.0f;
		statistics["max"] = 0.0f;
		statistics["mean"] = 0.0f;
		statistics["std"] = 0.0f;
		statistics["25%"] = 0.0f;
		statistics["50%"] = 0.0f;
		statistics["75%"] = 0.0f;
		return statistics;
	}

	float mean = 0.0f;
	float m2 = 0.0f;
	int64_t sample_count = 0;

	for (float value : values) {
		++sample_count;
		const float delta = value - mean;
		mean += delta / static_cast<float>(sample_count);
		const float delta2 = value - mean;
		m2 += delta * delta2;
	}

	std::vector<float> sorted_values = values;
	std::sort(sorted_values.begin(), sorted_values.end());

	auto percentile = [&sorted_values](float fraction) -> float {
		const float position = fraction * static_cast<float>(sorted_values.size() - 1);
		const size_t lower_index = static_cast<size_t>(std::floor(position));
		const size_t upper_index = static_cast<size_t>(std::ceil(position));
		const float interpolation = position - static_cast<float>(lower_index);
		return sorted_values[lower_index] + interpolation * (sorted_values[upper_index] - sorted_values[lower_index]);
	};

	statistics["min"] = sorted_values.front();
	statistics["max"] = sorted_values.back();
	statistics["mean"] = mean;
	statistics["std"] = count > 1 ? std::sqrt(m2 / static_cast<float>(count - 1)) : 0.0f;
	statistics["25%"] = percentile(0.25f);
	statistics["50%"] = percentile(0.50f);
	statistics["75%"] = percentile(0.75f);
	return statistics;
}

double BrainDynamicsNative::get_mean_weight() const {
	if (total_connections == 0) {
		return 0.0f;
	}

	float weight_sum = 0.0f;

	// Match the runtime connection traversal order: sources ascending, connections descending.
	for (int32_t source_index = 0; source_index < neuron_count; ++source_index) {
		const auto &source_connections = connections[source_index];

		for (int32_t connection_index = static_cast<int32_t>(source_connections.size()) - 1; connection_index >= 0; --connection_index) {
			weight_sum += source_connections[connection_index].weight;
		}
	}

	return weight_sum / static_cast<float>(total_connections);
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
