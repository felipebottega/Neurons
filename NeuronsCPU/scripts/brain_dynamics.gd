class_name BrainDynamics
extends RefCounted


var brain: Dictionary = {}

var iteration_count := 0
var elapsed_time_seconds := 0.0
var delta_time_seconds := 0.0
var mean_weight := 0.0
var mean_signal := 0.0
var total_connections := 0
var mean_outgoing_connections := 0
var min_out_degree := 0
var max_out_degree := 0
var added_connections_last_iteration = 0
var total_connections_formed = 0
var total_structural_plasticity_connections_formed = 0
var total_homeostatic_connections_formed = 0
var total_connections_destroyed = 0
var mean_modulatory_release = 0.0
var mean_modulatory_field = 0.0
var mean_effective_modulation = 0.0
var mean_eligibility_trace = 0.0

var decay_rates := PackedFloat64Array()
var retention_factors = PackedFloat64Array()
var fatigues = PackedFloat64Array()
var fatigues_used = PackedFloat64Array()
var effective_thresholds_used = PackedFloat64Array()
var modulatory_release_factors = PackedFloat64Array()
var modulatory_sensitivities = PackedFloat64Array()

var neuron_states := PackedFloat64Array()
var neuron_states_used = PackedFloat64Array()
var normalized_signals := PackedFloat64Array()
var firing_states = PackedFloat64Array()
var activity_traces = PackedFloat64Array()
var incoming_signals := PackedFloat64Array()
var incoming_connection_counts = PackedInt32Array()
var pending_input_signals := PackedFloat64Array()
var input_signals_used = PackedFloat64Array()
var output_signals := PackedFloat64Array()
var modulatory_releases = PackedFloat64Array()
var modulatory_fields = PackedFloat64Array()
var next_modulatory_fields = PackedFloat64Array()
var effective_modulations = PackedFloat64Array()

var input_neuron_indices := PackedInt32Array()
var output_slot_by_neuron_index := PackedInt32Array()
var neuron_index_by_spatial_index = PackedInt32Array()
var modulatory_neighbor_indices = PackedInt32Array()
var modulatory_neighbor_counts = PackedInt32Array()

var initial_connections: Array = []
var initial_modulatory_fields = PackedFloat64Array()

var structural_rng = RandomNumberGenerator.new()
var initial_structural_rng_state = 0

var modulatory_persistence = 0.0
var modulatory_spread = 0.0
var trace_persistence = 0.0
var refractory_strength = 0.0
var inverse_exponential_factor := 1.0
var homeostatic_low_state = 0.0
var homeostatic_high_state = 0.0
var input_gain = 1.0
var input_min_weight = 0.5

# Optional runtime profiling. No timing calls are made while this is false.
var profiling_enabled: bool = false
var profile_total_us: int = 0
var profile_propagate_us: int = 0
var profile_prepare_and_firing_us: int = 0
var profile_modulatory_field_us: int = 0
var profile_connections_us: int = 0
var profile_homeostatic_us: int = 0
var profile_propagation_statistics_us: int = 0
var profile_state_update_us: int = 0
var profile_structural_plasticity_us: int = 0


## Associates a generated brain with this runtime and resets lifetime counters.
func set_brain(new_brain: Dictionary) -> void:
	brain = new_brain
	iteration_count = 0
	elapsed_time_seconds = 0.0
	delta_time_seconds = 0.0
	mean_signal = 0.0
	mean_modulatory_release = 0.0
	mean_modulatory_field = 0.0
	mean_effective_modulation = 0.0
	mean_eligibility_trace = 0.0

	var neurons: Array = brain["neurons"]
	var neuron_count := neurons.size()
	var I: int = brain["size"].x
	var J: int = brain["size"].y
	var K: int = brain["size"].z
	var IJ: int = I * J
	var input_count := 0
	var output_count := 0

	decay_rates = PackedFloat64Array()
	decay_rates.resize(neuron_count)
	retention_factors = PackedFloat64Array()
	retention_factors.resize(neuron_count)
	fatigues = PackedFloat64Array()
	fatigues.resize(neuron_count)
	fatigues_used = PackedFloat64Array()
	fatigues_used.resize(neuron_count)
	effective_thresholds_used = PackedFloat64Array()
	effective_thresholds_used.resize(neuron_count)
	modulatory_release_factors = PackedFloat64Array()
	modulatory_release_factors.resize(neuron_count)
	modulatory_sensitivities = PackedFloat64Array()
	modulatory_sensitivities.resize(neuron_count)
	neuron_states = PackedFloat64Array()
	neuron_states.resize(neuron_count)
	neuron_states_used = PackedFloat64Array()
	neuron_states_used.resize(neuron_count)
	normalized_signals = PackedFloat64Array()
	normalized_signals.resize(neuron_count)
	firing_states = PackedFloat64Array()
	firing_states.resize(neuron_count)
	activity_traces = PackedFloat64Array()
	activity_traces.resize(neuron_count)
	incoming_signals = PackedFloat64Array()
	incoming_signals.resize(neuron_count)
	incoming_connection_counts = PackedInt32Array()
	incoming_connection_counts.resize(neuron_count)
	output_slot_by_neuron_index = PackedInt32Array()
	output_slot_by_neuron_index.resize(neuron_count)
	output_slot_by_neuron_index.fill(-1)
	modulatory_releases = PackedFloat64Array()
	modulatory_releases.resize(neuron_count)
	modulatory_fields = PackedFloat64Array()
	modulatory_fields.resize(neuron_count)
	next_modulatory_fields = PackedFloat64Array()
	next_modulatory_fields.resize(neuron_count)
	effective_modulations = PackedFloat64Array()
	effective_modulations.resize(neuron_count)
	modulatory_persistence = float(brain["lambda"])
	modulatory_spread = float(brain["nu"])
	trace_persistence = float(brain["mu"])
	refractory_strength = float(brain["refractory_strength"])
	var exponential_factor = float(brain["exponential_factor"])
	inverse_exponential_factor = 1.0 / exponential_factor
	homeostatic_low_state = -exponential_factor * log(1.0 - 0.05)
	homeostatic_high_state = -exponential_factor * log(1.0 - 0.95)
	input_gain = float(brain["input_gain"])
	input_min_weight = 1.0 / (1.0 + input_gain)

	neuron_index_by_spatial_index = PackedInt32Array()
	neuron_index_by_spatial_index.resize(neuron_count)
	modulatory_neighbor_indices = PackedInt32Array()
	modulatory_neighbor_indices.resize(neuron_count * 6)
	modulatory_neighbor_counts = PackedInt32Array()
	modulatory_neighbor_counts.resize(neuron_count)

	for neuron_index in neuron_count:
		var neuron: Dictionary = neurons[neuron_index]
		var position: Vector3i = neuron["position"]
		neuron_index_by_spatial_index[position.x + I * position.y + IJ * position.z] = neuron_index
		decay_rates[neuron_index] = 0.0 if neuron["input"] else exp(-12.0 / float(neuron["decay_factor"]))
		fatigues_used[neuron_index] = 0.0
		effective_thresholds_used[neuron_index] = 0.0 if neuron["input"] else float(neuron["activation_threshold"])
		modulatory_release_factors[neuron_index] = float(neuron["modulatory_release_factor"])
		modulatory_sensitivities[neuron_index] = float(neuron["modulatory_sensitivity"])
		modulatory_fields[neuron_index] = float(neuron["modulatory_field"])
		effective_modulations[neuron_index] = modulatory_sensitivities[neuron_index] * (1.0 - exp(-modulatory_fields[neuron_index]))

		if neuron["input"]:
			neuron["retention_factor"] = 0.0
			input_count += 1
		elif neuron["output"]:
			output_count += 1

		retention_factors[neuron_index] = float(neuron["retention_factor"])

	# Cache the fixed six-neighbor orthogonal mesh once. Runtime field updates then only
	# read these indices instead of rebuilding spatial positions and boundary checks.
	for neuron_index in neuron_count:
		var position: Vector3i = neurons[neuron_index]["position"]
		var spatial_index: int = position.x + I * position.y + IJ * position.z
		var neighbor_offset: int = neuron_index * 6
		var neighbor_count: int = 0

		if position.x > 0:
			modulatory_neighbor_indices[neighbor_offset + neighbor_count] = neuron_index_by_spatial_index[spatial_index - 1]
			neighbor_count += 1
		if position.x + 1 < I:
			modulatory_neighbor_indices[neighbor_offset + neighbor_count] = neuron_index_by_spatial_index[spatial_index + 1]
			neighbor_count += 1
		if position.y > 0:
			modulatory_neighbor_indices[neighbor_offset + neighbor_count] = neuron_index_by_spatial_index[spatial_index - I]
			neighbor_count += 1
		if position.y + 1 < J:
			modulatory_neighbor_indices[neighbor_offset + neighbor_count] = neuron_index_by_spatial_index[spatial_index + I]
			neighbor_count += 1
		if position.z > 0:
			modulatory_neighbor_indices[neighbor_offset + neighbor_count] = neuron_index_by_spatial_index[spatial_index - IJ]
			neighbor_count += 1
		if position.z + 1 < K:
			modulatory_neighbor_indices[neighbor_offset + neighbor_count] = neuron_index_by_spatial_index[spatial_index + IJ]
			neighbor_count += 1

		modulatory_neighbor_counts[neuron_index] = neighbor_count

	input_neuron_indices = PackedInt32Array()
	input_neuron_indices.resize(input_count)
	pending_input_signals = PackedFloat64Array()
	pending_input_signals.resize(input_count)
	input_signals_used = PackedFloat64Array()
	input_signals_used.resize(input_count)
	output_signals = PackedFloat64Array()
	output_signals.resize(output_count)

	var input_slot := 0
	var output_slot := 0

	for neuron_index in neuron_count:
		var neuron: Dictionary = neurons[neuron_index]

		if neuron["input"]:
			input_neuron_indices[input_slot] = neuron_index
			input_slot += 1
		elif neuron["output"]:
			output_slot_by_neuron_index[neuron_index] = output_slot
			output_slot += 1

		for connection in neuron["connections"]:
			var target: Vector3i = connection["target"]
			var target_index = neuron_index_by_spatial_index[target.x + I * target.y + IJ * target.z]
			connection["target_index"] = target_index
			incoming_connection_counts[target_index] += 1

	initial_connections.clear()
	initial_connections.resize(neuron_count)
	initial_modulatory_fields = PackedFloat64Array()
	initial_modulatory_fields.resize(neuron_count)

	for neuron_index in neuron_count:
		var neuron: Dictionary = neurons[neuron_index]
		initial_connections[neuron_index] = neuron["connections"].duplicate(true)
		initial_modulatory_fields[neuron_index] = float(neuron["modulatory_field"])

	structural_rng.randomize()
	initial_structural_rng_state = structural_rng.state
	added_connections_last_iteration = 0
	total_connections_formed = 0
	total_structural_plasticity_connections_formed = 0
	total_homeostatic_connections_formed = 0
	total_connections_destroyed = 0
	_update_modulatory_statistics()

	_update_initial_statistics()


## Restores the generated brain to its initial runtime state without rebuilding it from the genome.
## Initial connection dictionaries are restored so Hebbian changes, decay removals, eligibility traces,
## and future structural changes are discarded. Runtime signals, fatigue, activity traces,
## pending inputs, outputs, counters, and modulatory fields are also reset.
func reset() -> void:
	if brain.is_empty():
		return

	var neurons: Array = brain["neurons"]

	incoming_connection_counts.fill(0)

	for neuron_index in neurons.size():
		neurons[neuron_index]["connections"] = initial_connections[neuron_index].duplicate(true)
		neurons[neuron_index]["modulatory_field"] = initial_modulatory_fields[neuron_index]

		for connection in neurons[neuron_index]["connections"]:
			incoming_connection_counts[int(connection["target_index"])] += 1

	iteration_count = 0
	elapsed_time_seconds = 0.0
	delta_time_seconds = 0.0
	mean_signal = 0.0
	mean_modulatory_release = 0.0
	mean_modulatory_field = 0.0
	mean_effective_modulation = 0.0
	mean_eligibility_trace = 0.0
	added_connections_last_iteration = 0
	total_connections_formed = 0
	total_structural_plasticity_connections_formed = 0
	total_homeostatic_connections_formed = 0
	total_connections_destroyed = 0
	structural_rng.state = initial_structural_rng_state

	fatigues.fill(0.0)
	fatigues_used.fill(0.0)

	for neuron_index in neurons.size():
		effective_thresholds_used[neuron_index] = 0.0 if neurons[neuron_index]["input"] else float(neurons[neuron_index]["activation_threshold"])

	neuron_states.fill(0.0)
	neuron_states_used.fill(0.0)
	normalized_signals.fill(0.0)
	firing_states.fill(0.0)
	activity_traces.fill(0.0)
	incoming_signals.fill(0.0)
	pending_input_signals.fill(0.0)
	input_signals_used.fill(0.0)
	output_signals.fill(0.0)
	modulatory_releases.fill(0.0)
	next_modulatory_fields.fill(0.0)

	for neuron_index in neurons.size():
		modulatory_fields[neuron_index] = initial_modulatory_fields[neuron_index]
		effective_modulations[neuron_index] = modulatory_sensitivities[neuron_index] * (1.0 - exp(-modulatory_fields[neuron_index]))

	_update_modulatory_statistics()
	_update_initial_statistics()


## Queues the encoded scalar s = f(r) for one input neuron.
## The caller is responsible for mapping the raw input to [0, 1]. The latest queued value becomes
## that input neuron's state at the beginning of the next iteration.
func add_input_signal(input_index: int, input_signal: float) -> void:
	pending_input_signals[input_index] = input_signal


## Executes one brain iteration. The received delta is kept only for runtime timing statistics
## and does not affect the neuronal dynamics. Signal propagation is evaluated before connection decay,
## so a connection that dies in this iteration may still transmit once using its pre-decay weight.
## Returns the number of connections removed during the iteration.
func advance(delta_seconds: float) -> int:
	delta_time_seconds = delta_seconds
	elapsed_time_seconds += delta_seconds
	iteration_count += 1
	added_connections_last_iteration = 0

	if not profiling_enabled:
		var removed_connections = _propagate_signals()
		_apply_neuron_state_update()
		added_connections_last_iteration += _apply_structural_plasticity()
		return removed_connections

	var total_start_us: int = Time.get_ticks_usec()
	var stage_start_us: int = total_start_us

	var removed_connections = _propagate_signals()
	var current_us: int = Time.get_ticks_usec()
	profile_propagate_us = current_us - stage_start_us
	stage_start_us = current_us

	_apply_neuron_state_update()
	current_us = Time.get_ticks_usec()
	profile_state_update_us = current_us - stage_start_us
	stage_start_us = current_us

	added_connections_last_iteration += _apply_structural_plasticity()
	current_us = Time.get_ticks_usec()
	profile_structural_plasticity_us = current_us - stage_start_us
	profile_total_us = current_us - total_start_us

	return removed_connections


## Propagates signals synchronously through the current directional connections.
## Excitatory neurons add activity and inhibitory neurons subtract activity according to their polarity.
## Each non-input neuron's current fatigue is used directly in T_eff = T * (1 + rho).
## If the neuron fires, rho is reset to the genetically determined maximum fatigue F_0.4(x),
## cached in refractory_strength. Otherwise, rho is reduced according to rho <- rho / 2.
## Current firing also produces the local modulatory release r_t = Q * sigma(s_t).
## The new modulatory field c_t is computed synchronously from c_(t-1) and r_t.
## Every existing connection is then processed once. If it transmits, propagation uses the pre-update
## weight, followed by Hebbian reinforcement. Eligibility is updated next, then postsynaptic endogenous
## modulation is applied, and decay is applied last. Internal-to-output connections scale passive
## decay by the target output neuron's recent-activity trace R_t. This preserves the article-defined
## order while avoiding a second full traversal of the connections.
## Returns the number of dead connections removed during this iteration.
func _propagate_signals() -> int:
	var neurons: Array = brain["neurons"]
	var neuron_count = neurons.size()
	var signal_sum = 0.0
	var modulatory_release_sum = 0.0
	var recovery_factor = 0.5
	var removed_connections = 0
	var replacement_connections = 0
	var weight_sum = 0.0
	var eligibility_sum = 0.0
	var one_minus_trace_persistence = 1.0 - trace_persistence
	var orphaned_targets = PackedInt32Array()
	var profile_stage_start_us: int = Time.get_ticks_usec() if profiling_enabled else 0

	incoming_signals.fill(0.0)
	normalized_signals.fill(0.0)
	firing_states.fill(0.0)
	output_signals.fill(0.0)
	modulatory_releases.fill(0.0)

	input_signals_used.fill(0.0)

	for input_index in input_neuron_indices.size():
		var input_signal = pending_input_signals[input_index]
		input_signals_used[input_index] = input_signal
		neuron_states[input_neuron_indices[input_index]] = input_signal
		pending_input_signals[input_index] = 0.0

	# First pass: determine every neuron's current normalized activity and firing decision.
	# sigma(s) is still needed for modulation and eligibility even though inputs transmit psi*s directly.
	# This must finish before connections are processed because eligibility uses both endpoints at time t.
	for neuron_index in neuron_count:
		var neuron: Dictionary = neurons[neuron_index]

		if not neuron["input"]:
			fatigues_used[neuron_index] = fatigues[neuron_index]
			effective_thresholds_used[neuron_index] = float(neuron["activation_threshold"]) * (1.0 + fatigues[neuron_index])
			activity_traces[neuron_index] *= trace_persistence

		neuron_states_used[neuron_index] = neuron_states[neuron_index]
		var state = neuron_states_used[neuron_index]

		if state <= 0.0:
			if not neuron["input"]:
				fatigues[neuron_index] *= recovery_factor
			continue

		var normalized_signal = 1.0 - exp(-state * inverse_exponential_factor)
		normalized_signals[neuron_index] = normalized_signal
		signal_sum += normalized_signal

		# The neuronal activity trace uses the same persistence parameter mu as eligibility traces.
		if not neuron["input"]:
			activity_traces[neuron_index] += one_minus_trace_persistence * normalized_signal

		# Inputs do not use the firing threshold or refractory state. Any positive encoded input
		# is transmitted during this iteration, while sigma(s) remains available to the other mechanisms.
		if neuron["input"]:
			firing_states[neuron_index] = 1.0
			var input_modulatory_release = modulatory_release_factors[neuron_index] * normalized_signal
			modulatory_releases[neuron_index] = input_modulatory_release
			modulatory_release_sum += input_modulatory_release
			continue

		var effective_threshold = effective_thresholds_used[neuron_index]

		if normalized_signal <= effective_threshold:
			fatigues[neuron_index] *= recovery_factor
			continue

		firing_states[neuron_index] = 1.0
		var modulatory_release = modulatory_release_factors[neuron_index] * normalized_signal
		modulatory_releases[neuron_index] = modulatory_release
		modulatory_release_sum += modulatory_release
		fatigues[neuron_index] = refractory_strength

		var output_slot = output_slot_by_neuron_index[neuron_index]

		if output_slot >= 0:
			output_signals[output_slot] = normalized_signal

	if profiling_enabled:
		var profile_now_us: int = Time.get_ticks_usec()
		profile_prepare_and_firing_us = profile_now_us - profile_stage_start_us
		profile_stage_start_us = profile_now_us

	_update_modulatory_field()

	if profiling_enabled:
		var profile_now_us: int = Time.get_ticks_usec()
		profile_modulatory_field_us = profile_now_us - profile_stage_start_us
		profile_stage_start_us = profile_now_us

	# Second pass: process each existing connection exactly once in the article-defined order:
	# transmission with the old weight -> Hebbian -> eligibility -> modulation -> decay.
	for source_index in neuron_count:
		var source: Dictionary = neurons[source_index]
		var connections: Array = source["connections"]

		if connections.is_empty():
			continue

		var source_fired = firing_states[source_index] > 0.0
		var source_normalized_signal = normalized_signals[source_index]
		var source_is_input = bool(source["input"])
		var source_signal = input_gain * neuron_states[source_index] if source_is_input else source_normalized_signal
		var polarity = float(source["polarity_factor"])
		var hebbian_rate = float(source["hebbian_plasticity_rate"])
		var decrement = decay_rates[source_index]
		var minimum_weight = input_min_weight if source_is_input else 0.0

		for connection_index in range(connections.size() - 1, -1, -1):
			var connection: Dictionary = connections[connection_index]
			var target_index = int(connection["target_index"])
			var weight = float(connection["weight"])
			var instantaneous_eligibility = 0.0

			if source_fired:
				incoming_signals[target_index] += polarity * weight * source_signal
				weight += hebbian_rate * (1.0 - weight)
				instantaneous_eligibility = source_normalized_signal * normalized_signals[target_index]

			var eligibility = (
				trace_persistence * float(connection["eligibility_trace"])
				+ one_minus_trace_persistence * instantaneous_eligibility
			)
			connection["eligibility_trace"] = eligibility

			var modulation = effective_modulations[target_index]

			if modulation >= 0.0:
				weight += modulation * eligibility * (1.0 - weight)
			else:
				weight += modulation * eligibility * weight

			var connection_decrement = decrement

			# Connections from internal neurons to outputs decay in proportion to the
			# output neuron's recent activity trace. Internal-to-internal decay is unchanged.
			if not source_is_input and output_slot_by_neuron_index[target_index] >= 0:
				connection_decrement *= activity_traces[target_index]

			weight = maxf(weight - connection_decrement, minimum_weight)

			if weight <= 0.0:
				connections.remove_at(connection_index)
				removed_connections += 1
				incoming_connection_counts[target_index] -= 1

				if incoming_connection_counts[target_index] == 0 and not neurons[target_index]["input"]:
					orphaned_targets.append(target_index)
			else:
				connection["weight"] = weight
				weight_sum += weight
				eligibility_sum += eligibility

	if profiling_enabled:
		var profile_now_us: int = Time.get_ticks_usec()
		profile_connections_us = profile_now_us - profile_stage_start_us
		profile_stage_start_us = profile_now_us

	var replacement_weight = float(brain["structural_plasticity"]) * 0.25

	for orphan_index in orphaned_targets.size():
		if _replace_last_incoming_connection(orphaned_targets[orphan_index], replacement_weight):
			replacement_connections += 1

	if profiling_enabled:
		var profile_now_us: int = Time.get_ticks_usec()
		profile_homeostatic_us = profile_now_us - profile_stage_start_us
		profile_stage_start_us = profile_now_us

	total_connections += replacement_connections - removed_connections
	weight_sum += replacement_weight * replacement_connections
	mean_weight = 0.0 if total_connections == 0 else weight_sum / total_connections
	mean_eligibility_trace = 0.0 if total_connections == 0 else eligibility_sum / total_connections

	if removed_connections > 0:
		total_connections_destroyed += removed_connections

	if replacement_connections > 0:
		total_connections_formed += replacement_connections
		total_homeostatic_connections_formed += replacement_connections
		added_connections_last_iteration += replacement_connections

	if removed_connections > 0 or replacement_connections > 0:
		_update_connection_degree_statistics()

	mean_signal = 0.0 if neuron_count == 0 else signal_sum / neuron_count
	mean_modulatory_release = 0.0 if neuron_count == 0 else modulatory_release_sum / neuron_count

	if profiling_enabled:
		profile_propagation_statistics_us = Time.get_ticks_usec() - profile_stage_start_us

	return removed_connections


## Replaces the last incoming connection of an internal or output neuron after that connection dies.
## Candidate sources are restricted to internal neurons in the target's normal k-2 to k+2 connection region.
## The source with the largest recent-activity trace R_t is selected. Exact ties are
## resolved uniformly at random. Returns true when a replacement connection is created.
func _replace_last_incoming_connection(target_index: int, new_weight: float) -> bool:
	var neurons: Array = brain["neurons"]
	var size: Vector3i = brain["size"]
	var I = size.x
	var J = size.y
	var K = size.z
	var IJ = I * J
	var target: Dictionary = neurons[target_index]
	var target_position: Vector3i = target["position"]
	var source_k_min = maxi(0, target_position.z - 2)
	var source_k_max = mini(K - 1, target_position.z + 2)
	var best_source_index = -1
	var best_activity = -1.0
	var tie_count = 0

	for k in range(source_k_min, source_k_max + 1):
		for j in J:
			for i in I:
				var spatial_index = i + I * j + IJ * k
				var source_index = neuron_index_by_spatial_index[spatial_index]

				if source_index == target_index:
					continue

				var source: Dictionary = neurons[source_index]

				# Lifetime reconnection is kept internal to the brain. Inputs and outputs are not candidates.
				if source["input"] or source["output"]:
					continue


				var activity = activity_traces[source_index]

				if activity > best_activity:
					best_activity = activity
					best_source_index = source_index
					tie_count = 1
				elif activity == best_activity:
					tie_count += 1

					if structural_rng.randi_range(1, tie_count) == 1:
						best_source_index = source_index

	if best_source_index < 0:
		return false

	neurons[best_source_index]["connections"].append({
		"target": target_position,
		"target_index": target_index,
		"weight": new_weight,
		"eligibility_trace": 0.0,
	})
	incoming_connection_counts[target_index] += 1
	return true


## Updates c_t from c_(t-1) using the six-neighbor orthogonal mesh and the current local release.
## Neighbor indices are precomputed when the brain is associated with the runtime. The previous field
## is read only from modulatory_fields while next_modulatory_fields receives c_t, so spatial mixing
## remains synchronous and cannot depend on neuron traversal order.
func _update_modulatory_field() -> void:
	var neuron_count = modulatory_fields.size()
	var one_minus_spread = 1.0 - modulatory_spread
	var field_sum = 0.0
	var effective_modulation_sum = 0.0

	for neuron_index in neuron_count:
		var neighbor_offset: int = neuron_index * 6
		var neighbor_count: int = modulatory_neighbor_counts[neuron_index]
		var neighbor_sum = 0.0

		for neighbor_slot in neighbor_count:
			neighbor_sum += modulatory_fields[modulatory_neighbor_indices[neighbor_offset + neighbor_slot]]

		var neighbor_average = neighbor_sum / neighbor_count
		var new_field = (
			modulatory_persistence
			* (one_minus_spread * modulatory_fields[neuron_index] + modulatory_spread * neighbor_average)
			+ modulatory_releases[neuron_index]
		)
		var effective_modulation = modulatory_sensitivities[neuron_index] * (1.0 - exp(-new_field))

		next_modulatory_fields[neuron_index] = new_field
		effective_modulations[neuron_index] = effective_modulation
		field_sum += new_field
		effective_modulation_sum += effective_modulation

	var recycled_field_buffer = modulatory_fields
	modulatory_fields = next_modulatory_fields
	next_modulatory_fields = recycled_field_buffer

	mean_modulatory_field = 0.0 if neuron_count == 0 else field_sum / neuron_count
	mean_effective_modulation = 0.0 if neuron_count == 0 else effective_modulation_sum / neuron_count


## Advances every neuronal state from t to t+1 after all time-t firing and synaptic updates are complete.
## Incoming signals generated at t enter without being multiplied by retention until the next iteration.
func _apply_neuron_state_update() -> void:
	const HOMEOSTATIC_LOW_ACTIVITY = 0.05
	const HOMEOSTATIC_HIGH_ACTIVITY = 0.95
	var neurons: Array = brain["neurons"]

	for neuron_index in neuron_states.size():
		var state = maxf(
			0.0,
			retention_factors[neuron_index] * neuron_states[neuron_index] + incoming_signals[neuron_index]
		)

		if not neurons[neuron_index]["input"]:
			var activity = activity_traces[neuron_index]

			if activity < HOMEOSTATIC_LOW_ACTIVITY and state < homeostatic_low_state:
				state += (HOMEOSTATIC_LOW_ACTIVITY - activity) / HOMEOSTATIC_LOW_ACTIVITY * (homeostatic_low_state - state)
			elif activity > HOMEOSTATIC_HIGH_ACTIVITY and state > homeostatic_high_state:
				state -= (activity - HOMEOSTATIC_HIGH_ACTIVITY) / (1.0 - HOMEOSTATIC_HIGH_ACTIVITY) * (state - homeostatic_high_state)

		neuron_states[neuron_index] = state


## Attempts to create at most one new outgoing connection per firing internal neuron.
## A connection is created when the sum of the source and target recent-activity traces exceeds P(x).
## Returns the number of connections created in this iteration.
func _apply_structural_plasticity() -> int:
	var neurons: Array = brain["neurons"]
	var neuron_count = neurons.size()
	var plasticity = float(brain["structural_plasticity"])
	var new_weight = plasticity * 0.25
	var added_connections = 0

	for source_index in neuron_count:
		var source: Dictionary = neurons[source_index]

		# Inputs and outputs do not create new outgoing connections during structural plasticity.
		if source["input"] or source["output"]:
			continue

		# Structural plasticity is triggered only when the source neuron fires in the current iteration.
		if firing_states[source_index] <= 0.0:
			continue

		var connections: Array = source["connections"]
		var target_index = _random_structural_target_index(source_index)

		if target_index < 0:
			continue

		var target: Dictionary = neurons[target_index]

		# Input neurons receive only external signals.
		if target["input"]:
			continue

		var already_connected = false

		for connection in connections:
			if int(connection["target_index"]) == target_index:
				already_connected = true
				break

		# One random attempt per iteration: do not choose another target if this one already exists.
		if already_connected:
			continue

		if activity_traces[source_index] + activity_traces[target_index] <= plasticity:
			continue

		connections.append({
			"target": target["position"],
			"target_index": target_index,
			"weight": new_weight,
			"eligibility_trace": 0.0,
		})
		incoming_connection_counts[target_index] += 1
		added_connections += 1

	if added_connections > 0:
		var previous_weight_sum = mean_weight * total_connections
		var previous_eligibility_sum = mean_eligibility_trace * total_connections
		total_connections += added_connections
		total_connections_formed += added_connections
		total_structural_plasticity_connections_formed += added_connections
		mean_weight = (previous_weight_sum + new_weight * added_connections) / total_connections
		mean_eligibility_trace = previous_eligibility_sum / total_connections
		_update_connection_degree_statistics()

	return added_connections

## Selects one random neuron uniformly from the source neuron's geometric connection region.
## The source neuron itself is excluded without retrying or allocating a candidate array.
func _random_structural_target_index(source_index: int) -> int:
	var size: Vector3i = brain["size"]
	var I = size.x
	var J = size.y
	var K = size.z
	var IJ = I * J
	var source_position: Vector3i = brain["neurons"][source_index]["position"]
	var target_k_min = maxi(0, source_position.z - 2)
	var target_k_max = mini(K - 1, source_position.z + 2)
	var slice_count = target_k_max - target_k_min + 1
	var candidate_count = IJ * slice_count - 1

	if candidate_count <= 0:
		return -1

	var source_slot = (
		source_position.x
		+ I * source_position.y
		+ IJ * (source_position.z - target_k_min)
	)
	var random_slot = structural_rng.randi_range(0, candidate_count - 1)

	# Map [0, candidate_count) onto the complete local region while skipping the source slot.
	if random_slot >= source_slot:
		random_slot += 1

	var local_k = int(random_slot / IJ)
	var within_slice = random_slot % IJ
	var u = within_slice % I
	var v = int(within_slice / I)
	var w = target_k_min + local_k
	var spatial_index = u + I * v + IJ * w

	return neuron_index_by_spatial_index[spatial_index]


## Calculates the initial connection statistics once when a brain is associated with the runtime.
func _update_initial_statistics() -> void:
	var weight_sum = 0.0
	var eligibility_sum = 0.0
	total_connections = 0

	for neuron in brain["neurons"]:
		for connection in neuron["connections"]:
			weight_sum += float(connection["weight"])
			eligibility_sum += float(connection["eligibility_trace"])
			total_connections += 1

	mean_weight = 0.0 if total_connections == 0 else weight_sum / total_connections
	mean_eligibility_trace = 0.0 if total_connections == 0 else eligibility_sum / total_connections
	_update_connection_degree_statistics()


## Recalculates aggregate modulatory statistics from the current runtime field.
func _update_modulatory_statistics() -> void:
	var neuron_count = modulatory_fields.size()
	var field_sum = 0.0
	var effective_modulation_sum = 0.0

	for neuron_index in neuron_count:
		field_sum += modulatory_fields[neuron_index]
		effective_modulation_sum += effective_modulations[neuron_index]

	mean_modulatory_field = 0.0 if neuron_count == 0 else field_sum / neuron_count
	mean_effective_modulation = 0.0 if neuron_count == 0 else effective_modulation_sum / neuron_count


## Updates outgoing connection statistics after the topology changes.
func _update_connection_degree_statistics() -> void:
	var outgoing_neuron_count := 0
	var outgoing_connection_count := 0
	var first_outgoing_neuron := true

	for neuron in brain["neurons"]:
		if neuron["output"]:
			continue

		var out_degree: int = neuron["connections"].size()
		outgoing_neuron_count += 1
		outgoing_connection_count += out_degree

		if first_outgoing_neuron:
			min_out_degree = out_degree
			max_out_degree = out_degree
			first_outgoing_neuron = false
		else:
			min_out_degree = mini(min_out_degree, out_degree)
			max_out_degree = maxi(max_out_degree, out_degree)

	mean_outgoing_connections = 0 if outgoing_neuron_count == 0 else int(float(outgoing_connection_count) / outgoing_neuron_count)


## Detaches the current brain and resets all runtime state.
func clear() -> void:
	brain = {}
	iteration_count = 0
	elapsed_time_seconds = 0.0
	delta_time_seconds = 0.0
	mean_weight = 0.0
	mean_signal = 0.0
	total_connections = 0
	mean_outgoing_connections = 0
	min_out_degree = 0
	max_out_degree = 0
	added_connections_last_iteration = 0
	total_connections_formed = 0
	total_structural_plasticity_connections_formed = 0
	total_homeostatic_connections_formed = 0
	total_connections_destroyed = 0
	mean_modulatory_release = 0.0
	mean_modulatory_field = 0.0
	mean_effective_modulation = 0.0
	mean_eligibility_trace = 0.0
	decay_rates = PackedFloat64Array()
	retention_factors = PackedFloat64Array()
	fatigues = PackedFloat64Array()
	fatigues_used = PackedFloat64Array()
	effective_thresholds_used = PackedFloat64Array()
	modulatory_release_factors = PackedFloat64Array()
	modulatory_sensitivities = PackedFloat64Array()
	neuron_states = PackedFloat64Array()
	neuron_states_used = PackedFloat64Array()
	normalized_signals = PackedFloat64Array()
	firing_states = PackedFloat64Array()
	activity_traces = PackedFloat64Array()
	incoming_signals = PackedFloat64Array()
	incoming_connection_counts = PackedInt32Array()
	pending_input_signals = PackedFloat64Array()
	input_signals_used = PackedFloat64Array()
	output_signals = PackedFloat64Array()
	modulatory_releases = PackedFloat64Array()
	modulatory_fields = PackedFloat64Array()
	next_modulatory_fields = PackedFloat64Array()
	effective_modulations = PackedFloat64Array()
	input_neuron_indices = PackedInt32Array()
	output_slot_by_neuron_index = PackedInt32Array()
	neuron_index_by_spatial_index = PackedInt32Array()
	modulatory_neighbor_indices = PackedInt32Array()
	modulatory_neighbor_counts = PackedInt32Array()
	initial_connections.clear()
	initial_modulatory_fields = PackedFloat64Array()
	structural_rng = RandomNumberGenerator.new()
	initial_structural_rng_state = 0
	modulatory_persistence = 0.0
	modulatory_spread = 0.0
	trace_persistence = 0.0
	refractory_strength = 0.0
	inverse_exponential_factor = 1.0
	homeostatic_low_state = 0.0
	homeostatic_high_state = 0.0
	input_gain = 1.0
