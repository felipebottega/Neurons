class_name BrainBuilder
extends RefCounted


var math = Functions.new()

## Builds the initial brain structure from a genome.
## [param genome] contains the genes used to build the brain.
## [param size] defines the brain dimensions.
## [param num_inputs] defines the number of input neurons.
## [param num_outputs] defines the number of output neurons.
## [param progress_callback] optionally receives the build percentage from 0 to 100.
## Returns a dictionary containing the brain structure.
func build(genome: Genome, size: Vector3i, num_inputs: int, num_outputs: int, progress_callback: Callable = Callable(), connections_density: float = 1.0) -> Dictionary:
	if size.x < 2 or size.y < 2 or size.z < 2 or (is_nan(connections_density) or is_inf(connections_density)) or connections_density <= 0.0:
		push_error("Brain dimensions I, J, K must all be at least 2.")
		return {}

	_report_progress(progress_callback, 0)

	var beta := math.Beta(genome.beta, size.x, size.y)
	var modulatory_dynamics := _modulatory_dynamics(genome.modulatory_dynamics)
	var region_extent: float = math.S(genome.initial_search_region)
	var initial_radii := Vector3i(mini(size.x - 1, ceili(region_extent * size.x)), mini(size.y - 1, ceili(region_extent * size.y)), mini(size.z - 1, ceili(region_extent * size.z)))
	var brain = {
		"size": size,
		"beta": beta,
		"max_connections": maxi(1, floori(connections_density * math.F(beta, genome.max_connections))),
		"connections_density": connections_density,
		"initial_search_extent": region_extent,
		"initial_search_radius_i": initial_radii.x,
		"initial_search_radius_j": initial_radii.y,
		"initial_search_radius_k": initial_radii.z,
		"refractory_strength": math.F(0.4, genome.refractory_strength),
		"exponential_factor": math.F(0.4, genome.exponential_factor),
		"structural_plasticity": math.P(genome.structural_plasticity),
		"lambda": modulatory_dynamics.x,
		"nu": modulatory_dynamics.y,
		"mu": modulatory_dynamics.z,
		"neurons": []
	}

	brain["neurons"] = _create_neurons(size, genome, progress_callback)
	_set_input_output_neurons(brain["neurons"], size, num_inputs, num_outputs, progress_callback)
	_create_connections(brain, genome.connections, genome.initial_search_region, progress_callback)
	brain["valid"] = _clear_invalid_connections(brain, genome.connections, genome.input_influence, progress_callback)
	_finalize_connection_weights(brain)
	_report_progress(progress_callback, 100)

	return brain

## Creates all neurons in the brain mesh.
## [param size] defines the brain dimensions.
## [param genome] contains the genes used to define neuron parameters.
## Returns an array containing all neurons.
func _create_neurons(size: Vector3i, genome: Genome, progress_callback: Callable) -> Array:
	var neurons = []
	var I = size.x
	var J = size.y
	var K = size.z
	var total_neurons = I * J * K
	var completed = 0
	var report_interval = maxi(1, int(total_neurons / 100.0))
	var first_polarity = math.A(0, 0, 0, genome.polarity_factor, I, J, K)
	var retention_normalization = 1.0 - exp(-2.0)

	for i in I:
		for j in J:
			for k in K:
				neurons.append({
					"position": Vector3i(i, j, k),
					"input": false,
					"output": false,
					"activation_threshold": (1.0 + math.A(i, j, k, genome.activation_threshold, I, J, K)) / 2.0,
					"decay_factor": math.D(i, j, k, genome.decay_factor, I, J, K),
					"retention_factor": (1.0 - exp(-1.0 - math.A(i, j, k, genome.retention_factor, I, J, K))) / retention_normalization,
					"polarity_factor": 1 if math.A(i, j, k, genome.polarity_factor, I, J, K) >= first_polarity else -1,
					"hebbian_plasticity_rate": (1.0 + math.A(i, j, k, genome.hebbian_plasticity_rate, I, J, K)) / 4.0,
					"modulatory_release_factor": (1.0 + math.A(i, j, k, genome.modulatory_release_factor, I, J, K)) / 2.0,
					"modulatory_sensitivity": math.A(i, j, k, genome.modulatory_sensitivity, I, J, K),
					"modulatory_field": 0.0,
					"connections_value": math.A(i, j, k, genome.connections, I, J, K),
					"connections": []
				})

				completed += 1
				
				if completed % report_interval == 0 or completed == total_neurons:
					_report_progress(progress_callback, int(20.0 * completed / total_neurons))

	return neurons

## Selects the input and output neurons.
## Neurons are sorted independently by connection energy inside each slice and then
## interleaved by rank: the lowest-energy neuron of every slice comes first, followed
## by the second-lowest-energy neuron of every slice, and so on.
## [param neurons] contains the brain neurons.
## [param size] defines the brain dimensions.
## [param num_inputs] defines the number of input neurons.
## [param num_outputs] defines the number of output neurons.
## Returns nothing.
func _set_input_output_neurons(neurons: Array, size: Vector3i, num_inputs: int, num_outputs: int, progress_callback: Callable) -> void:
	var K = size.z
	var neurons_per_slice = size.x * size.y
	var slices: Array = []
	slices.resize(K)

	for k in K:
		slices[k] = []

	for neuron in neurons:
		slices[neuron["position"].z].append(neuron)

	for k in K:
		slices[k].sort_custom(func(a, b): return a["connections_value"] < b["connections_value"])

	neurons.clear()

	for rank in neurons_per_slice:
		for k in K:
			neurons.append(slices[k][rank])

	for i in num_inputs:
		neurons[i]["input"] = true
		neurons[i]["activation_threshold"] = 0.0
		neurons[i]["retention_factor"] = 0.0
		neurons[i]["polarity_factor"] = 1

	for i in num_outputs:
		neurons[-1 - i]["output"] = true
		
	_report_progress(progress_callback, 30)

## Creates the initial connections and their weights.
## Connection generation is distributed across the WorkerThreadPool, one source neuron per task element.
## [param brain] contains the brain structure.
## [param gene] is used to create the connections.
## Returns nothing.
func _create_connections(brain: Dictionary, gene: Array, region_gene: Array, progress_callback: Callable) -> void:
	var size: Vector3i = brain["size"]
	var neurons: Array = brain["neurons"]
	var total: int = neurons.size()
	var I: int = size.x
	var IJ: int = I * size.y
	var connection_values := PackedFloat64Array()
	connection_values.resize(total)
	for neuron in neurons:
		var p: Vector3i = neuron["position"]
		connection_values[p.x + I * p.y + IJ * p.z] = neuron["connections_value"]
	var offsets: PackedInt32Array = _initial_offset_table(
		region_gene, size.x, size.y, size.z,
		brain["initial_search_radius_i"], brain["initial_search_radius_j"],
		brain["initial_search_radius_k"], brain["max_connections"])
	var results: Array = []
	results.resize(total)
	var coefficients: Vector4 = math.H_coefficients(gene)
	var task_id: int = WorkerThreadPool.add_group_task(
		_create_connections_for_neuron.bind(
			neurons, size, int(brain["max_connections"]),
			Vector3i(brain["initial_search_radius_i"], brain["initial_search_radius_j"], brain["initial_search_radius_k"]),
			coefficients, connection_values, offsets, results), total)
	while not WorkerThreadPool.is_group_task_completed(task_id):
		_report_progress(progress_callback, 30 + int(45.0 * WorkerThreadPool.get_group_processed_element_count(task_id) / total))
		OS.delay_msec(10)
	WorkerThreadPool.wait_for_group_task_completion(task_id)
	for index in total:
		neurons[index]["connections"] = results[index]
	_report_progress(progress_callback, 75)


# Same local region and signed-offset candidate-test budget as brain_connections.glsl.
# The second pass deliberately DOES NOT rank or retry failed attempts.
func _create_connections_for_neuron(neuron_index: int, neurons: Array, size: Vector3i, max_connections: int, radii: Vector3i, coefficients: Vector4, values: PackedFloat64Array, offsets: PackedInt32Array, results: Array) -> void:
	if max_connections <= 0 or bool(neurons[neuron_index]["output"]):
		results[neuron_index] = []
		return
	var pos: Vector3i = neurons[neuron_index]["position"]
	var I: int = size.x
	var J: int = size.y
	var K: int = size.z
	var IJ: int = I * J
	var origin: int = pos.x + I * pos.y + IJ * pos.z
	var energy: float = values[origin]
	var candidates: Array = []
	for w in range(maxi(0, pos.z - radii.z), mini(K - 1, pos.z + radii.z) + 1):
		for v in range(maxi(0, pos.y - radii.y), mini(J - 1, pos.y + radii.y) + 1):
			for u in range(maxi(0, pos.x - radii.x), mini(I - 1, pos.x + radii.x) + 1):
				var spatial: int = u + I * v + IJ * w
				if spatial == origin:
					continue
				var h: float = math.H(energy, values[spatial], coefficients)
				if h > 0.0:
					candidates.append({"spatial": spatial, "h": h})
	candidates.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return a["h"] > b["h"] or (a["h"] == b["h"] and a["spatial"] < b["spatial"]))
	var cap: int = mini(max_connections, neurons.size() - 1)
	var retained: Array = []
	var retained_ids: Dictionary = {}
	for idx in mini(cap, candidates.size()):
		var item: Dictionary = candidates[idx]
		var s: int = item["spatial"]
		retained.append({"target": Vector3i(s % I, (s / I) % J, s / IJ), "weight": item["h"], "eligibility_trace": 0.0, "_initial_local": true})
		retained_ids[s] = true
	# The hash tracks only second-pass targets, as in the shader.
	var tested_ids: Dictionary = {}
	var attempts: int = 0
	for t in offsets.size() / 4:
		if retained.size() >= cap or attempts >= max_connections:
			break
		var dx: int = offsets[t * 4]
		var dy: int = offsets[t * 4 + 1]
		var dz: int = offsets[t * 4 + 2]
		for sign_mask in 8:
			if retained.size() >= cap or attempts >= max_connections:
				break
			attempts += 1
			var u: int = pos.x + (dx if (sign_mask & 4) == 0 else -dx)
			var v: int = pos.y + (dy if (sign_mask & 2) == 0 else -dy)
			var w: int = pos.z + (dz if (sign_mask & 1) == 0 else -dz)
			if u < 0 or v < 0 or w < 0 or u >= I or v >= J or w >= K:
				continue
			var s: int = u + I * v + IJ * w
			if s == origin or tested_ids.has(s):
				continue
			tested_ids[s] = true
			var h: float = math.H(energy, values[s], coefficients)
			if h > 0.0 and not retained_ids.has(s):
				retained.append({"target": Vector3i(u, v, w), "weight": h, "eligibility_trace": 0.0, "_initial_local": false})
	results[neuron_index] = retained


## Removes connections forbidden for input and output neurons, rescues isolated neurons
## when a valid connection can still be created, and computes the fixed input gain.
## [param brain] contains the generated brain structure.
## [param gene] is the connections gene used to evaluate rescue candidates.
## [param input_influence_gene] determines the fixed amplification of input signals.
## Returns true when every internal/output neuron has incoming connectivity and every input has outgoing connectivity.
func _clear_invalid_connections(brain: Dictionary, gene: Array, input_influence_gene: Array, progress_callback: Callable) -> bool:
	var neurons: Array = brain["neurons"]
	var size: Vector3i = brain["size"]
	var I: int = size.x
	var J: int = size.y
	var IJ: int = I * J
	var total_neurons: int = neurons.size()
	var neuron_by_spatial_index: Array = []
	neuron_by_spatial_index.resize(total_neurons)
	var incoming_connections = PackedInt32Array()
	incoming_connections.resize(total_neurons)

	for neuron in neurons:
		var position: Vector3i = neuron["position"]
		var spatial_index: int = position.x + I * position.y + IJ * position.z
		neuron_by_spatial_index[spatial_index] = neuron

	var completed = 0
	var report_interval = maxi(1, int(total_neurons / 100.0))

	for neuron in neurons:
		var valid_connections = []

		for connection in neuron["connections"]:
			var target_position: Vector3i = connection["target"]
			var target_spatial_index: int = target_position.x + I * target_position.y + IJ * target_position.z
			var target: Dictionary = neuron_by_spatial_index[target_spatial_index]

			# Input neurons receive only external signals and can send only to internal neurons.
			if target["input"]:
				continue

			if neuron["input"] and target["output"]:
				continue

			# Output neurons receive only from internal neurons and do not emit neural signals.
			if neuron["output"]:
				continue

			valid_connections.append(connection)
			incoming_connections[target_spatial_index] += 1

		neuron["connections"] = valid_connections
		completed += 1

		if completed % report_interval == 0 or completed == total_neurons:
			_report_progress(progress_callback, 95 + int(3.0 * completed / total_neurons))

	# Count edges that survived role filtering, before rescue.
	var local_count: int = 0
	var total_initial_count: int = 0
	for source in neurons:
		for connection in source["connections"]:
			total_initial_count += 1
			if connection.get("_initial_local", false):
				local_count += 1
			connection.erase("_initial_local")
	brain["initial_connections_first_pass"] = local_count
	brain["initial_connections_second_pass"] = total_initial_count - local_count
	brain["initial_local_connections"] = local_count
	brain["initial_additional_connections"] = total_initial_count - local_count

	var isolated_internal_before = 0
	var isolated_output_before = 0
	var isolated_input_before = 0

	for neuron in neurons:
		var position: Vector3i = neuron["position"]
		var spatial_index: int = position.x + I * position.y + IJ * position.z
		var incoming_count: int = incoming_connections[spatial_index]

		if neuron["input"]:
			if neuron["connections"].is_empty():
				isolated_input_before += 1
		elif neuron["output"]:
			if incoming_count == 0:
				isolated_output_before += 1
		elif incoming_count == 0:
			isolated_internal_before += 1

	brain["isolated_before_rescue"] = {
		"internal": isolated_internal_before,
		"output": isolated_output_before,
		"input": isolated_input_before
	}

	var rescue_success = _rescue_isolated_neurons(
		brain,
		gene,
		neuron_by_spatial_index,
		incoming_connections,
		progress_callback
	)

	var isolated_internal = 0
	var isolated_output = 0
	var isolated_input = 0
	var internal_incoming_sum = 0
	var internal_neuron_count = 0

	for neuron in neurons:
		var position: Vector3i = neuron["position"]
		var spatial_index: int = position.x + I * position.y + IJ * position.z
		var incoming_count: int = incoming_connections[spatial_index]

		if neuron["input"]:
			if neuron["connections"].is_empty():
				isolated_input += 1
		elif neuron["output"]:
			if incoming_count == 0:
				isolated_output += 1
		else:
			if incoming_count == 0:
				isolated_internal += 1
			internal_incoming_sum += incoming_count
			internal_neuron_count += 1

	brain["isolated_after_rescue"] = {
		"internal": isolated_internal,
		"output": isolated_output,
		"input": isolated_input
	}

	# psi is fixed at generation time and uses the initial mean incoming degree of internal neurons.
	var mean_internal_incoming = 0.0 if internal_neuron_count == 0 else float(internal_incoming_sum) / internal_neuron_count
	brain["input_gain"] = (1.0 + sin(math.G(true, true, input_influence_gene))) * mean_internal_incoming

	return rescue_success and isolated_internal == 0 and isolated_output == 0 and isolated_input == 0


## Developmental rescue: direct 6-neighbor lattice, ranked by positive H^2.
## Receiver rescue precedes input rescue. No other initial connection is removed.
## The temporary rescue out-degree ceiling is ceil(1.01 * N_max) + 1.
func _rescue_isolated_neurons(brain: Dictionary, gene: Array, neuron_by_spatial_index: Array, incoming_connections: PackedInt32Array, progress_callback: Callable) -> bool:
	var neurons: Array = brain["neurons"]
	var size: Vector3i = brain["size"]
	var I: int = size.x
	var J: int = size.y
	var K: int = size.z
	var IJ: int = I * J
	var max_connections: int = mini(neurons.size() - 1, ceili(1.01 * float(brain["max_connections"])) + 1)
	var coefficients: Vector4 = math.H_coefficients(gene)
	var reasons := {"no_structural_candidate": 0, "no_positive_h_candidate": 0, "positive_h_candidates_saturated": 0, "input_source_saturated": 0}
	var receiver_candidates: int = 0
	var input_candidates: int = 0
	var added: int = 0
	var valid: bool = true
	for phase in 2:
		for neuron in neurons:
			var pos: Vector3i = neuron["position"]
			var spatial: int = pos.x + I * pos.y + IJ * pos.z
			if phase == 0:
				if neuron["input"] or incoming_connections[spatial] != 0:
					continue
			else:
				if not neuron["input"] or not neuron["connections"].is_empty():
					continue
			var candidates: Array = []
			var structural_count := 0
			for displacement in [Vector3i(0,0,-1), Vector3i(0,-1,0), Vector3i(-1,0,0), Vector3i(1,0,0), Vector3i(0,1,0), Vector3i(0,0,1)]:
				var q: Vector3i = pos + displacement
				if q.x < 0 or q.y < 0 or q.z < 0 or q.x >= I or q.y >= J or q.z >= K:
					continue
				var q_spatial: int = q.x + I * q.y + IJ * q.z
				var other: Dictionary = neuron_by_spatial_index[q_spatial]
				var source: Dictionary = other if phase == 0 else neuron
				var target: Dictionary = neuron if phase == 0 else other
				if phase == 0:
					if source["output"] or (target["output"] and source["input"]):
						continue
				else:
					if target["input"] or target["output"]:
						continue
				structural_count += 1
				if phase == 0:
					receiver_candidates += 1
				else:
					input_candidates += 1
				var h: float = math.H(source["connections_value"], target["connections_value"], coefficients)
				var h2: float = h * h
				if h2 > 0.0:
					candidates.append({"source": source, "target": target, "score": h2})
			candidates.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return a["score"] > b["score"])
			var successful := false
			for item in candidates:
				var source: Dictionary = item["source"]
				if source["connections"].size() >= max_connections:
					continue
				var target: Dictionary = item["target"]
				source["connections"].append({"target": target["position"], "weight": item["score"], "eligibility_trace": 0.0})
				var tp: Vector3i = target["position"]
				incoming_connections[tp.x + I * tp.y + IJ * tp.z] += 1
				added += 1
				successful = true
				break
			if not successful:
				valid = false
				if structural_count == 0:
					reasons["no_structural_candidate"] += 1
				elif candidates.is_empty():
					reasons["no_positive_h_candidate"] += 1
				elif phase == 0:
					reasons["positive_h_candidates_saturated"] += 1
				else:
					reasons["input_source_saturated"] += 1
	brain["rescue_failure_causes"] = reasons
	brain["rescue_candidates_receiver"] = receiver_candidates
	brain["rescue_candidates_input"] = input_candidates
	brain["rescue_candidates_total"] = receiver_candidates + input_candidates
	brain["rescue_edge_count"] = added
	_report_progress(progress_callback, 99)
	return valid


## Computes the three global modulatory dynamics parameters from the 8 digit gene.
## The returned components are lambda, nu and mu, in this order.
func _modulatory_dynamics(gene: Array) -> Vector3:
	return Vector3(
		1.0 / (1.0 + exp(-math.G(true, false, gene))),
		1.0 / (1.0 + exp(-math.G(false, true, gene))),
		1.0 / (1.0 + exp(-math.G(true, true, gene)))
	)


## Sends a percentage update when a progress callback was provided.
func _report_progress(progress_callback: Callable, percent: int) -> void:
	if progress_callback.is_valid():
		progress_callback.call(clampi(percent, 0, 100))


func _initial_offset_table(gene: Array, I: int, J: int, K: int, radius_i: int, radius_j: int, radius_k: int, limit: int) -> PackedInt32Array:
	var offsets := PackedInt32Array()
	var capacity_i: int = I - radius_i - 1
	var capacity_j: int = J - radius_j - 1
	var capacity_k: int = K - radius_k - 1
	if capacity_i <= 0 and capacity_j <= 0 and capacity_k <= 0:
		return PackedInt32Array([0, 0, 0, 0]) # A nonempty SSBO is required.
	var tuple_limit: int = mini(5985, ceili(float(limit) / 8.0))
	offsets = _append_initial_orbit(offsets, gene, 0, 5, 10, 15, I, J, K, radius_i, radius_j, radius_k, capacity_i, capacity_j, capacity_k, tuple_limit)
	if offsets.size() / 4 >= tuple_limit:
		return offsets
	# For every orbit, exactly four rotated members start at position 0.
	# Among those, the lexicographically smallest one is its representative.
	# Their encoded indices are compared directly to avoid generating all
	# 5985 quadruples and avoid maintaining a 'visited' set.
	for b in range(1, 19):
		for c in range(b + 1, 20):
			for d in range(c + 1, 21):
				if b == 5 and c == 10 and d == 15:
					continue # The preferred first orbit was already emitted.
				var key: int = b * 441 + c * 21 + d
				if (key > (c - b) * 441 + (d - b) * 21 + (21 - b)
					or key > (d - c) * 441 + (21 - c) * 21 + (21 + b - c)
					or key > (21 - d) * 441 + (21 + b - d) * 21 + (21 + c - d)):
					continue
				offsets = _append_initial_orbit(offsets, gene, 0, b, c, d, I, J, K, radius_i, radius_j, radius_k, capacity_i, capacity_j, capacity_k, tuple_limit)
				if offsets.size() / 4 >= tuple_limit:
					return offsets
	return offsets


func _append_initial_orbit(offsets: PackedInt32Array, gene: Array, q0: int, q1: int, q2: int, q3: int, I: int, J: int, K: int, radius_i: int, radius_j: int, radius_k: int, capacity_i: int, capacity_j: int, capacity_k: int, tuple_limit: int) -> PackedInt32Array:
	for rotation in 21:
		var indices: Array[int] = [(q0 + rotation) % 21, (q1 + rotation) % 21, (q2 + rotation) % 21, (q3 + rotation) % 21]
		indices.sort()
		var digit_i: int = 0
		var digit_j: int = 0
		var digit_k: int = 0
		for idx in indices:
			digit_i = digit_i * 10 + int(gene[idx])
			digit_j = digit_j * 10 + int(gene[21 + idx])
			digit_k = digit_k * 10 + int(gene[42 + idx])
		var a: int = digit_i % I
		var b: int = digit_j % J
		var c: int = digit_k % K
		# Cycle the forced-outside axis i -> j -> k, with cyclic fallback
		# whenever that axis has no room beyond its local search radius.
		var tuple_index: int = offsets.size() / 4
		var axis: int = -1
		for shift in 3:
			var candidate_axis: int = (tuple_index + shift) % 3
			if ((candidate_axis == 0 and capacity_i > 0)
				or (candidate_axis == 1 and capacity_j > 0)
				or (candidate_axis == 2 and capacity_k > 0)):
				axis = candidate_axis
				break
		if axis == 0:
			a = radius_i + 1 + a % capacity_i
		elif axis == 1:
			b = radius_j + 1 + b % capacity_j
		else:
			c = radius_k + 1 + c % capacity_k
		offsets.append(a)
		offsets.append(b)
		offsets.append(c)
		offsets.append(0) # ivec4 for std430 alignment.
		if offsets.size() / 4 >= tuple_limit:
			return offsets
	return offsets


func _finalize_connection_weights(brain: Dictionary) -> void:
	var floor_input: float = 1.0 / (1.0 + float(brain["input_gain"]))
	for neuron in brain["neurons"]:
		var A: float = neuron["connections_value"]
		var attenuation: float = 1.0 - A * A
		for edge in neuron["connections"]:
			var weight: float = maxf(float(edge["weight"]) * attenuation, 0.1)
			if neuron["input"]:
				weight = maxf(weight, floor_input)
			edge["weight"] = weight

