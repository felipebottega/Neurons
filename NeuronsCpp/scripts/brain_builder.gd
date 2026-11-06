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
	if size.x < 2 or size.y < 2 or size.z < 2 or num_inputs < 0 or num_outputs < 0 or num_inputs + num_outputs > size.x * size.y * size.z:
		push_error("Invalid brain dimensions or input/output counts.")
		return {}
	if is_nan(connections_density) or is_inf(connections_density) or connections_density <= 0.0:
		push_error("Connections density eta must be positive and finite.")
		return {}

	_report_progress(progress_callback, 0)

	var beta := math.Beta(genome.beta, size.x, size.y)
	var modulatory_dynamics := _modulatory_dynamics(genome.modulatory_dynamics)
	var brain = {
		"size": size,
		"beta": beta,
		"max_connections": maxi(1, floori(connections_density * math.F(beta, genome.max_connections))),
		"connections_density": connections_density,
		"refractory_strength": math.F(0.4, genome.refractory_strength),
		"exponential_factor": math.F(0.4, genome.exponential_factor),
		"structural_plasticity": math.P(genome.structural_plasticity),
		"lambda": modulatory_dynamics.x,
		"nu": modulatory_dynamics.y,
		"mu": modulatory_dynamics.z,
		"neurons": []
	}

	var start = Time.get_ticks_msec()
	brain["neurons"] = _create_neurons(size, genome, progress_callback)
	var end = Time.get_ticks_msec()
	print('A: ', (end - start)/1000.0, ' seconds')
	
	start = Time.get_ticks_msec()
	_create_connections(brain, genome.connections, genome.initial_search_region, progress_callback)
	end = Time.get_ticks_msec()
	print('B: ', (end - start)/1000.0, ' seconds')
	
	start = Time.get_ticks_msec()
	brain["role_selection_gap"] = 0 if num_inputs + num_outputs == 0 else _prime_below(floori(float(size.x * size.y * size.z) / float(num_inputs + num_outputs)))
	brain["role_selection_valid"] = _set_input_output_neurons(brain["neurons"], size, num_inputs, num_outputs, progress_callback)
	end = Time.get_ticks_msec()
	print('C: ', (end - start)/1000.0, ' seconds')
	
	start = Time.get_ticks_msec()
	var valid_brain = _clear_invalid_connections(brain, genome.connections, genome.input_influence, progress_callback)
	end = Time.get_ticks_msec()
	print('D: ', (end - start)/1000.0, ' seconds')
	
	brain["valid"] = valid_brain and brain["role_selection_valid"]
	
	start = Time.get_ticks_msec()
	_report_progress(progress_callback, 100)
	end = Time.get_ticks_msec()
	print('E: ', (end - start)/1000.0, ' seconds')
	print('==============================')
	
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

## Assigns interface roles after the all-internal provisional graph exists.
## Selection uses prime-gap linear positions and provisional degree >= 2.
## Outputs retain channel order, while internal neurons are spatially ordered.
static func _prime_below(value: int) -> int:
	for candidate in range(value - 1, 1, -1):
		var prime := true
		var divisor := 2
		while divisor * divisor <= candidate:
			if candidate % divisor == 0:
				prime = false
				break
			divisor += 1
		if prime:
			return candidate
	return 0

func _set_input_output_neurons(neurons: Array, size: Vector3i, num_inputs: int, num_outputs: int, progress_callback: Callable) -> bool:
	var total: int = neurons.size()
	var channels: int = num_inputs + num_outputs
	var gap: int = 0 if channels == 0 else _prime_below(floori(float(total) / float(channels)))
	if channels > 0 and gap == 0:
		_report_progress(progress_callback, 75)
		return false

	var by_spatial: Array = []
	by_spatial.resize(total)
	var incoming := PackedInt32Array()
	incoming.resize(total)
	for neuron in neurons:
		var pos: Vector3i = neuron["position"]
		var spatial: int = pos.x + size.x * (pos.y + size.y * pos.z)
		by_spatial[spatial] = neuron
		for connection in neuron["connections"]:
			var target: Vector3i = connection["target"]
			incoming[target.x + size.x * (target.y + size.y * target.z)] += 1
	var occupied := PackedByteArray()
	occupied.resize(total)
	var inputs: Array[int] = []
	var outputs: Array[int] = []
	var valid := true
	for channel in channels:
		var is_input: bool = channel < num_inputs
		var preferred: int = ((channel + 1) * gap - 1) % total
		var chosen := -1
		for advance in total:
			var candidate: int = (preferred + advance) % total
			if occupied[candidate] != 0:
				continue
			var degree: int = by_spatial[candidate]["connections"].size() if is_input else incoming[candidate]
			if degree < 2:
				continue
			chosen = candidate
			break
		if chosen < 0:
			valid = false
			break
		occupied[chosen] = 1
		if is_input:
			inputs.append(chosen)
		else:
			outputs.append(chosen)
	var ordered: Array = []
	for spatial in inputs:
		var neuron: Dictionary = by_spatial[spatial]
		neuron["input"] = true
		neuron["activation_threshold"] = 0.0
		neuron["retention_factor"] = 0.0
		neuron["polarity_factor"] = 1
		ordered.append(neuron)
	for spatial in total:
		if occupied[spatial] == 0:
			ordered.append(by_spatial[spatial])
	for spatial in outputs:
		var neuron: Dictionary = by_spatial[spatial]
		neuron["output"] = true
		ordered.append(neuron)
	neurons.assign(ordered)
	_report_progress(progress_callback, 75)
	return valid

## Creates the initial connections and their weights.
## Connection generation is distributed across the WorkerThreadPool, one source neuron per task element.
## [param brain] contains the brain structure.
## [param gene] is used to create the connections.
## Returns nothing.
func _create_connections(brain: Dictionary, gene: Array, search_gene: Array, progress_callback: Callable) -> void:
	var size: Vector3i = brain["size"]
	var neurons: Array = brain["neurons"]
	var total_neurons: int = neurons.size()
	var I: int = size.x
	var IJ: int = I * size.y

	# A for the connection gene was already calculated while creating the neurons.
	# Store it by spatial linear index because the neuron array is reordered when inputs and outputs are selected.
	var connection_values := PackedFloat64Array()
	connection_values.resize(total_neurons)

	for neuron in neurons:
		var pos: Vector3i = neuron["position"]
		var linear_index: int = pos.x + I * pos.y + IJ * pos.z
		connection_values[linear_index] = neuron["connections_value"]

	var connection_results: Array = []
	connection_results.resize(total_neurons)

	# These coefficients depend only on the connection gene, so calculate them once.
	var h_coefficients: Vector4 = math.H_coefficients(gene)
	var search_extent: float = math.S(search_gene)
	var radius_i: int = mini(size.x - 1, ceili(search_extent * size.x))
	var radius_j: int = mini(size.y - 1, ceili(search_extent * size.y))
	var radius_k: int = mini(size.z - 1, ceili(search_extent * size.z))
	brain["initial_search_extent"] = search_extent
	brain["initial_search_radius_i"] = radius_i
	brain["initial_search_radius_j"] = radius_j
	brain["initial_search_radius_k"] = radius_k
	var supplementary_offsets: Array[Vector3i] = _supplementary_offsets(search_gene, size.x, size.y, size.z, radius_i, radius_j, radius_k)

	var group_task_id = WorkerThreadPool.add_group_task(
		_create_connections_for_neuron.bind(
			neurons,
			size,
			brain["max_connections"],
			h_coefficients,
			connection_values,
			connection_results,
			radius_i,
			radius_j,
			radius_k,
			supplementary_offsets
		),
		total_neurons
	)

	# This function itself already runs outside the main thread. Polling here keeps the
	# BrainExplorer progress callback updated while the worker pool uses the CPU cores.
	var last_completed = -1

	while not WorkerThreadPool.is_group_task_completed(group_task_id):
		var completed = WorkerThreadPool.get_group_processed_element_count(group_task_id)
		if completed != last_completed:
			last_completed = completed
			_report_progress(progress_callback, 20 + int(55.0 * completed / total_neurons))
		OS.delay_msec(10)

	WorkerThreadPool.wait_for_group_task_completion(group_task_id)

	# Workers only write to the fixed size result array. Apply the results after every
	# worker has finished so the neuron dictionaries themselves are never mutated concurrently.
	for neuron_index in total_neurons:
		neurons[neuron_index]["connections"] = connection_results[neuron_index]

	_report_progress(progress_callback, 75)

## Calculates the outgoing connections of one source neuron.
## This method is called concurrently by WorkerThreadPool.
# A fixed cyclic ordering of all 5985 four-position combinations.
func _rotate_quad(q: Array[int], rotation: int) -> Array[int]:
	var result: Array[int] = []
	for index in q:
		result.append((index + rotation) % 21)
	result.sort()
	return result

func _quad_less(a: Array[int], b: Array[int]) -> bool:
	for t in 4:
		if a[t] != b[t]:
			return a[t] < b[t]
	return false

func _four_decimal(gene: Array, start: int, q: Array[int]) -> int:
	var number: int = 0
	for index in q:
		number = number * 10 + int(gene[start + index])
	return number

func _supplementary_offsets(gene: Array, I: int, J: int, K: int, radius_i: int, radius_j: int, radius_k: int) -> Array[Vector3i]:
	var offsets: Array[Vector3i] = []
	var Di: int = I - radius_i - 1
	var Dj: int = J - radius_j - 1
	var Dk: int = K - radius_k - 1
	if Di <= 0 and Dj <= 0 and Dk <= 0:
		return offsets

	var first: Array[int] = [0, 5, 10, 15]
	var first_canonical: Array[int] = first
	for h in range(1, 21):
		var rotated := _rotate_quad(first, h)
		if _quad_less(rotated, first_canonical):
			first_canonical = rotated
	var quadruples: Array = []
	for h in 21:
		quadruples.append(_rotate_quad(first, h))
	for a in range(18):
		for b in range(a + 1, 19):
			for c in range(b + 1, 20):
				for d in range(c + 1, 21):
					var q: Array[int] = [a, b, c, d]
					if q == first_canonical:
						continue
					var canonical: bool = true
					for h in range(1, 21):
						if _quad_less(_rotate_quad(q, h), q):
							canonical = false
							break
					if canonical:
						for h in 21:
							quadruples.append(_rotate_quad(q, h))
	assert(quadruples.size() == 5985)
	for t in quadruples.size():
		var q: Array[int] = quadruples[t]
		var a: int = _four_decimal(gene, 0, q) % I
		var b: int = _four_decimal(gene, 21, q) % J
		var c: int = _four_decimal(gene, 42, q) % K
		for shift in 3:
			var axis: int = (t + shift) % 3
			if axis == 0 and Di > 0:
				offsets.append(Vector3i(radius_i + 1 + a % Di, b, c))
				break
			if axis == 1 and Dj > 0:
				offsets.append(Vector3i(a, radius_j + 1 + b % Dj, c))
				break
			if axis == 2 and Dk > 0:
				offsets.append(Vector3i(a, b, radius_k + 1 + c % Dk))
				break
	return offsets

## Two initial passes: strongest positive H inside the genetic neighborhood,
## then at most N_max individually tested signed supplementary candidates.
func _create_connections_for_neuron(neuron_index: int, neurons: Array, size: Vector3i, max_connections: int, h_coefficients: Vector4, connection_values: PackedFloat64Array, connection_results: Array, radius_i: int, radius_j: int, radius_k: int, supplementary_offsets: Array[Vector3i]) -> void:
	if max_connections <= 0 or neurons[neuron_index]["output"]:
		connection_results[neuron_index] = []
		return
	var pos: Vector3i = neurons[neuron_index]["position"]
	var I: int = size.x
	var J: int = size.y
	var K: int = size.z
	var IJ: int = I * J
	var A_ijk: float = connection_values[pos.x + I * pos.y + IJ * pos.z]
	var candidates: Array = []
	for u in range(maxi(0, pos.x - radius_i), mini(I - 1, pos.x + radius_i) + 1):
		for v in range(maxi(0, pos.y - radius_j), mini(J - 1, pos.y + radius_j) + 1):
			for w in range(maxi(0, pos.z - radius_k), mini(K - 1, pos.z + radius_k) + 1):
				if u == pos.x and v == pos.y and w == pos.z:
					continue
				var target_index: int = u + I * v + IJ * w
				var h: float = math.H(A_ijk, connection_values[target_index], h_coefficients)
				if h > 0.0:
					candidates.append({"target": Vector3i(u, v, w), "value": h})
	candidates.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return a["value"] > b["value"] or (a["value"] == b["value"] and (a["target"].x + I * a["target"].y + IJ * a["target"].z) < (b["target"].x + I * b["target"].y + IJ * b["target"].z)))
	var count: int = mini(max_connections, candidates.size())
	var connections: Array = []
	var seen: Dictionary = {}
	for idx in count:
		var candidate: Dictionary = candidates[idx]
		var target: Vector3i = candidate["target"]
		connections.append({"target": target, "weight": candidate["value"], "eligibility_trace": 0.0, "_construction_pass": 0})
		seen[target.x + I * target.y + IJ * target.z] = true
	if count < max_connections:
		var attempts: int = 0
		for delta in supplementary_offsets:
			for sign in 8:
				if attempts >= max_connections or connections.size() >= max_connections:
					connection_results[neuron_index] = connections
					return
				attempts += 1
				var u: int = pos.x + (-delta.x if (sign & 4) != 0 else delta.x)
				var v: int = pos.y + (-delta.y if (sign & 2) != 0 else delta.y)
				var w: int = pos.z + (-delta.z if (sign & 1) != 0 else delta.z)
				if u < 0 or u >= I or v < 0 or v >= J or w < 0 or w >= K:
					continue
				if u == pos.x and v == pos.y and w == pos.z:
					continue
				var target_index: int = u + I * v + IJ * w
				if seen.has(target_index):
					continue
				seen[target_index] = true
				var h: float = math.H(A_ijk, connection_values[target_index], h_coefficients)
				if h > 0.0:
					connections.append({"target": Vector3i(u, v, w), "weight": h, "eligibility_trace": 0.0, "_construction_pass": 1})
	connection_results[neuron_index] = connections

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
	brain["initial_connections_first_pass"] = 0
	brain["initial_connections_second_pass"] = 0

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

			if connection.get("_construction_pass", 0) == 0:
				brain["initial_connections_first_pass"] += 1
			elif connection.get("_construction_pass", 0) == 1:
				brain["initial_connections_second_pass"] += 1
			connection.erase("_construction_pass")
			valid_connections.append(connection)
			incoming_connections[target_spatial_index] += 1

		neuron["connections"] = valid_connections
		completed += 1

		if completed % report_interval == 0 or completed == total_neurons:
			_report_progress(progress_callback, 75 + int(4.0 * completed / total_neurons))

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

	brain["initial_local_connections"] = brain["initial_connections_first_pass"]
	brain["initial_additional_connections"] = brain["initial_connections_second_pass"]
	brain["role_cleanup_valid"] = isolated_input_before == 0 and isolated_output_before == 0
	brain["isolated_before_rescue"] = {
		"internal": isolated_internal_before,
		"output": isolated_output_before,
		"input": isolated_input_before
	}

	if isolated_internal_before == 0 and isolated_output_before == 0 and isolated_input_before == 0:
		print("No isolated neurons before developmental rescue.")
	else:
		print(
			"Isolated neurons before developmental rescue: ",
			isolated_internal_before, " internal without incoming connections, ",
			isolated_output_before, " output without incoming connections, ",
			isolated_input_before, " input without outgoing connections."
		)

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

	if isolated_internal == 0 and isolated_output == 0 and isolated_input == 0:
		print("No isolated neurons after developmental rescue.")
	else:
		print(
			"Isolated neurons after developmental rescue: ",
			isolated_internal, " internal without incoming connections, ",
			isolated_output, " output without incoming connections, ",
			isolated_input, " input without outgoing connections."
		)

	var rescue_failure_causes: Dictionary = brain.get("rescue_failure_causes", {})
	print(
		"Rescue failure causes: ",
		rescue_failure_causes.get("no_structural_candidate", 0), " no structurally valid candidate, ",
		rescue_failure_causes.get("no_positive_h_candidate", 0), " no candidate with H > 0, ",
		rescue_failure_causes.get("positive_h_candidates_saturated", 0), " H > 0 candidates saturated, ",
		rescue_failure_causes.get("input_source_saturated", 0), " input source at N_max."
	)

	# psi is fixed at generation time and uses the initial mean incoming degree of internal neurons.
	var mean_internal_incoming = 0.0 if internal_neuron_count == 0 else float(internal_incoming_sum) / internal_neuron_count
	brain["mean_initial_incoming"] = mean_internal_incoming
	brain["input_gain"] = (1.0 + sin(math.G(true, true, input_influence_gene))) * mean_internal_incoming
	var input_min_weight = 1.0 / (1.0 + float(brain["input_gain"]))

	# Apply the source-dependent attenuation after top-H selection, invalid-connection
	# filtering and developmental rescue. Rescued connections are included.
	for neuron in neurons:
		var source_a: float = float(neuron["connections_value"])
		var attenuation: float = 1.0 - source_a * source_a

		for connection in neuron["connections"]:
			var initial_weight: float = maxf(0.1, float(connection["weight"]) * attenuation)
			if neuron["input"]:
				# Keep the existing minimum initial weight for input synapses.
				initial_weight = maxf(initial_weight, input_min_weight)
			connection["weight"] = initial_weight

	return brain["role_cleanup_valid"] and rescue_success and isolated_internal == 0 and isolated_output == 0 and isolated_input == 0


## Creates the minimum connectivity required to rescue isolated neurons when possible.
## Internal and output neurons without incoming connections receive exactly one incoming connection.
## Input neurons without outgoing connections receive exactly one outgoing connection.
## Existing connections are never removed or redirected. Rescued candidates maximize H squared,
## obey input/output restrictions and the permanent N_max outgoing capacity.
## Returns false when any isolated neuron cannot be rescued under these rules.
func _rescue_isolated_neurons(brain: Dictionary, gene: Array, neuron_by_spatial_index: Array, incoming_connections: PackedInt32Array, progress_callback: Callable) -> bool:
	var neurons: Array = brain["neurons"]
	var size: Vector3i = brain["size"]
	var max_connections: int = int(brain["max_connections"])
	var I: int = size.x
	var J: int = size.y
	var K: int = size.z
	var IJ: int = I * J
	var h_coefficients: Vector4 = math.H_coefficients(gene)
	var isolated_receivers: Array = []
	var rescue_success = true
	var no_structural_candidate = 0
	var no_positive_h_candidate = 0
	var positive_h_candidates_saturated = 0
	var input_source_saturated = 0
	var rescue_candidates_receiver: int = 0
	var rescue_candidates_input: int = 0

	# Internal and output neurons require at least one incoming connection.
	for neuron in neurons:
		if neuron["input"]:
			continue

		var position: Vector3i = neuron["position"]
		var spatial_index: int = position.x + I * position.y + IJ * position.z

		if incoming_connections[spatial_index] == 0:
			isolated_receivers.append(neuron)

	var isolated_receiver_count: int = isolated_receivers.size()

	for isolated_index in isolated_receiver_count:
		var target: Dictionary = isolated_receivers[isolated_index]
		var target_position: Vector3i = target["position"]
		var target_spatial_index: int = target_position.x + I * target_position.y + IJ * target_position.z
		var target_connection_value: float = target["connections_value"]
		var best_source: Dictionary = {}
		var best_h: float = 0.0
		var structural_candidates: int = 0
		var positive_h_candidates: int = 0
		var free_positive_h_candidates: int = 0
		for delta in [Vector3i(0, 0, -1), Vector3i(0, -1, 0), Vector3i(-1, 0, 0), Vector3i(1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, 0, 1)]:
			var neighbor: Vector3i = target_position + delta
			if neighbor.x < 0 or neighbor.x >= I or neighbor.y < 0 or neighbor.y >= J or neighbor.z < 0 or neighbor.z >= K:
				continue
			var source: Dictionary = neuron_by_spatial_index[neighbor.x + I * neighbor.y + IJ * neighbor.z]
			if source["output"] or (target["output"] and source["input"]):
				continue
			structural_candidates += 1
			var h: float = math.H(float(source["connections_value"]), target_connection_value, h_coefficients)
			rescue_candidates_receiver += 1
			var h_squared: float = h * h
			if h_squared <= 0.0:
				continue
			positive_h_candidates += 1
			if source["connections"].size() >= max_connections:
				continue
			free_positive_h_candidates += 1
			if h_squared > best_h:
				best_h = h_squared
				best_source = source

		if best_source.is_empty():
			rescue_success = false

			if structural_candidates == 0:
				no_structural_candidate += 1
			elif positive_h_candidates == 0:
				no_positive_h_candidate += 1
			elif free_positive_h_candidates == 0:
				positive_h_candidates_saturated += 1

			continue

		best_source["connections"].append({
			"target": target_position,
			"weight": best_h,
			"eligibility_trace": 0.0,
		})
		incoming_connections[target_spatial_index] = 1

		_report_progress(progress_callback, 79 + int(2.0 * float(isolated_index + 1) / isolated_receiver_count))

	# Some initially isolated inputs may already have been used above to rescue an internal neuron.
	# Any input that is still isolated receives exactly one outgoing connection to an internal neuron.
	var isolated_inputs: Array = []

	for neuron in neurons:
		if neuron["input"] and neuron["connections"].is_empty():
			isolated_inputs.append(neuron)

	var isolated_input_count: int = isolated_inputs.size()

	for isolated_index in isolated_input_count:
		var source: Dictionary = isolated_inputs[isolated_index]

		if source["connections"].size() >= max_connections:
			rescue_success = false
			input_source_saturated += 1
			continue

		var source_position: Vector3i = source["position"]
		var source_connection_value: float = source["connections_value"]
		var best_target: Dictionary = {}
		var best_target_spatial_index: int = -1
		var best_input_h: float = 0.0
		var structural_targets: int = 0
		var positive_h_targets: int = 0
		for delta in [Vector3i(0, 0, -1), Vector3i(0, -1, 0), Vector3i(-1, 0, 0), Vector3i(1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, 0, 1)]:
			var neighbor: Vector3i = source_position + delta
			if neighbor.x < 0 or neighbor.x >= I or neighbor.y < 0 or neighbor.y >= J or neighbor.z < 0 or neighbor.z >= K:
				continue
			var target_index: int = neighbor.x + I * neighbor.y + IJ * neighbor.z
			var target: Dictionary = neuron_by_spatial_index[target_index]
			if target["input"] or target["output"]:
				continue
			structural_targets += 1
			var h: float = math.H(source_connection_value, float(target["connections_value"]), h_coefficients)
			rescue_candidates_input += 1
			var h_squared: float = h * h
			if h_squared <= 0.0:
				continue
			positive_h_targets += 1
			if h_squared > best_input_h:
				best_input_h = h_squared
				best_target = target
				best_target_spatial_index = target_index

		if best_target.is_empty():
			rescue_success = false

			if structural_targets == 0:
				no_structural_candidate += 1
			elif positive_h_targets == 0:
				no_positive_h_candidate += 1

			continue

		source["connections"].append({
			"target": best_target["position"],
			"weight": best_input_h,
			"eligibility_trace": 0.0,
		})
		incoming_connections[best_target_spatial_index] += 1

		_report_progress(progress_callback, 81 + int(2.0 * float(isolated_index + 1) / isolated_input_count))

	brain["rescue_candidates_receiver"] = rescue_candidates_receiver
	brain["rescue_candidates_input"] = rescue_candidates_input
	brain["rescue_candidates_total"] = rescue_candidates_receiver + rescue_candidates_input
	brain["rescue_failure_causes"] = {
		"no_structural_candidate": no_structural_candidate,
		"no_positive_h_candidate": no_positive_h_candidate,
		"positive_h_candidates_saturated": positive_h_candidates_saturated,
		"input_source_saturated": input_source_saturated
	}

	_report_progress(progress_callback, 83)
	return rescue_success

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
