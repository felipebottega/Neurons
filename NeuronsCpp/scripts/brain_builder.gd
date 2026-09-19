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
func build(genome: Genome, size: Vector3i, num_inputs: int, num_outputs: int, progress_callback: Callable = Callable()) -> Dictionary:
	if size.x < 2 or size.y < 2 or size.z < 2:
		push_error("Brain dimensions I, J, K must all be at least 2.")
		return {}

	_report_progress(progress_callback, 0)

	var beta := math.Beta(genome.beta, size.x, size.y)
	var modulatory_dynamics := _modulatory_dynamics(genome.modulatory_dynamics)
	var brain = {
		"size": size,
		"beta": beta,
		"max_connections": floori(math.F(beta, genome.max_connections)),
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
	_set_input_output_neurons(brain["neurons"], size, num_inputs, num_outputs, progress_callback)
	end = Time.get_ticks_msec()
	print('B: ', (end - start)/1000.0, ' seconds')
	
	start = Time.get_ticks_msec()
	_create_connections(brain, genome.connections, progress_callback)
	end = Time.get_ticks_msec()
	print('C: ', (end - start)/1000.0, ' seconds')
	
	start = Time.get_ticks_msec()
	var valid_brain = _clear_invalid_connections(brain, genome.connections, genome.input_influence, progress_callback)
	end = Time.get_ticks_msec()
	print('D: ', (end - start)/1000.0, ' seconds')
	
	brain["valid"] = valid_brain
	
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
func _create_connections(brain: Dictionary, gene: Array, progress_callback: Callable) -> void:
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

	var group_task_id = WorkerThreadPool.add_group_task(
		_create_connections_for_neuron.bind(
			neurons,
			size,
			brain["max_connections"],
			h_coefficients,
			connection_values,
			connection_results
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
			_report_progress(progress_callback, 30 + int(65.0 * completed / total_neurons))
		OS.delay_msec(10)

	WorkerThreadPool.wait_for_group_task_completion(group_task_id)

	# Workers only write to the fixed size result array. Apply the results after every
	# worker has finished so the neuron dictionaries themselves are never mutated concurrently.
	for neuron_index in total_neurons:
		neurons[neuron_index]["connections"] = connection_results[neuron_index]

	_report_progress(progress_callback, 95)

## Calculates the outgoing connections of one source neuron.
## This method is called concurrently by WorkerThreadPool.
func _create_connections_for_neuron(neuron_index: int, neurons: Array, size: Vector3i, max_connections: int, h_coefficients: Vector4, connection_values: PackedFloat64Array, connection_results: Array) -> void:
	if max_connections <= 0 or neurons[neuron_index]["output"]:
		connection_results[neuron_index] = []
		return

	var pos: Vector3i = neurons[neuron_index]["position"]
	var I: int = size.x
	var J: int = size.y
	var K: int = size.z
	var IJ: int = I * J
	var source_linear_index: int = pos.x + I * pos.y + IJ * pos.z
	var A_ijk: float = connection_values[source_linear_index]
	var target_k_min: int = maxi(0, pos.z - 2)
	var target_k_max: int = mini(K - 1, pos.z + 2)
	var candidates: Array = []

	for u in I:
		var u_equals_posx: bool = u == pos.x

		for v in J:
			var Iv: int = I * v
			var v_equals_posy: bool = v == pos.y

			for w in range(target_k_min, target_k_max + 1):
				if u_equals_posx and v_equals_posy and w == pos.z:
					continue

				var target_linear_index: int = u + Iv + IJ * w
				var A_uvw: float = connection_values[target_linear_index]
				var h := math.H(A_ijk, A_uvw, h_coefficients)
				
				if h > 0.0:
					candidates.append({"target": Vector3i(u, v, w), "value": h})

	candidates.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return a["value"] > b["value"])

	var connection_count: int = mini(max_connections, candidates.size())
	var connections: Array = []
	connections.resize(connection_count)

	for index in connection_count:
		connections[index] = {
			"target": candidates[index]["target"],
			"weight": candidates[index]["value"],
			"eligibility_trace": 0.0,
		}

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
	brain["input_gain"] = (1.0 + sin(math.G(true, true, input_influence_gene))) * mean_internal_incoming
	var input_min_weight = 1.0 / (1.0 + float(brain["input_gain"]))

	for neuron in neurons:
		if not neuron["input"]:
			continue

		for connection in neuron["connections"]:
			connection["weight"] = maxf(float(connection["weight"]), input_min_weight)

	return rescue_success and isolated_internal == 0 and isolated_output == 0 and isolated_input == 0


## Creates the minimum connectivity required to rescue isolated neurons when possible.
## Internal and output neurons without incoming connections receive exactly one incoming connection.
## Input neurons without outgoing connections receive exactly one outgoing connection.
## Existing connections are never removed or redirected. Every new connection must satisfy H > 0,
## obey the normal input/output restrictions, and respect a temporary rescue limit of
## ceil(1.01 * N_max) + 1 outgoing connections per source neuron.
## Returns false when any isolated neuron cannot be rescued under these rules.
func _rescue_isolated_neurons(brain: Dictionary, gene: Array, neuron_by_spatial_index: Array, incoming_connections: PackedInt32Array, progress_callback: Callable) -> bool:
	var neurons: Array = brain["neurons"]
	var size: Vector3i = brain["size"]
	var max_connections: int = int(ceil(1.01 * brain["max_connections"]) + 1)
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
		var source_k_min: int = maxi(0, target_position.z - 2)
		var source_k_max: int = mini(K - 1, target_position.z + 2)
		var best_source: Dictionary = {}
		var best_h: float = 0.0
		var structural_candidates = 0
		var positive_h_candidates = 0
		var free_positive_h_candidates = 0

		for source_k in range(source_k_min, source_k_max + 1):
			var slice_offset: int = IJ * source_k

			for source_j in J:
				var row_offset: int = slice_offset + I * source_j

				for source_i in I:
					var source_spatial_index: int = row_offset + source_i

					if source_spatial_index == target_spatial_index:
						continue

					var source: Dictionary = neuron_by_spatial_index[source_spatial_index]

					# Output neurons never emit signals.
					if source["output"]:
						continue

					# Output neurons may receive incoming connections only from internal neurons.
					if target["output"] and source["input"]:
						continue

					structural_candidates += 1
					var h: float = math.H(float(source["connections_value"]), target_connection_value, h_coefficients)

					if h <= 0.0:
						continue

					positive_h_candidates += 1

					if source["connections"].size() >= max_connections:
						continue

					free_positive_h_candidates += 1

					if h > best_h:
						best_h = h
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

		_report_progress(progress_callback, 98 + int(float(isolated_index + 1) / isolated_receiver_count))

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
		var target_k_min: int = maxi(0, source_position.z - 2)
		var target_k_max: int = mini(K - 1, source_position.z + 2)
		var best_target: Dictionary = {}
		var best_target_spatial_index: int = -1
		var best_input_h: float = 0.0
		var structural_targets = 0
		var positive_h_targets = 0

		for target_k in range(target_k_min, target_k_max + 1):
			var slice_offset: int = IJ * target_k

			for target_j in J:
				var row_offset: int = slice_offset + I * target_j

				for target_i in I:
					var target_spatial_index: int = row_offset + target_i
					var target: Dictionary = neuron_by_spatial_index[target_spatial_index]

					# Inputs can connect only to internal neurons.
					if target["input"] or target["output"]:
						continue

					structural_targets += 1
					var h: float = math.H(source_connection_value, float(target["connections_value"]), h_coefficients)

					if h <= 0.0:
						continue

					positive_h_targets += 1

					if h > best_input_h:
						best_input_h = h
						best_target = target
						best_target_spatial_index = target_spatial_index

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

		_report_progress(progress_callback, 99 + int(float(isolated_index + 1) / isolated_input_count))

	brain["rescue_failure_causes"] = {
		"no_structural_candidate": no_structural_candidate,
		"no_positive_h_candidate": no_positive_h_candidate,
		"positive_h_candidates_saturated": positive_h_candidates_saturated,
		"input_source_saturated": input_source_saturated
	}

	_report_progress(progress_callback, 100)
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
