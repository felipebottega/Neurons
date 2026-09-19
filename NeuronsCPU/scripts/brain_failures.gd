extends Node2D


const NMAX_BIN_WIDTH = 10


class NMaxValidityPlot:
	extends Control

	var bins: Array = []


	func set_bins(new_bins: Array) -> void:
		"""
		Input: aggregated N_max bins containing total count, valid count, and validity percentage.
		Output: stores the data and redraws the plot.
		"""
		bins = new_bins
		queue_redraw()


	func _draw() -> void:
		"""
		Input: the currently stored N_max validity bins.
		Output: draws validity probability by N_max directly inside this Control.
		"""
		var font = ThemeDB.fallback_font
		var title_font_size = 18
		var axis_font_size = 14
		var small_font_size = 12
		var left = 55.0
		var top = 45.0
		var right = 20.0
		var bottom = 65.0
		var graph_width = size.x - left - right
		var graph_height = size.y - top - bottom
		var axis_color = Color(1.0, 1.0, 1.0, 0.85)
		var grid_color = Color(1.0, 1.0, 1.0, 0.18)
		var bar_color = Color(0.95, 0.55, 0.20, 0.90)

		draw_string(
			font,
			Vector2(left, 24.0),
			"Validity probability by N_max",
			HORIZONTAL_ALIGNMENT_LEFT,
			-1.0,
			title_font_size,
			axis_color
		)

		if bins.is_empty() or graph_width <= 0.0 or graph_height <= 0.0:
			draw_string(
				font,
				Vector2(left, top + 30.0),
				"Waiting for generated brains...",
				HORIZONTAL_ALIGNMENT_LEFT,
				-1.0,
				axis_font_size,
				axis_color
			)
			return

		for percentage in [0, 25, 50, 75, 100]:
			var y = top + graph_height * (1.0 - float(percentage) / 100.0)
			draw_line(Vector2(left, y), Vector2(left + graph_width, y), grid_color, 1.0)
			draw_string(
				font,
				Vector2(10.0, y + 4.0),
				"%d%%" % percentage,
				HORIZONTAL_ALIGNMENT_LEFT,
				40.0,
				axis_font_size,
				axis_color
			)

		draw_line(Vector2(left, top), Vector2(left, top + graph_height), axis_color, 1.0)
		draw_line(
			Vector2(left, top + graph_height),
			Vector2(left + graph_width, top + graph_height),
			axis_color,
			1.0
		)

		var slot_width = graph_width / bins.size()
		var bar_width = maxf(6.0, slot_width * 0.62)

		for index in bins.size():
			var bin: Dictionary = bins[index]
			var rate = float(bin["rate"])
			var total = int(bin["total"])
			var x_center = left + slot_width * (index + 0.5)
			var bar_height = graph_height * rate / 100.0
			var bar_rect = Rect2(
				Vector2(x_center - bar_width / 2.0, top + graph_height - bar_height),
				Vector2(bar_width, bar_height)
			)

			draw_rect(bar_rect, bar_color)

			draw_string(
				font,
				Vector2(x_center - slot_width / 2.0, top + graph_height + 20.0),
				str(bin["label"]),
				HORIZONTAL_ALIGNMENT_CENTER,
				slot_width,
				small_font_size,
				axis_color
			)

			draw_string(
				font,
				Vector2(x_center - slot_width / 2.0, top + graph_height + 38.0),
				"n=%d" % total,
				HORIZONTAL_ALIGNMENT_CENTER,
				slot_width,
				small_font_size,
				axis_color
			)

			if slot_width >= 42.0:
				var rate_y = maxf(top + 14.0, top + graph_height - bar_height - 6.0)
				draw_string(
					font,
					Vector2(x_center - slot_width / 2.0, rate_y),
					"%.0f%%" % rate,
					HORIZONTAL_ALIGNMENT_CENTER,
					slot_width,
					small_font_size,
					axis_color
				)


var generation_thread: Thread
var state_mutex = Mutex.new()

var stop_requested = false
var generation_progress = 0
var current_brain_number = 0

var generated_brains = 0
var valid_brains = 0
var invalid_brains = 0

var brains_requiring_rescue = 0
var brains_rescued_successfully = 0

var isolated_before_internal = 0
var isolated_before_output = 0
var isolated_before_input = 0

var isolated_after_internal = 0
var isolated_after_output = 0
var isolated_after_input = 0

var failure_no_structural_candidate = 0
var failure_no_positive_h_candidate = 0
var failure_positive_h_candidates_saturated = 0
var failure_input_source_saturated = 0

var valid_max_connections_sum = 0.0
var invalid_max_connections_sum = 0.0
var nmax_total_counts: Dictionary = {}
var nmax_valid_counts: Dictionary = {}

var reciprocal_rate_sum = 0.0
var reciprocal_rate_squared_sum = 0.0
var reciprocal_rate_min = INF
var reciprocal_rate_max = -INF
var valid_reciprocal_rate_sum = 0.0
var invalid_reciprocal_rate_sum = 0.0
var total_directional_connections = 0
var total_reciprocal_directional_connections = 0

var nmax_plot: NMaxValidityPlot
var last_plot_generated = -1

var run_start_msec = 0
var run_end_msec = 0


func _ready() -> void:
	"""
	Input: none.
	Output: initializes the interface and connects the controls.
	"""
	$Start.pressed.connect(_on_start_pressed)
	$Stop.pressed.connect(_on_stop_pressed)

	for field in [$LineEdit, $LineEdit2, $LineEdit3, $LineEdit4, $LineEdit5]:
		field.text_changed.connect(_on_external_parameter_changed)

	$BrainInfoLabel.size = Vector2(390.0, 780.0)
	$BrainInfoLabel.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART

	nmax_plot = NMaxValidityPlot.new()
	nmax_plot.position = Vector2(410.0, 330.0)
	nmax_plot.size = Vector2(790.0, 410.0)
	nmax_plot.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(nmax_plot)

	$Stop.disabled = true
	_validate_form()
	_update_summary()


func _exit_tree() -> void:
	"""
	Input: none.
	Output: safely finishes the generation thread before this node is destroyed.
	"""
	if generation_thread == null:
		return

	state_mutex.lock()
	stop_requested = true
	state_mutex.unlock()

	if generation_thread.is_started():
		generation_thread.wait_to_finish()

	generation_thread = null


func _on_external_parameter_changed(_new_text: String) -> void:
	"""
	Input: the new text of one external parameter field.
	Output: refreshes whether generation can be started.
	"""
	_validate_form()


func _validate_form() -> void:
	"""
	Input: current external parameter fields.
	Output: enables Start only when the external parameters are valid.
	"""
	if _is_generating():
		$Start.disabled = true
		return

	var fields = [$LineEdit, $LineEdit2, $LineEdit3, $LineEdit4, $LineEdit5]

	for field in fields:
		if field.text.is_empty() or not field.text.is_valid_int():
			$Start.disabled = true
			return

	var size = Vector3i(
		int($LineEdit.text),
		int($LineEdit2.text),
		int($LineEdit3.text)
	)
	var num_inputs = int($LineEdit4.text)
	var num_outputs = int($LineEdit5.text)

	if size.x < 2 or size.y < 2 or size.z < 2:
		$Start.disabled = true
		return

	if num_inputs < 0 or num_outputs < 0:
		$Start.disabled = true
		return

	if num_inputs + num_outputs > size.x * size.y * size.z:
		$Start.disabled = true
		return

	$Start.disabled = false


func _on_start_pressed() -> void:
	"""
	Input: the external parameters currently written in the interface.
	Output: starts continuous random genome and brain generation.
	"""
	if _is_generating():
		return

	var size = Vector3i(
		int($LineEdit.text),
		int($LineEdit2.text),
		int($LineEdit3.text)
	)
	var num_inputs = int($LineEdit4.text)
	var num_outputs = int($LineEdit5.text)

	_reset_statistics()

	state_mutex.lock()
	stop_requested = false
	generation_progress = 0
	current_brain_number = 1
	state_mutex.unlock()

	run_start_msec = Time.get_ticks_msec()
	run_end_msec = 0

	_set_external_fields_editable(false)
	$Start.disabled = true
	$Stop.disabled = false

	generation_thread = Thread.new()
	var error = generation_thread.start(
		_generation_loop.bind(size, num_inputs, num_outputs)
	)

	if error != OK:
		generation_thread = null
		_set_external_fields_editable(true)
		$Stop.disabled = true
		_validate_form()
		$BrainInfoLabel.text = "Could not start generation thread.\nError: %s" % error_string(error)


func _on_stop_pressed() -> void:
	"""
	Input: none.
	Output: requests termination after the brain currently being generated finishes.
	"""
	if not _is_generating():
		return

	state_mutex.lock()
	stop_requested = true
	state_mutex.unlock()

	$Stop.disabled = true
	_update_summary()


func _generation_loop(size: Vector3i, num_inputs: int, num_outputs: int) -> void:
	"""
	Input: fixed external brain parameters.
	Output: continuously generates random genomes and records brain validity statistics until Stop is requested.
	"""
	var builder = BrainBuilder.new()

	while not _should_stop():
		var genome = Genome.new()

		state_mutex.lock()
		generation_progress = 0
		current_brain_number = generated_brains + 1
		state_mutex.unlock()

		var brain = builder.build(
			genome,
			size,
			num_inputs,
			num_outputs,
			_set_generation_progress_from_thread
		)

		genome.free()
		_record_brain(brain)

	state_mutex.lock()
	generation_progress = 100
	state_mutex.unlock()


func _set_generation_progress_from_thread(percent: int) -> void:
	"""
	Input: build progress from 0 to 100 for the current brain.
	Output: stores the latest progress for the interface thread.
	"""
	state_mutex.lock()
	generation_progress = percent
	state_mutex.unlock()


func _record_brain(brain: Dictionary) -> void:
	"""
	Input: one completed brain returned by BrainBuilder.
	Output: accumulates validity, rescue, isolation, failure cause, and connectivity statistics.
	"""
	var valid = bool(brain.get("valid", false))
	var before: Dictionary = brain.get("isolated_before_rescue", {})
	var after: Dictionary = brain.get("isolated_after_rescue", {})
	var causes: Dictionary = brain.get("rescue_failure_causes", {})

	var before_internal = int(before.get("internal", 0))
	var before_output = int(before.get("output", 0))
	var before_input = int(before.get("input", 0))

	var after_internal = int(after.get("internal", 0))
	var after_output = int(after.get("output", 0))
	var after_input = int(after.get("input", 0))

	var before_total = before_internal + before_output + before_input
	var after_total = after_internal + after_output + after_input
	var max_connections = int(brain.get("max_connections", 0))
	var reciprocity = _calculate_reciprocity(brain)
	var reciprocal_rate = float(reciprocity["rate"])
	var directional_connections = int(reciprocity["total"])
	var reciprocal_directional_connections = int(reciprocity["reciprocal"])

	state_mutex.lock()

	generated_brains += 1

	nmax_total_counts[max_connections] = int(nmax_total_counts.get(max_connections, 0)) + 1

	reciprocal_rate_sum += reciprocal_rate
	reciprocal_rate_squared_sum += reciprocal_rate * reciprocal_rate
	reciprocal_rate_min = minf(reciprocal_rate_min, reciprocal_rate)
	reciprocal_rate_max = maxf(reciprocal_rate_max, reciprocal_rate)
	total_directional_connections += directional_connections
	total_reciprocal_directional_connections += reciprocal_directional_connections

	if valid:
		valid_brains += 1
		valid_max_connections_sum += max_connections
		valid_reciprocal_rate_sum += reciprocal_rate
		nmax_valid_counts[max_connections] = int(nmax_valid_counts.get(max_connections, 0)) + 1
	else:
		invalid_brains += 1
		invalid_max_connections_sum += max_connections
		invalid_reciprocal_rate_sum += reciprocal_rate

	if before_total > 0:
		brains_requiring_rescue += 1

		if after_total == 0:
			brains_rescued_successfully += 1

	isolated_before_internal += before_internal
	isolated_before_output += before_output
	isolated_before_input += before_input

	isolated_after_internal += after_internal
	isolated_after_output += after_output
	isolated_after_input += after_input

	failure_no_structural_candidate += int(causes.get("no_structural_candidate", 0))
	failure_no_positive_h_candidate += int(causes.get("no_positive_h_candidate", 0))
	failure_positive_h_candidates_saturated += int(causes.get("positive_h_candidates_saturated", 0))
	failure_input_source_saturated += int(causes.get("input_source_saturated", 0))

	state_mutex.unlock()


func _calculate_reciprocity(brain: Dictionary) -> Dictionary:
	"""
	Input: one completed brain with directional neuron connections.
	Output: total directional connections, reciprocal directional connections, and reciprocity percentage.
	"""
	var neurons: Array = brain.get("neurons", [])
	var neuron_count = neurons.size()

	if neuron_count == 0:
		return {
			"total": 0,
			"reciprocal": 0,
			"rate": 0.0,
		}

	var neuron_index_by_position: Dictionary = {}

	for neuron_index in neuron_count:
		neuron_index_by_position[neurons[neuron_index]["position"]] = neuron_index

	var connection_keys: Dictionary = {}
	var total = 0

	for source_index in neuron_count:
		for connection in neurons[source_index]["connections"]:
			var target_index = int(neuron_index_by_position[connection["target"]])
			connection_keys[source_index * neuron_count + target_index] = true
			total += 1

	var reciprocal = 0

	for source_index in neuron_count:
		for connection in neurons[source_index]["connections"]:
			var target_index = int(neuron_index_by_position[connection["target"]])

			if connection_keys.has(target_index * neuron_count + source_index):
				reciprocal += 1

	return {
		"total": total,
		"reciprocal": reciprocal,
		"rate": 0.0 if total == 0 else 100.0 * reciprocal / total,
	}


func _should_stop() -> bool:
	"""
	Input: none.
	Output: true when Stop has been requested.
	"""
	state_mutex.lock()
	var requested = stop_requested
	state_mutex.unlock()
	return requested


func _process(_delta: float) -> void:
	"""
	Input: frame delta.
	Output: refreshes progress and accumulated statistics without accessing scene nodes from the worker thread.
	"""
	if generation_thread != null and not generation_thread.is_alive():
		generation_thread.wait_to_finish()
		generation_thread = null
		run_end_msec = Time.get_ticks_msec()
		_set_external_fields_editable(true)
		$Stop.disabled = true
		_validate_form()

	_update_summary()


func _update_summary() -> void:
	"""
	Input: accumulated generation state.
	Output: displays the current experiment summary in BrainInfoLabel.
	"""
	state_mutex.lock()

	var local_stop_requested = stop_requested
	var local_progress = generation_progress
	var local_current_brain = current_brain_number
	var local_generated = generated_brains
	var local_valid = valid_brains
	var local_invalid = invalid_brains
	var local_requiring_rescue = brains_requiring_rescue
	var local_rescued = brains_rescued_successfully

	var local_before_internal = isolated_before_internal
	var local_before_output = isolated_before_output
	var local_before_input = isolated_before_input

	var local_after_internal = isolated_after_internal
	var local_after_output = isolated_after_output
	var local_after_input = isolated_after_input

	var local_no_structural = failure_no_structural_candidate
	var local_no_positive_h = failure_no_positive_h_candidate
	var local_saturated = failure_positive_h_candidates_saturated
	var local_input_saturated = failure_input_source_saturated

	var local_valid_max_connections_sum = valid_max_connections_sum
	var local_invalid_max_connections_sum = invalid_max_connections_sum
	var local_nmax_total_counts: Dictionary = nmax_total_counts.duplicate()
	var local_nmax_valid_counts: Dictionary = nmax_valid_counts.duplicate()

	var local_reciprocal_rate_sum = reciprocal_rate_sum
	var local_reciprocal_rate_squared_sum = reciprocal_rate_squared_sum
	var local_reciprocal_rate_min = reciprocal_rate_min
	var local_reciprocal_rate_max = reciprocal_rate_max
	var local_valid_reciprocal_rate_sum = valid_reciprocal_rate_sum
	var local_invalid_reciprocal_rate_sum = invalid_reciprocal_rate_sum
	var local_total_directional_connections = total_directional_connections
	var local_total_reciprocal_directional_connections = total_reciprocal_directional_connections

	state_mutex.unlock()

	var running = _is_generating()
	var status = "Stopped"

	if running:
		status = "Stopping after current brain..." if local_stop_requested else "Running"

	var elapsed_seconds = 0.0

	if run_start_msec > 0:
		var end_msec: int = run_end_msec

		if running:
			end_msec = Time.get_ticks_msec()

		if end_msec > 0:
			elapsed_seconds = (end_msec - run_start_msec) / 1000.0

	var valid_percentage = 0.0
	var invalid_percentage = 0.0
	var rescue_success_percentage = 0.0
	var brains_per_second = 0.0
	var mean_valid_max_connections = 0.0
	var mean_invalid_max_connections = 0.0
	var mean_reciprocal_rate = 0.0
	var reciprocal_rate_standard_deviation = 0.0
	var mean_valid_reciprocal_rate = 0.0
	var mean_invalid_reciprocal_rate = 0.0
	var pooled_reciprocal_rate = 0.0
	var displayed_reciprocal_rate_min = 0.0
	var displayed_reciprocal_rate_max = 0.0

	if local_generated > 0:
		valid_percentage = 100.0 * local_valid / local_generated
		invalid_percentage = 100.0 * local_invalid / local_generated
		mean_reciprocal_rate = local_reciprocal_rate_sum / local_generated
		var reciprocal_variance = maxf(
			0.0,
			local_reciprocal_rate_squared_sum / local_generated
			- mean_reciprocal_rate * mean_reciprocal_rate
		)
		reciprocal_rate_standard_deviation = sqrt(reciprocal_variance)
		displayed_reciprocal_rate_min = local_reciprocal_rate_min
		displayed_reciprocal_rate_max = local_reciprocal_rate_max

	if local_requiring_rescue > 0:
		rescue_success_percentage = 100.0 * local_rescued / local_requiring_rescue

	if elapsed_seconds > 0.0:
		brains_per_second = local_generated / elapsed_seconds

	if local_valid > 0:
		mean_valid_max_connections = local_valid_max_connections_sum / local_valid
		mean_valid_reciprocal_rate = local_valid_reciprocal_rate_sum / local_valid

	if local_invalid > 0:
		mean_invalid_max_connections = local_invalid_max_connections_sum / local_invalid
		mean_invalid_reciprocal_rate = local_invalid_reciprocal_rate_sum / local_invalid

	if local_total_directional_connections > 0:
		pooled_reciprocal_rate = (
			100.0
			* local_total_reciprocal_directional_connections
			/ local_total_directional_connections
		)

	var before_total = local_before_internal + local_before_output + local_before_input
	var after_total = local_after_internal + local_after_output + local_after_input
	var rescued_isolated_neurons = before_total - after_total
	var rescued_isolated_percentage = 0.0
	var mean_residual_isolated_per_invalid_brain = 0.0

	if before_total > 0:
		rescued_isolated_percentage = 100.0 * rescued_isolated_neurons / before_total

	if local_invalid > 0:
		mean_residual_isolated_per_invalid_brain = float(after_total) / local_invalid

	if local_generated != last_plot_generated:
		nmax_plot.set_bins(_build_nmax_bins(local_nmax_total_counts, local_nmax_valid_counts))
		last_plot_generated = local_generated

	var progress_text = ""

	if running:
		progress_text = "Current brain: %d\nProgress: %d%%\n" % [
			local_current_brain,
			local_progress
		]

	$BrainInfoLabel.text = (
		"Random Brain Validity Experiment\n"
		+ "Status: %s\n" % status
		+ progress_text
		+ "Generated brains: %d\n" % local_generated
		+ "Valid brains: %d (%.2f%%)\n" % [local_valid, valid_percentage]
		+ "Invalid brains: %d (%.2f%%)\n" % [local_invalid, invalid_percentage]
		+ "Brains requiring rescue: %d\n" % local_requiring_rescue
		+ "Successfully rescued brains: %d (%.2f%%)\n" % [
			local_rescued,
			rescue_success_percentage
		]
		+ "Rescued isolated neurons: %d (%.2f%%)\n" % [
			rescued_isolated_neurons,
			rescued_isolated_percentage
		]
		+ "\nIsolated neurons before rescue: %d\n" % before_total
		+ "     Internal without incoming: %d\n" % local_before_internal
		+ "     Output without incoming: %d\n" % local_before_output
		+ "     Input without outgoing: %d\n" % local_before_input
		+ "\nIsolated neurons after rescue: %d\n" % after_total
		+ "     Internal without incoming: %d\n" % local_after_internal
		+ "     Output without incoming: %d\n" % local_after_output
		+ "     Input without outgoing: %d\n" % local_after_input
		+ "\nRescue failures by cause:\n"
		+ "     No structurally valid candidate: %d\n" % local_no_structural
		+ "     No candidate with H > 0: %d\n" % local_no_positive_h
		+ "     H > 0 candidates saturated: %d\n" % local_saturated
		+ "     Input source at N_max: %d\n" % local_input_saturated
		+ "\nMean N_max of valid brains: %.3f\n" % mean_valid_max_connections
		+ "Mean N_max of invalid brains: %.3f\n" % mean_invalid_max_connections
		+ "\nReciprocal directional connections:\n"
		+ "     Mean rate / brain: %.2f%%\n" % mean_reciprocal_rate
		+ "     Standard deviation: %.2f pp\n" % reciprocal_rate_standard_deviation
		+ "     Range: %.2f%% - %.2f%%\n" % [
			displayed_reciprocal_rate_min,
			displayed_reciprocal_rate_max
		]
		+ "     Mean valid-brain rate: %.2f%%\n" % mean_valid_reciprocal_rate
		+ "     Mean invalid-brain rate: %.2f%%\n" % mean_invalid_reciprocal_rate
		+ "     Pooled rate: %.2f%% (%d / %d)\n" % [
			pooled_reciprocal_rate,
			local_total_reciprocal_directional_connections,
			local_total_directional_connections
		]
		+ "\nMean residual isolated / invalid brain: %.3f\n" % mean_residual_isolated_per_invalid_brain
		+ "Generation rate: %.3f brains/s\n" % brains_per_second
		+ "Elapsed time: %.1f s" % elapsed_seconds
	)


func _reset_statistics() -> void:
	"""
	Input: none.
	Output: clears every statistic before a new experiment starts.
	"""
	state_mutex.lock()

	stop_requested = false
	generation_progress = 0
	current_brain_number = 0

	generated_brains = 0
	valid_brains = 0
	invalid_brains = 0

	brains_requiring_rescue = 0
	brains_rescued_successfully = 0

	isolated_before_internal = 0
	isolated_before_output = 0
	isolated_before_input = 0

	isolated_after_internal = 0
	isolated_after_output = 0
	isolated_after_input = 0

	failure_no_structural_candidate = 0
	failure_no_positive_h_candidate = 0
	failure_positive_h_candidates_saturated = 0
	failure_input_source_saturated = 0

	valid_max_connections_sum = 0.0
	invalid_max_connections_sum = 0.0
	nmax_total_counts.clear()
	nmax_valid_counts.clear()

	reciprocal_rate_sum = 0.0
	reciprocal_rate_squared_sum = 0.0
	reciprocal_rate_min = INF
	reciprocal_rate_max = -INF
	valid_reciprocal_rate_sum = 0.0
	invalid_reciprocal_rate_sum = 0.0
	total_directional_connections = 0
	total_reciprocal_directional_connections = 0

	state_mutex.unlock()

	last_plot_generated = -1

	if nmax_plot != null:
		nmax_plot.set_bins([])


func _build_nmax_bins(total_counts: Dictionary, valid_counts: Dictionary) -> Array:
	"""
	Input: exact N_max total and valid brain counts.
	Output: aggregates them into fixed-width bins containing sample size and empirical validity percentage.
	"""
	if total_counts.is_empty():
		return []

	var keys = total_counts.keys()
	var minimum_nmax = int(keys[0])
	var maximum_nmax = int(keys[0])

	for key in keys:
		var nmax = int(key)
		minimum_nmax = mini(minimum_nmax, nmax)
		maximum_nmax = maxi(maximum_nmax, nmax)

	var first_bin = int(floor(float(minimum_nmax) / NMAX_BIN_WIDTH)) * NMAX_BIN_WIDTH
	var last_bin = int(floor(float(maximum_nmax) / NMAX_BIN_WIDTH)) * NMAX_BIN_WIDTH
	var bins: Array = []

	for bin_start in range(first_bin, last_bin + 1, NMAX_BIN_WIDTH):
		var total = 0
		var valid = 0
		var bin_end = bin_start + NMAX_BIN_WIDTH - 1

		for key in keys:
			var nmax = int(key)

			if nmax < bin_start or nmax > bin_end:
				continue

			total += int(total_counts.get(nmax, 0))
			valid += int(valid_counts.get(nmax, 0))

		if total == 0:
			continue

		bins.append({
			"label": "%d-%d" % [bin_start, bin_end],
			"total": total,
			"valid": valid,
			"rate": 100.0 * valid / total
		})

	return bins


func _set_external_fields_editable(editable: bool) -> void:
	"""
	Input: whether the external parameter fields may be edited.
	Output: applies the editability state to all five fields.
	"""
	for field in [$LineEdit, $LineEdit2, $LineEdit3, $LineEdit4, $LineEdit5]:
		field.editable = editable


func _is_generating() -> bool:
	"""
	Input: none.
	Output: true while the generation thread exists and is active.
	"""
	return generation_thread != null and generation_thread.is_started()
