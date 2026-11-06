extends Node2D


const NMAX_MAX_BARS = 10
const RESCUE_BIN_WIDTH = 5
const RESCUE_BIN_COUNT = 20

const GENES_64 = [
	"connections", "initial_search_region", "activation_threshold", "decay_factor",
	"retention_factor", "polarity_factor", "hebbian_plasticity_rate",
	"modulatory_release_factor", "modulatory_sensitivity"
]
const GENES_8 = [
	"max_connections", "beta", "refractory_strength", "exponential_factor",
	"modulatory_dynamics", "structural_plasticity", "input_influence"
]




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


class RescueNeedPlot:
	extends Control

	var counts: PackedInt32Array = PackedInt32Array()
	var total_brains: int = 0
	var mean_rate: float = 0.0
	var standard_deviation: float = 0.0


	func set_distribution(new_counts: PackedInt32Array, new_total: int, new_mean: float, new_sd: float) -> void:
		"""
		Input: distribution of per-brain isolation percentages and its summary statistics.
		Output: refreshes the histogram without retaining any brain's connection data.
		"""
		counts = new_counts.duplicate()
		total_brains = new_total
		mean_rate = new_mean
		standard_deviation = new_sd
		queue_redraw()


	func _draw() -> void:
		var font = ThemeDB.fallback_font
		var title_color = Color(1.0, 1.0, 1.0, 0.92)
		var text_color = Color(1.0, 1.0, 1.0, 0.83)
		var grid_color = Color(1.0, 1.0, 1.0, 0.18)
		var bar_color = Color(0.95, 0.55, 0.20, 0.90)
		var left = 53.0
		var top = 55.0
		var right = 12.0
		var bottom = 64.0
		var graph_width = size.x - left - right
		var graph_height = size.y - top - bottom

		draw_string(font, Vector2(10.0, 22.0), "Rescue need per brain", HORIZONTAL_ALIGNMENT_LEFT, -1.0, 18, title_color)

		if total_brains == 0 or graph_width <= 0.0 or graph_height <= 0.0:
			draw_string(font, Vector2(left, top + 30.0), "Waiting for generated brains...", HORIZONTAL_ALIGNMENT_LEFT, -1.0, 13, text_color)
			return

		# Heights show the percentage of brains in each bin (not neuron counts).
		var max_percentage = 0.0
		for count in counts:
			max_percentage = maxf(max_percentage, 100.0 * count / total_brains)
		var axis_max = clampf(ceil(max_percentage / 10.0) * 10.0, 10.0, 100.0)

		for tick in 3:
			var tick_ratio = float(tick) / 2.0
			var y = top + graph_height * (1.0 - tick_ratio)
			draw_line(Vector2(left, y), Vector2(left + graph_width, y), grid_color, 1.0)
			draw_string(font, Vector2(5.0, y + 4.0), "%.0f%%" % (axis_max * tick_ratio), HORIZONTAL_ALIGNMENT_RIGHT, left - 12.0, 12, text_color)

		draw_line(Vector2(left, top), Vector2(left, top + graph_height), text_color, 1.0)
		draw_line(Vector2(left, top + graph_height), Vector2(left + graph_width, top + graph_height), text_color, 1.0)

		var slot_width = graph_width / 20.0
		var bar_width = maxf(2.0, slot_width * 0.76)
		for index in counts.size():
			var percentage = 100.0 * counts[index] / total_brains
			if percentage <= 0.0:
				continue
			var bar_height = graph_height * percentage / axis_max
			var center_x = left + slot_width * (index + 0.5)
			draw_rect(Rect2(center_x - bar_width / 2.0, top + graph_height - bar_height, bar_width, bar_height), bar_color)

		for percentage in range(0, 101, 20):
			var x = left + graph_width * float(percentage) / 100.0
			draw_string(font, Vector2(x - 13.0, top + graph_height + 19.0), str(percentage), HORIZONTAL_ALIGNMENT_CENTER, 26.0, 12, text_color)

		draw_string(font, Vector2(left, top + graph_height + 39.0), "Neurons needing rescue (%)", HORIZONTAL_ALIGNMENT_LEFT, -1.0, 12, text_color)
		draw_string(font, Vector2(10.0, size.y - 5.0), "Mean: %.1f%%    SD: %.1f pp" % [mean_rate, standard_deviation], HORIZONTAL_ALIGNMENT_LEFT, -1.0, 13, text_color)


	func _get_tooltip(at_position: Vector2) -> String:
		# Each bar covers [5*i, 5*(i+1)) percent, except the last, which includes 100%.
		var left = 53.0
		var width = size.x - left - 12.0
		if total_brains == 0 or width <= 0.0 or at_position.x < left or at_position.x > left + width:
			return ""
		if at_position.y < 55.0 or at_position.y > size.y - 64.0:
			return ""
		var index = clampi(int((at_position.x - left) / width * 20.0), 0, 19)
		var count = counts[index]
		var label = "%d%%–%d%%" % [index * 5, (index + 1) * 5]
		return "%s neurons needing rescue\n%d of %d brains (%.2f%%)" % [label, count, total_brains, 100.0 * count / total_brains]


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

# Counts of actual directed synapses, not attempted candidates.
var initial_connection_sum = 0
var initial_first_pass_sum = 0
var initial_second_pass_sum = 0
var rescue_connection_sum = 0

var interface_input_count_sum = 0
var interface_output_count_sum = 0
var interface_input_outgoing_sum = 0
var interface_output_incoming_sum = 0
var interface_input_mean_outgoing_sum = 0.0
var interface_output_mean_incoming_sum = 0.0
var interface_input_connection_share_sum = 0.0
var interface_output_connection_share_sum = 0.0
# Accumulate exact Jaccard percentages and counts of defined pairs.
var input_jaccard_all_sum = 0.0
var input_jaccard_all_count = 0
var input_jaccard_nearest_sum = 0.0
var input_jaccard_nearest_count = 0
var output_jaccard_all_sum = 0.0
var output_jaccard_all_count = 0
var output_jaccard_nearest_sum = 0.0
var output_jaccard_nearest_count = 0

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

# One 5-percentage-point bin per brain; populated only by the worker thread.
var rescue_need_counts: PackedInt32Array = PackedInt32Array()
var rescue_need_rate_sum = 0.0
var rescue_need_rate_squared_sum = 0.0

var reciprocal_rate_sum = 0.0
var reciprocal_rate_squared_sum = 0.0
var reciprocal_rate_min = INF
var reciprocal_rate_max = -INF
var valid_reciprocal_rate_sum = 0.0
var invalid_reciprocal_rate_sum = 0.0
var total_directional_connections = 0
var total_reciprocal_directional_connections = 0

var nmax_plot: NMaxValidityPlot
var rescue_need_plot: RescueNeedPlot
var last_plot_generated = -1

var run_start_msec = 0
var run_end_msec = 0
var worker_error: String = ""

func _ready() -> void:
	"""
	Input: none.
	Output: initializes the interface and connects the controls.
	"""
	$Start.pressed.connect(_on_start_pressed)
	$Stop.pressed.connect(_on_stop_pressed)

	for field in [$LineEdit, $LineEdit2, $LineEdit3, $LineEdit4, $LineEdit5, $LineEdit6]:
		field.text_changed.connect(_on_external_parameter_changed)
	$LineEdit6.tooltip_text = "Connections density eta (> 0); 1 restores the original genetic ceiling."
	if $LineEdit6.text.is_empty():
		$LineEdit6.text = "1"

	$BrainInfoLabel.size = Vector2(390.0, 780.0)
	$BrainInfoLabel.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART

	nmax_plot = NMaxValidityPlot.new()
	nmax_plot.position = Vector2(410.0, 330.0)
	nmax_plot.size = Vector2(790.0, 410.0)
	nmax_plot.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(nmax_plot)

	# Compact companion chart, below External Parameters and right of the N_max plot.
	rescue_need_plot = RescueNeedPlot.new()
	rescue_need_plot.position = Vector2(1260.0, 360.0)
	rescue_need_plot.size = Vector2(390.0, 390.0)
	add_child(rescue_need_plot)

	# Descriptive boxes sit below both charts, aligned at the same height.
	var caption_top = maxf(
		nmax_plot.position.y + nmax_plot.size.y,
		rescue_need_plot.position.y + rescue_need_plot.size.y
	) + 14.0
	_add_chart_caption(
		Vector2(nmax_plot.position.x + 55.0, caption_top),
		Vector2(nmax_plot.size.x - 55.0, 100.0),
		"Each bar gives the percentage of valid brains among all brains whose N_max "
		+ "falls within that interval. The n label shows how many brains were tested "
		+ "in the interval."
	)
	_add_chart_caption(
		Vector2(rescue_need_plot.position.x + 10.0, caption_top),
		Vector2(rescue_need_plot.size.x - 10.0, 100.0),
		"Each bar shows the percentage of brains with that proportion of neurons "
		+ "isolated before rescue (5 percentage-point intervals). Mean and SD "
		+ "describe the percentages across brains."
	)

	$Stop.disabled = true
	_validate_form()
	_update_summary()


func _add_chart_caption(at: Vector2, dimensions: Vector2, description: String) -> void:
	"""Create a subtle gray explanation panel without changing the scene file."""
	var panel = PanelContainer.new()
	panel.position = at
	panel.size = dimensions
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE

	var background = StyleBoxFlat.new()
	background.bg_color = Color(0.27, 0.28, 0.29, 0.9)
	background.border_color = Color(0.7, 0.7, 0.7, 0.35)
	background.set_border_width_all(1)
	background.set_corner_radius_all(6)
	background.set_content_margin_all(11.0)
	panel.add_theme_stylebox_override("panel", background)

	var caption = Label.new()
	caption.text = description
	caption.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	caption.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	caption.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	caption.size_flags_vertical = Control.SIZE_EXPAND_FILL
	caption.mouse_filter = Control.MOUSE_FILTER_IGNORE
	caption.add_theme_font_size_override("font_size", 14)
	caption.add_theme_color_override("font_color", Color(0.97, 0.97, 0.97))
	panel.add_child(caption)
	add_child(panel)


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

	if not _valid_connections_density($LineEdit6.text):
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


func _valid_connections_density(value: String) -> bool:
	"""Validates the positive finite density factor specified by the model."""
	if not value.is_valid_float():
		return false
	var eta := float(value)
	return not is_nan(eta) and not is_inf(eta) and eta > 0.0


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
	var connections_density := float($LineEdit6.text)

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
		_generation_loop.bind(size, num_inputs, num_outputs, connections_density)
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


# The random-gene distribution matches the Explorer: independent uniform digits 0..3.
# Keep all keys present; the GPU builder expects the full modern genome.
func _random_genome(rng: RandomNumberGenerator) -> Dictionary:
	var genome: Dictionary = {}
	for key in GENES_64:
		var digits: Array[int] = []
		for i in 64:
			digits.append(rng.randi_range(0, 3))
		genome[key] = digits
	for key in GENES_8:
		var digits: Array[int] = []
		for i in 8:
			digits.append(rng.randi_range(0, 3))
		genome[key] = digits
	return genome


func _set_worker_error(message: String) -> void:
	state_mutex.lock()
	worker_error = message
	stop_requested = true
	state_mutex.unlock()


func _generation_loop(size: Vector3i, num_inputs: int, num_outputs: int, connections_density: float) -> void:
	"""Generate random brains with the validated GPU constructor and evaluate GPU reciprocity."""
	var builder := BrainBuilderGPU.new()
	var reciprocity_gpu := BrainFailuresGPU.new()
	var rng := RandomNumberGenerator.new()
	rng.randomize()
	# Reuse one local RD for every brain, avoiding device-creation overhead and
	# allocating/freeing each brain's buffers before constructing the next one.
	var rd: RenderingDevice = RenderingServer.create_local_rendering_device()
	if rd == null:
		_set_worker_error("Could not create local GPU RenderingDevice.")
		return
	while not _should_stop():
		state_mutex.lock()
		generation_progress = 0
		current_brain_number = generated_brains + 1
		state_mutex.unlock()
		var genome: Dictionary = _random_genome(rng)
		var brain: Dictionary = builder.build(
			genome, size, num_inputs, num_outputs,
			_set_build_progress_from_thread, true, rd, connections_density
		)
		if brain.is_empty():
			_set_worker_error("BrainBuilderGPU failed; check the Godot debugger for details.")
			break
		var reciprocity: Dictionary = reciprocity_gpu.calculate(brain)
		# Read the exact synaptic partners before releasing the GPU phenotype.
		var overlap: Dictionary = _measure_interface_overlap(brain) if not reciprocity.is_empty() else {}
		builder.release_gpu_state(brain)
		if reciprocity.is_empty():
			_set_worker_error("GPU reciprocity failed; check the shader compilation and GPU logs.")
			break
		if overlap.is_empty():
			_set_worker_error("Could not read GPU connection slots for input/output overlap.")
			break
		_record_brain(brain, reciprocity, overlap)
		_set_generation_progress_from_thread(100)
	rd.free()
	state_mutex.lock()
	generation_progress = 100
	state_mutex.unlock()


func _set_build_progress_from_thread(percent: int) -> void:
	# Reserve the last 10% for GPU reciprocity and accumulation.
	_set_generation_progress_from_thread(clampi(int(percent * 0.9), 0, 90))


func _set_generation_progress_from_thread(percent: int) -> void:
	"""
	Input: build progress from 0 to 100 for the current brain.
	Output: stores the latest progress for the interface thread.
	"""
	state_mutex.lock()
	generation_progress = percent
	state_mutex.unlock()



# These methods run in the generation worker, while its GPU buffers still exist.
# Only the current brain's slot slices are read; no edge history is retained.
func _measure_interface_overlap(brain: Dictionary) -> Dictionary:
	var state: Dictionary = brain["gpu_state"]
	var rd: RenderingDevice = state["rd"]
	var dimensions: Vector3i = brain["size"]
	var plane: int = dimensions.x * dimensions.y
	var total: int = plane * dimensions.z
	var stride: int = int(state["slot_stride"])
	var outgoing: PackedInt32Array = brain["outgoing_counts"]
	var input_index: Dictionary = {}
	var output_index: Dictionary = {}
	var input_positions: Array[Vector3i] = []
	var output_positions: Array[Vector3i] = []
	var input_destinations: Array[Dictionary] = []
	var output_sources: Array[Dictionary] = []

	for neuron in brain["neurons"]:
		var pos: Vector3i = neuron["position"]
		var spatial: int = _spatial_index(pos, dimensions)
		if neuron["input"]:
			input_index[spatial] = input_positions.size()
			input_positions.append(pos)
			input_destinations.append({})
		elif neuron["output"]:
			output_index[spatial] = output_positions.size()
			output_positions.append(pos)
			output_sources.append({})

	# A pair exists only when its two members have at least one synaptic partner.
	# No need to read slots when neither role has two neurons to compare.
	if input_positions.size() > 1 or output_positions.size() > 1:
		for k in dimensions.z:
			var slot_bytes: PackedByteArray = rd.buffer_get_data(state["slot_buffers"][k])
			if slot_bytes.size() != plane * stride * 16:
				return {}
			for local_source in plane:
				var source: int = k * plane + local_source
				var count: int = outgoing[source]
				if count < 0 or count > stride:
					return {}
				var input_id: int = int(input_index.get(source, -1))
				for c in count:
					var target: int = slot_bytes.decode_s32((local_source * stride + c) * 16)
					if target < 0 or target >= total:
						return {}
					if input_id >= 0:
						input_destinations[input_id][target] = true
					var output_id: int = int(output_index.get(target, -1))
					if output_id >= 0:
						output_sources[output_id][source] = true

	var input_stats: Dictionary = _interface_pairwise_jaccard(input_destinations, input_positions)
	var output_stats: Dictionary = _interface_pairwise_jaccard(output_sources, output_positions)
	return {
		"input_all_sum": input_stats["all_sum"],
		"input_all_count": input_stats["all_count"],
		"input_nearest_sum": input_stats["nearest_sum"],
		"input_nearest_count": input_stats["nearest_count"],
		"output_all_sum": output_stats["all_sum"],
		"output_all_count": output_stats["all_count"],
		"output_nearest_sum": output_stats["nearest_sum"],
		"output_nearest_count": output_stats["nearest_count"]
	}


func _interface_pairwise_jaccard(sets: Array[Dictionary], positions: Array[Vector3i]) -> Dictionary:
	var all_sum := 0.0
	var all_count := 0
	var nearest_sum := 0.0
	var nearest_count := 0
	var n: int = sets.size()
	# Each neuron selects its closest spatial partner; each undirected pair
	# is measured once, even if both choose one another.
	var nearest_pairs: Dictionary = {}
	for i in n:
		var nearest := -1
		var best_distance := 9223372036854775807
		for j in n:
			if i == j:
				continue
			var delta: Vector3i = positions[i] - positions[j]
			var squared_distance: int = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
			if squared_distance < best_distance:
				best_distance = squared_distance
				nearest = j
		if nearest >= 0:
			nearest_pairs[mini(i, nearest) * n + maxi(i, nearest)] = true
	for i in n:
		for j in range(i + 1, n):
			var first: Dictionary = sets[i]
			var second: Dictionary = sets[j]
			var union_size: int = first.size() + second.size()
			if union_size == 0:
				continue # Jaccard(empty, empty) is undefined.
			var common := 0
			if first.size() <= second.size():
				for partner in first:
					if second.has(partner):
						common += 1
			else:
				for partner in second:
					if first.has(partner):
						common += 1
			var jaccard: float = 100.0 * float(common) / float(union_size - common)
			all_sum += jaccard
			all_count += 1
			if nearest_pairs.has(i * n + j):
				nearest_sum += jaccard
				nearest_count += 1
	return {"all_sum": all_sum, "all_count": all_count, "nearest_sum": nearest_sum, "nearest_count": nearest_count}


func _record_brain(brain: Dictionary, reciprocity: Dictionary, overlap: Dictionary) -> void:
	"""
	Input: one completed GPU brain, GPU reciprocity, and interface overlap statistics.
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
	var dimensions: Vector3i = brain["size"]
	var neuron_count: int = dimensions.x * dimensions.y * dimensions.z
	var rescue_need_percentage: float = 100.0 * float(before_total) / float(neuron_count)
	var rescue_bin_index = clampi(int(floor(rescue_need_percentage / RESCUE_BIN_WIDTH)), 0, RESCUE_BIN_COUNT - 1)
	var reciprocal_rate = float(reciprocity["rate"])
	var first_pass_connections: int = int(brain["initial_local_connections"])
	var second_pass_connections: int = int(brain["initial_additional_connections"])
	var initial_connections: int = first_pass_connections + second_pass_connections
	var rescue_connections: int = int(brain["rescue_edge_count"])
	var directional_connections = int(reciprocity["total"])
	var reciprocal_directional_connections = int(reciprocity["reciprocal"])
	var neurons: Array = brain["neurons"]
	var incoming_counts: PackedInt32Array = brain["incoming_counts"]
	var outgoing_counts: PackedInt32Array = brain["outgoing_counts"]
	var input_count := 0
	var output_count := 0
	var input_outgoing_total := 0
	var output_incoming_total := 0
	for neuron in neurons:
		var spatial: int = _spatial_index(neuron["position"], dimensions)
		if neuron["input"]:
			input_count += 1
			input_outgoing_total += outgoing_counts[spatial]
		elif neuron["output"]:
			output_count += 1
			output_incoming_total += incoming_counts[spatial]
	var input_mean_outgoing := 0.0 if input_count == 0 else float(input_outgoing_total) / float(input_count)
	var output_mean_incoming := 0.0 if output_count == 0 else float(output_incoming_total) / float(output_count)
	var input_connection_share := 0.0 if directional_connections == 0 else 100.0 * float(input_outgoing_total) / float(directional_connections)
	var output_connection_share := 0.0 if directional_connections == 0 else 100.0 * float(output_incoming_total) / float(directional_connections)

	state_mutex.lock()

	generated_brains += 1
	initial_connection_sum += initial_connections
	initial_first_pass_sum += first_pass_connections
	initial_second_pass_sum += second_pass_connections
	rescue_connection_sum += rescue_connections
	interface_input_count_sum += input_count
	interface_output_count_sum += output_count
	interface_input_outgoing_sum += input_outgoing_total
	interface_output_incoming_sum += output_incoming_total
	interface_input_mean_outgoing_sum += input_mean_outgoing
	interface_output_mean_incoming_sum += output_mean_incoming
	interface_input_connection_share_sum += input_connection_share
	interface_output_connection_share_sum += output_connection_share
	input_jaccard_all_sum += float(overlap["input_all_sum"])
	input_jaccard_all_count += int(overlap["input_all_count"])
	input_jaccard_nearest_sum += float(overlap["input_nearest_sum"])
	input_jaccard_nearest_count += int(overlap["input_nearest_count"])
	output_jaccard_all_sum += float(overlap["output_all_sum"])
	output_jaccard_all_count += int(overlap["output_all_count"])
	output_jaccard_nearest_sum += float(overlap["output_nearest_sum"])
	output_jaccard_nearest_count += int(overlap["output_nearest_count"])
	rescue_need_counts[rescue_bin_index] += 1
	rescue_need_rate_sum += rescue_need_percentage
	rescue_need_rate_squared_sum += rescue_need_percentage * rescue_need_percentage

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


func _should_stop() -> bool:
	"""
	Input: none.
	Output: true when Stop has been requested.
	"""
	state_mutex.lock()
	var requested = stop_requested
	state_mutex.unlock()
	return requested


func _spatial_index(pos: Vector3i, size: Vector3i) -> int:
	return pos.x + size.x * (pos.y + size.y * pos.z)


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
	var local_worker_error = worker_error
	var local_progress = generation_progress
	var local_current_brain = current_brain_number
	var local_generated = generated_brains
	var local_valid = valid_brains
	var local_invalid = invalid_brains
	var local_requiring_rescue = brains_requiring_rescue
	var local_rescued = brains_rescued_successfully
	var local_initial_connection_sum = initial_connection_sum
	var local_initial_first_pass_sum = initial_first_pass_sum
	var local_initial_second_pass_sum = initial_second_pass_sum
	var local_rescue_connection_sum = rescue_connection_sum
	var local_interface_input_count_sum = interface_input_count_sum
	var local_interface_output_count_sum = interface_output_count_sum
	var local_interface_input_outgoing_sum = interface_input_outgoing_sum
	var local_interface_output_incoming_sum = interface_output_incoming_sum
	var local_interface_input_mean_outgoing_sum = interface_input_mean_outgoing_sum
	var local_interface_output_mean_incoming_sum = interface_output_mean_incoming_sum
	var local_interface_input_connection_share_sum = interface_input_connection_share_sum
	var local_interface_output_connection_share_sum = interface_output_connection_share_sum
	var local_input_jaccard_all_sum = input_jaccard_all_sum
	var local_input_jaccard_all_count = input_jaccard_all_count
	var local_input_jaccard_nearest_sum = input_jaccard_nearest_sum
	var local_input_jaccard_nearest_count = input_jaccard_nearest_count
	var local_output_jaccard_all_sum = output_jaccard_all_sum
	var local_output_jaccard_all_count = output_jaccard_all_count
	var local_output_jaccard_nearest_sum = output_jaccard_nearest_sum
	var local_output_jaccard_nearest_count = output_jaccard_nearest_count

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
	var local_rescue_need_counts = rescue_need_counts.duplicate()
	var local_rescue_need_rate_sum = rescue_need_rate_sum
	var local_rescue_need_rate_squared_sum = rescue_need_rate_squared_sum

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
	var status = "Failed" if not local_worker_error.is_empty() else "Stopped"

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
	var mean_initial_connections = 0.0
	var first_pass_percentage = 0.0
	var second_pass_percentage = 0.0
	var mean_rescue_connections = 0.0
	var mean_valid_max_connections = 0.0
	var mean_invalid_max_connections = 0.0
	var mean_reciprocal_rate = 0.0
	var reciprocal_rate_standard_deviation = 0.0
	var mean_valid_reciprocal_rate = 0.0
	var mean_invalid_reciprocal_rate = 0.0
	var pooled_reciprocal_rate = 0.0
	var displayed_reciprocal_rate_min = 0.0
	var displayed_reciprocal_rate_max = 0.0
	var mean_input_outgoing_per_input = 0.0
	var mean_input_total_outgoing_per_brain = 0.0
	var mean_input_connection_share = 0.0
	var mean_output_incoming_per_output = 0.0
	var mean_output_total_incoming_per_brain = 0.0
	var mean_output_connection_share = 0.0
	var zero_outgoing_input_rate = 0.0
	var zero_incoming_output_rate = 0.0

	if local_generated > 0:
		mean_initial_connections = float(local_initial_connection_sum) / local_generated
		if local_initial_connection_sum > 0:
			first_pass_percentage = 100.0 * float(local_initial_first_pass_sum) / float(local_initial_connection_sum)
			second_pass_percentage = 100.0 * float(local_initial_second_pass_sum) / float(local_initial_connection_sum)
		mean_rescue_connections = float(local_rescue_connection_sum) / local_generated
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
		mean_input_total_outgoing_per_brain = float(local_interface_input_outgoing_sum) / float(local_generated)
		mean_output_total_incoming_per_brain = float(local_interface_output_incoming_sum) / float(local_generated)
		mean_input_connection_share = local_interface_input_connection_share_sum / float(local_generated)
		mean_output_connection_share = local_interface_output_connection_share_sum / float(local_generated)
		mean_input_outgoing_per_input = 0.0 if local_interface_input_count_sum == 0 else float(local_interface_input_outgoing_sum) / float(local_interface_input_count_sum)
		mean_output_incoming_per_output = 0.0 if local_interface_output_count_sum == 0 else float(local_interface_output_incoming_sum) / float(local_interface_output_count_sum)
		zero_outgoing_input_rate = 0.0 if local_interface_input_count_sum == 0 else 100.0 * float(local_before_input) / float(local_interface_input_count_sum)
		zero_incoming_output_rate = 0.0 if local_interface_output_count_sum == 0 else 100.0 * float(local_before_output) / float(local_interface_output_count_sum)

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
		var mean_rescue_need = 0.0
		var sd_rescue_need = 0.0
		if local_generated > 0:
			mean_rescue_need = local_rescue_need_rate_sum / local_generated
			sd_rescue_need = sqrt(maxf(0.0, local_rescue_need_rate_squared_sum / local_generated - mean_rescue_need * mean_rescue_need))
		rescue_need_plot.set_distribution(local_rescue_need_counts, local_generated, mean_rescue_need, sd_rescue_need)
		last_plot_generated = local_generated

	var input_all_text := "n/a" if local_input_jaccard_all_count == 0 else "%.2f%%" % (local_input_jaccard_all_sum / local_input_jaccard_all_count)
	var input_nearest_text := "n/a" if local_input_jaccard_nearest_count == 0 else "%.2f%%" % (local_input_jaccard_nearest_sum / local_input_jaccard_nearest_count)
	var output_all_text := "n/a" if local_output_jaccard_all_count == 0 else "%.2f%%" % (local_output_jaccard_all_sum / local_output_jaccard_all_count)
	var output_nearest_text := "n/a" if local_output_jaccard_nearest_count == 0 else "%.2f%%" % (local_output_jaccard_nearest_sum / local_output_jaccard_nearest_count)
	$BrainInfoLabel2.text = (
		"INTERFACE CONNECTIONS\n"
		+ "Inputs (sources):\n"
		+ "Mean outgoing / input: %.2f\n" % mean_input_outgoing_per_input
		+ "Mean total outgoing / brain: %.2f\n" % mean_input_total_outgoing_per_brain
		+ "Mean share of all connections: %.2f%%\n" % mean_input_connection_share
		+ "Zero-outgoing before rescue: %.2f%%\n" % zero_outgoing_input_rate
		+ "Shared targets, Jaccard (all / nearest): %s / %s\n" % [input_all_text, input_nearest_text]
		+ "\nOutputs (targets):\n"
		+ "Mean incoming / output: %.2f\n" % mean_output_incoming_per_output
		+ "Mean total incoming / brain: %.2f\n" % mean_output_total_incoming_per_brain
		+ "Mean share of all connections: %.2f%%\n" % mean_output_connection_share
		+ "Zero-incoming before rescue: %.2f%%\n" % zero_incoming_output_rate
		+ "Shared sources, Jaccard (all / nearest): %s / %s" % [output_all_text, output_nearest_text]
	)

	var progress_text = ""

	if running:
		progress_text = "Current brain: %d\nProgress: %d%%\n" % [
			local_current_brain,
			local_progress
		]

	$BrainInfoLabel.text = (
		"RANDOM BRAIN VALIDITY EXPERIMENT\n"
		+ "Status: %s\n" % status
		+ ("Error: %s\n" % local_worker_error if not local_worker_error.is_empty() else "")
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
		+ "Mean initial connections / brain: %.2f\n" % mean_initial_connections
		+ "Initial connections (1st/2nd pass): %.1f%% / %.1f%%\n" % [first_pass_percentage, second_pass_percentage]
		+ "Mean connections added by rescue / brain: %.2f\n" % mean_rescue_connections
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
	worker_error = ""
	generation_progress = 0
	current_brain_number = 0

	generated_brains = 0
	valid_brains = 0
	invalid_brains = 0

	brains_requiring_rescue = 0
	brains_rescued_successfully = 0
	initial_connection_sum = 0
	initial_first_pass_sum = 0
	initial_second_pass_sum = 0
	rescue_connection_sum = 0
	interface_input_count_sum = 0
	interface_output_count_sum = 0
	interface_input_outgoing_sum = 0
	interface_output_incoming_sum = 0
	interface_input_mean_outgoing_sum = 0.0
	interface_output_mean_incoming_sum = 0.0
	interface_input_connection_share_sum = 0.0
	interface_output_connection_share_sum = 0.0
	input_jaccard_all_sum = 0.0
	input_jaccard_all_count = 0
	input_jaccard_nearest_sum = 0.0
	input_jaccard_nearest_count = 0
	output_jaccard_all_sum = 0.0
	output_jaccard_all_count = 0
	output_jaccard_nearest_sum = 0.0
	output_jaccard_nearest_count = 0

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
	rescue_need_counts = PackedInt32Array()
	rescue_need_counts.resize(RESCUE_BIN_COUNT)
	rescue_need_rate_sum = 0.0
	rescue_need_rate_squared_sum = 0.0

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
	if rescue_need_plot != null:
		rescue_need_plot.set_distribution(PackedInt32Array(), 0, 0.0, 0.0)


func _build_nmax_bins(total_counts: Dictionary, valid_counts: Dictionary) -> Array:
	"""
	Input: exact N_max frequencies for all sampled brains.
	Output: at most ten adaptive bins, each with an exact sample count and validity rate.
	Rebinning always uses the original per-N_max frequencies, never previous bars.
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

	# Choose the smallest readable 1/2/5 x 10^n interval that fits
	# the entire observed range into at most NMAX_MAX_BARS intervals.
	var span = maximum_nmax - minimum_nmax + 1
	var magnitude = 1
	var bin_width = 1
	var found = false
	while not found:
		for multiplier in [1, 2, 5]:
			bin_width = multiplier * magnitude
			if ceili(float(span) / float(bin_width)) <= NMAX_MAX_BARS:
				found = true
				break
		if not found:
			magnitude *= 10

	# Aggregate the original exact frequencies without allocating an array
	# spanning every possible N_max, even when values are far apart.
	var grouped: Dictionary = {}
	for key in keys:
		var nmax = int(key)
		var bin_index = int(floor(float(nmax - minimum_nmax) / float(bin_width)))
		if not grouped.has(bin_index):
			grouped[bin_index] = {"total": 0, "valid": 0}
		var group: Dictionary = grouped[bin_index]
		group["total"] += int(total_counts[key])
		group["valid"] += int(valid_counts.get(key, 0))

	var bins: Array = []
	var indices = grouped.keys()
	indices.sort()
	for index in indices:
		var bin_start = minimum_nmax + int(index) * bin_width
		var bin_end = bin_start + bin_width - 1
		var group: Dictionary = grouped[index]
		var total = int(group["total"])
		bins.append({
			"label": "%d-%d" % [bin_start, bin_end],
			"total": total,
			"valid": int(group["valid"]),
			"rate": 100.0 * int(group["valid"]) / total
		})

	return bins

func _set_external_fields_editable(editable: bool) -> void:
	"""
	Input: whether the external parameter fields may be edited.
	Output: applies the editability state to all six fields.
	"""
	for field in [$LineEdit, $LineEdit2, $LineEdit3, $LineEdit4, $LineEdit5, $LineEdit6]:
		field.editable = editable


func _is_generating() -> bool:
	"""
	Input: none.
	Output: true while the generation thread exists and is active.
	"""
	return generation_thread != null and generation_thread.is_started()
