extends Node2D


const GENE_SPECS = {
	"max_connections": 8,
	"beta": 8,
	"connections": 64,
	"activation_threshold": 64,
	"refractory_strength": 8,
	"exponential_factor": 8,
	"decay_factor": 64,
	"retention_factor": 64,
	"polarity_factor": 64,
	"hebbian_plasticity_rate": 64,
	"modulatory_release_factor": 64,
	"modulatory_sensitivity": 64,
	"modulatory_dynamics": 8,
	"structural_plasticity": 8,
	"input_influence": 8
}

const GENE_ORDER = [
	"max_connections",
	"connections",
	"activation_threshold",
	"refractory_strength",
	"exponential_factor",
	"decay_factor",
	"retention_factor",
	"polarity_factor",
	"hebbian_plasticity_rate",
	"structural_plasticity",
	"beta",
	"modulatory_release_factor",
	"modulatory_sensitivity",
	"modulatory_dynamics",
	"input_influence"
]

const GENE_LABELS = {
	"max_connections": "Maximum connections",
	"connections": "Connections",
	"activation_threshold": "Activation threshold",
	"refractory_strength": "Refractory strength",
	"exponential_factor": "Exponential factor",
	"decay_factor": "Decay factor",
	"retention_factor": "Retention factor",
	"polarity_factor": "Polarity factor",
	"hebbian_plasticity_rate": "Hebbian plasticity rate",
	"structural_plasticity": "Structural plasticity",
	"beta": "Beta",
	"modulatory_release_factor": "Modulatory release factor",
	"modulatory_sensitivity": "Modulatory sensitivity",
	"modulatory_dynamics": "Modulatory dynamics",
	"input_influence": "Input influence"
}

const EXTERNAL_PARAMETER_NAMES = [
	"size_x",
	"size_y",
	"size_z",
	"num_inputs",
	"num_outputs"
]

const EXTERNAL_PARAMETER_LABELS = {
	"size_x": "Size X",
	"size_y": "Size Y",
	"size_z": "Size Z",
	"num_inputs": "Inputs",
	"num_outputs": "Outputs"
}

const BINS := 20
const LOADED_BUTTON_MODULATE := Color(0.45, 1.0, 0.45, 1.0)

const HISTOGRAM_PARAMETER_KEYS = [
	"connection_weight",
	"incoming_connections",
	"outgoing_connections",
	"activation_threshold",
	"decay_factor",
	"retention_factor",
	"hebbian_plasticity_rate",
	"modulatory_release_factor",
	"modulatory_sensitivity",
	"polarity_factor",
	"connection_energy",
	"neuron_role"
]

const HISTOGRAM_PARAMETER_LABELS = [
	"Connection weight",
	"Incoming connections",
	"Outgoing connections",
	"Activation threshold",
	"Decay factor",
	"Retention factor",
	"Hebbian plasticity rate",
	"Modulatory release factor",
	"Modulatory sensitivity",
	"Polarity factor",
	"Connection energy",
	"Neuron role transition"
]

const HEATMAP_PARAMETER_KEYS = [
	"activation_threshold",
	"decay_factor",
	"retention_factor",
	"hebbian_plasticity_rate",
	"modulatory_release_factor",
	"modulatory_sensitivity",
	"polarity_factor",
	"connection_energy",
	"neuron_role",
	"incoming_connections"
]

const HEATMAP_PARAMETER_LABELS = [
	"Activation threshold",
	"Decay factor",
	"Retention factor",
	"Hebbian plasticity rate",
	"Modulatory release factor",
	"Modulatory sensitivity",
	"Polarity factor",
	"Connection energy",
	"Neuron role transition",
	"Incoming connections"
]

const ROLE_TRANSITION_TOOLTIP := (
	"Neuron role transition from Brain 1 to Brain 2. These numbers are categorical codes, not arithmetic differences:\n"
	+ "-3: Input -> Output\n"
	+ "-2: Input -> Hidden\n"
	+ "-1: Hidden -> Output\n"
	+ " 0: Same role\n"
	+ "+1: Hidden -> Input\n"
	+ "+2: Output -> Hidden\n"
	+ "+3: Output -> Input"
)


var brain_1: Dictionary = {}
var brain_2: Dictionary = {}
var config_1: Dictionary = {}
var config_2: Dictionary = {}
var generation_config_1: Dictionary = {}
var generation_config_2: Dictionary = {}

var load_brain_1_default_modulate := Color.WHITE
var load_brain_2_default_modulate := Color.WHITE

var summary_label: Label
var summary_error_label: Label
var load_dialog: FileDialog
var load_target := 1

var generation_thread: Thread
var generation_progress_mutex = Mutex.new()
var generation_progress := 0
var displayed_generation_progress := -1
var generation_start_time := 0

var histogram_data: Dictionary = {}

var heatmap_data: Dictionary = {}
var heatmap_ranges: Dictionary = {}

@onready var genetic_parameters_label: RichTextLabel = $Label
@onready var compute_diff_button: Button = $ComputeDiff
@onready var load_brain_1_button: Button = $LoadBrain1
@onready var load_brain_2_button: Button = $LoadBrain2

@onready var histogram_label: Label = $HistogramLabel
@onready var histogram: Histogram = $Histogram
@onready var histogram_slice_selector: OptionButton = $HistogramSliceSelector
@onready var histogram_parameter_selector: OptionButton = $HistogramParameterSelector

@onready var heatmap_label: Label = $HeatmapLabel
@onready var neuron_heatmap: Heatmap = $Heatmap
@onready var heatmap_slice_selector: OptionButton = $HeatmapSliceSelector
@onready var heatmap_parameter_selector: OptionButton = $HeatmapParameterSelector


func _ready() -> void:
	load_brain_1_default_modulate = load_brain_1_button.modulate
	load_brain_2_default_modulate = load_brain_2_button.modulate

	_setup_summary()
	_setup_load_dialog()
	_setup_genetic_parameters_label()
	_hide_analysis()

	load_brain_1_button.pressed.connect(_open_load_dialog.bind(1))
	load_brain_2_button.pressed.connect(_open_load_dialog.bind(2))
	compute_diff_button.pressed.connect(_compute_diff)

	histogram_slice_selector.item_selected.connect(_on_histogram_selection_changed)
	histogram_parameter_selector.item_selected.connect(_on_histogram_selection_changed)
	heatmap_slice_selector.item_selected.connect(_on_heatmap_selection_changed)
	heatmap_parameter_selector.item_selected.connect(_on_heatmap_selection_changed)

	_update_compute_button()


## Creates the general summary and the external-parameter error area.
func _setup_summary() -> void:
	var title = Label.new()
	title.position = Vector2(30, 10)
	title.add_theme_font_size_override("font_size", 22)
	title.text = "Brain Difference Summary"
	add_child(title)

	summary_label = Label.new()
	summary_label.position = Vector2(30, 45)
	summary_label.size = Vector2(800, 590)
	summary_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	summary_label.add_theme_font_size_override("font_size", 16)
	summary_label.text = "Load Brain 1 and Brain 2, then compute their difference."
	add_child(summary_label)

	summary_error_label = Label.new()
	summary_error_label.position = Vector2(30, 645)
	summary_error_label.size = Vector2(800, 130)
	summary_error_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	summary_error_label.add_theme_font_size_override("font_size", 16)
	summary_error_label.add_theme_color_override("font_color", Color(1.0, 0.35, 0.35))
	add_child(summary_error_label)


## Configures the existing RichTextLabel used to compare both genomes.
func _setup_genetic_parameters_label() -> void:
	genetic_parameters_label.bbcode_enabled = true
	genetic_parameters_label.scroll_active = true
	genetic_parameters_label.fit_content = false
	genetic_parameters_label.text = "[font_size=22][b]Genetic Parameters[/b][/font_size]\nLoad both brains to compare their genomes."

## Creates the native file chooser used by both load buttons.
func _setup_load_dialog() -> void:
	load_dialog = FileDialog.new()
	load_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	load_dialog.access = FileDialog.ACCESS_FILESYSTEM
	load_dialog.use_native_dialog = true
	load_dialog.filters = PackedStringArray(["*.json ; JSON files"])
	load_dialog.file_selected.connect(_load_brain_configuration)
	add_child(load_dialog)


## Opens the load dialog for Brain 1 or Brain 2.
## [param target] must be 1 or 2.
func _open_load_dialog(target: int) -> void:
	if _is_generating():
		return

	load_target = target
	load_dialog.popup_centered()


## Loads one brain configuration saved by BrainExplorer.
## [param path] is the JSON file selected by the user.
func _load_brain_configuration(path: String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)

	if file == null:
		summary_error_label.text = "Could not load brain configuration. Error: %s" % error_string(FileAccess.get_open_error())
		return

	var data = JSON.parse_string(file.get_as_text())

	_migrate_legacy_retention_factor_key(data)

	if not _is_valid_genome_file(data):
		summary_error_label.text = "Invalid brain configuration file:\n%s" % path
		return

	if load_target == 1:
		config_1 = data.duplicate(true)
		brain_1.clear()
		load_brain_1_button.tooltip_text = path
		load_brain_1_button.modulate = LOADED_BUTTON_MODULATE
	else:
		config_2 = data.duplicate(true)
		brain_2.clear()
		load_brain_2_button.tooltip_text = path
		load_brain_2_button.modulate = LOADED_BUTTON_MODULATE

	_hide_analysis()
	_update_genetic_parameters()
	_update_external_parameter_error()
	_update_loaded_summary()
	_update_compute_button()


## Consumes the two loaded configurations after Compute Diff is pressed.
## The worker keeps private copies while the load buttons return to their initial state.
func _consume_loaded_configurations() -> void:
	config_1.clear()
	config_2.clear()
	load_brain_1_button.tooltip_text = ""
	load_brain_2_button.tooltip_text = ""
	load_brain_1_button.modulate = load_brain_1_default_modulate
	load_brain_2_button.modulate = load_brain_2_default_modulate


## Restores the loaded configurations if the worker thread cannot be started.
func _restore_loaded_configurations() -> void:
	config_1 = generation_config_1.duplicate(true)
	config_2 = generation_config_2.duplicate(true)
	load_brain_1_button.modulate = LOADED_BUTTON_MODULATE
	load_brain_2_button.modulate = LOADED_BUTTON_MODULATE


## Converts the former saved JSON key "memory_factor" to the current "retention_factor" key.
## This keeps previously saved brain configurations loadable after the terminology change.
func _migrate_legacy_retention_factor_key(data: Variant) -> void:
	if not (data is Dictionary):
		return

	if not data.has("genome") or not (data["genome"] is Dictionary):
		return

	var genome_data: Dictionary = data["genome"]

	if not genome_data.has("retention_factor") and genome_data.has("memory_factor"):
		genome_data["retention_factor"] = genome_data["memory_factor"]
		genome_data.erase("memory_factor")


## Returns true when the parsed JSON contains every field required by BrainBuilder.
func _is_valid_genome_file(data: Variant) -> bool:
	if not (data is Dictionary):
		return false

	if not data.has("external_parameters") or not (data["external_parameters"] is Dictionary):
		return false

	if not data.has("genome") or not (data["genome"] is Dictionary):
		return false

	var external_parameters: Dictionary = data["external_parameters"]
	var genome_data: Dictionary = data["genome"]

	for parameter_name in EXTERNAL_PARAMETER_NAMES:
		if not external_parameters.has(parameter_name):
			return false

		var parameter_text = external_parameters[parameter_name]

		if not (parameter_text is String):
			return false

		if parameter_text.is_empty() or _filter_characters(parameter_text, "0123456789") != parameter_text:
			return false

	for gene_name in GENE_ORDER:
		if not genome_data.has(gene_name):
			return false

		var gene_text = genome_data[gene_name]

		if not (gene_text is String):
			return false

		if gene_text.length() != GENE_SPECS[gene_name]:
			return false

		if _filter_characters(gene_text, "0123") != gene_text:
			return false

	var size_x = int(external_parameters["size_x"])
	var size_y = int(external_parameters["size_y"])
	var size_z = int(external_parameters["size_z"])
	var num_inputs = int(external_parameters["num_inputs"])
	var num_outputs = int(external_parameters["num_outputs"])

	if size_x < 2 or size_y < 2 or size_z < 2:
		return false

	if num_inputs + num_outputs > size_x * size_y * size_z:
		return false

	return true


## Returns only characters contained in allowed_characters.
func _filter_characters(text: String, allowed_characters: String) -> String:
	var filtered = ""

	for character in text:
		if allowed_characters.contains(character):
			filtered += character

	return filtered


## Updates the pre-computation summary after one or both files are loaded.
func _update_loaded_summary() -> void:
	if config_1.is_empty() and config_2.is_empty():
		summary_label.text = "Load Brain 1 and Brain 2, then compute their difference."
		return

	if config_1.is_empty():
		summary_label.text = "Brain 2 loaded.\nLoad Brain 1 to continue."
		return

	if config_2.is_empty():
		summary_label.text = "Brain 1 loaded.\nLoad Brain 2 to continue."
		return

	if not _external_parameters_match():
		summary_label.text = "Both brains are loaded, but their external parameters are incompatible."
		return

	var external: Dictionary = config_1["external_parameters"]
	summary_label.text = (
		"Dimensions: %d x %d x %d\n" % [int(external["size_x"]), int(external["size_y"]), int(external["size_z"])]
		+ "Inputs: %d    Outputs: %d\n" % [int(external["num_inputs"]), int(external["num_outputs"])]
		+ "Ready to compute Brain 1 - Brain 2."
	)


## Shows an error whenever external parameters are not identical.
func _update_external_parameter_error() -> void:
	summary_error_label.text = ""

	if config_1.is_empty() or config_2.is_empty():
		return

	var external_1: Dictionary = config_1["external_parameters"]
	var external_2: Dictionary = config_2["external_parameters"]
	var differences: Array[String] = []

	for parameter_name in EXTERNAL_PARAMETER_NAMES:
		var value_1 := int(external_1[parameter_name])
		var value_2 := int(external_2[parameter_name])

		if value_1 != value_2:
			differences.append(
				"%s: %d x %d" % [EXTERNAL_PARAMETER_LABELS[parameter_name], value_1, value_2]
			)

	if not differences.is_empty():
		summary_error_label.text = (
			"ERROR: External parameters must be identical before a spatial difference can be computed.\n"
			+ "\n".join(differences)
		)


## Returns true when all external parameters are numerically identical.
func _external_parameters_match() -> bool:
	if config_1.is_empty() or config_2.is_empty():
		return false

	var external_1: Dictionary = config_1["external_parameters"]
	var external_2: Dictionary = config_2["external_parameters"]

	for parameter_name in EXTERNAL_PARAMETER_NAMES:
		if int(external_1[parameter_name]) != int(external_2[parameter_name]):
			return false

	return true


## Enables Compute Diff only when both compatible configurations are loaded.
func _update_compute_button() -> void:
	var done := not config_1.is_empty() and not config_2.is_empty() and _external_parameters_match()
	compute_diff_button.disabled = not done or _is_generating()
	load_brain_1_button.disabled = _is_generating()
	load_brain_2_button.disabled = _is_generating()


## Builds a Genome object from one saved configuration.
func _genome_from_config(config: Dictionary) -> Genome:
	var genome = Genome.new()
	var data: Dictionary = config["genome"]

	genome.max_connections = _gene_from_text(data["max_connections"])
	genome.connections = _gene_from_text(data["connections"])
	genome.activation_threshold = _gene_from_text(data["activation_threshold"])
	genome.refractory_strength = _gene_from_text(data["refractory_strength"])
	genome.exponential_factor = _gene_from_text(data["exponential_factor"])
	genome.decay_factor = _gene_from_text(data["decay_factor"])
	genome.retention_factor = _gene_from_text(data["retention_factor"])
	genome.polarity_factor = _gene_from_text(data["polarity_factor"])
	genome.hebbian_plasticity_rate = _gene_from_text(data["hebbian_plasticity_rate"])
	genome.modulatory_release_factor = _gene_from_text(data["modulatory_release_factor"])
	genome.modulatory_sensitivity = _gene_from_text(data["modulatory_sensitivity"])
	genome.modulatory_dynamics = _gene_from_text(data["modulatory_dynamics"])
	genome.structural_plasticity = _gene_from_text(data["structural_plasticity"])
	genome.input_influence = _gene_from_text(data["input_influence"])
	genome.beta = _gene_from_text(data["beta"])

	return genome


## Converts a gene string into the integer array expected by Genome.
func _gene_from_text(text: String) -> Array[int]:
	var gene: Array[int] = []

	for character in text:
		gene.append(int(character))

	return gene


## Starts generation of both brains in one worker thread.
func _compute_diff() -> void:
	if compute_diff_button.disabled or _is_generating():
		return

	if not _external_parameters_match():
		_update_external_parameter_error()
		return

	_hide_analysis()
	brain_1.clear()
	brain_2.clear()
	generation_config_1 = config_1.duplicate(true)
	generation_config_2 = config_2.duplicate(true)

	generation_progress_mutex.lock()
	generation_progress = 0
	generation_progress_mutex.unlock()
	displayed_generation_progress = -1
	generation_start_time = Time.get_ticks_msec()

	summary_label.text = "Generating both brains... 0%"
	generation_thread = Thread.new()
	var error = generation_thread.start(_build_brains_thread)

	if error != OK:
		generation_thread = null
		_restore_loaded_configurations()
		summary_error_label.text = "Could not start brain-generation thread. Error: %s" % error_string(error)
		_update_compute_button()
		return

	_consume_loaded_configurations()
	_update_compute_button()


## Worker-thread entry point. It builds both brains without touching scene nodes.
func _build_brains_thread() -> Dictionary:
	var external: Dictionary = generation_config_1["external_parameters"]
	var size := Vector3i(
		int(external["size_x"]),
		int(external["size_y"]),
		int(external["size_z"])
	)
	var num_inputs := int(external["num_inputs"])
	var num_outputs := int(external["num_outputs"])
	var builder_1 = BrainBuilder.new()
	var builder_2 = BrainBuilder.new()

	var start_1 := Time.get_ticks_msec()
	var result_1: Dictionary = builder_1.build(
		_genome_from_config(generation_config_1),
		size,
		num_inputs,
		num_outputs,
		_set_generation_progress_from_thread.bind(1)
	)
	var time_1 := (Time.get_ticks_msec() - start_1) / 1000.0

	if result_1.is_empty():
		return {"brain_1": {}, "brain_2": {}, "time_1": time_1, "time_2": 0.0}

	var start_2 := Time.get_ticks_msec()
	var result_2: Dictionary = builder_2.build(
		_genome_from_config(generation_config_2),
		size,
		num_inputs,
		num_outputs,
		_set_generation_progress_from_thread.bind(2)
	)
	var time_2 := (Time.get_ticks_msec() - start_2) / 1000.0

	return {"brain_1": result_1, "brain_2": result_2, "time_1": time_1, "time_2": time_2}


## Maps each BrainBuilder progress value to the combined 0 to 100 range.
func _set_generation_progress_from_thread(percent: int, brain_index: int) -> void:
	var combined := int(percent * 0.5) if brain_index == 1 else 50 + int(percent * 0.5)
	generation_progress_mutex.lock()
	generation_progress = combined
	generation_progress_mutex.unlock()


## Keeps the UI responsive while both brains are generated.
func _process(_delta: float) -> void:
	if not _is_generating():
		return

	generation_progress_mutex.lock()
	var progress := generation_progress
	generation_progress_mutex.unlock()

	if progress != displayed_generation_progress:
		displayed_generation_progress = progress
		summary_label.text = "Generating both brains... %d%%" % progress

	if generation_thread.is_alive():
		return

	var result: Dictionary = generation_thread.wait_to_finish()
	generation_thread = null
	brain_1 = result["brain_1"]
	brain_2 = result["brain_2"]

	if brain_1.is_empty() or brain_2.is_empty():
		var failed_brain := "Brain 1" if brain_1.is_empty() else "Brain 2"
		summary_label.text = "Brain difference generation failed."
		summary_error_label.text = "%s could not be generated. Isolated neurons were found." % failed_brain
		generation_config_1.clear()
		generation_config_2.clear()
		_update_compute_button()
		return

	_update_summary(brain_1, brain_2, result["time_1"], result["time_2"])
	_setup_histogram(brain_1, brain_2)
	_setup_heatmap(brain_1, brain_2)

	summary_error_label.text = ""
	generation_config_1.clear()
	generation_config_2.clear()
	_update_compute_button()


## Returns true while the worker thread is still owned by this scene.
func _is_generating() -> bool:
	return generation_thread != null and generation_thread.is_started()


## Displays paired structural statistics as Brain 1 x Brain 2.
func _update_summary(first_brain: Dictionary, second_brain: Dictionary, time_1: float, time_2: float) -> void:
	var stats_1 := _brain_statistics(first_brain)
	var stats_2 := _brain_statistics(second_brain)
	var overlap := _connection_overlap(first_brain, second_brain)
	var external: Dictionary = generation_config_1["external_parameters"]

	summary_label.text = (
		"Dimensions: %d x %d x %d\n" % [int(external["size_x"]), int(external["size_y"]), int(external["size_z"])]
		+ "Inputs: %d    Outputs: %d\n" % [int(external["num_inputs"]), int(external["num_outputs"])]
		+ "Beta: %.6f vs %.6f\n" % [first_brain["beta"], second_brain["beta"]]
		+ "Maximum connections per neuron: %d vs %d\n" % [first_brain["max_connections"], second_brain["max_connections"]]
		+ "Existing directional connections: %d vs %d\n" % [stats_1["total_connections"], stats_2["total_connections"]]
		+ "Shared directional connections: %d / %d (%.2f%%)\n" % [
			overlap["shared"], overlap["union"], overlap["shared_percent"]
		]
		+ "Mean outgoing connections: %.3f vs %.3f\n" % [stats_1["mean_outgoing"], stats_2["mean_outgoing"]]
		+ "Outgoing degree range: %d-%d vs %d-%d\n" % [
			stats_1["min_out_degree"], stats_1["max_out_degree"],
			stats_2["min_out_degree"], stats_2["max_out_degree"]
		]
		+ "Excitatory neurons: %d (%.2f%%) vs %d (%.2f%%)\n" % [
			stats_1["excitatory"], stats_1["excitatory_percent"],
			stats_2["excitatory"], stats_2["excitatory_percent"]
		]
		+ "Inhibitory neurons: %d (%.2f%%) vs %d (%.2f%%)\n\n" % [
			stats_1["inhibitory"], stats_1["inhibitory_percent"],
			stats_2["inhibitory"], stats_2["inhibitory_percent"]
		]
		+ "Mean activation threshold: %.6f vs %.6f\n" % [stats_1["mean_activation"], stats_2["mean_activation"]]
		+ "Mean decay factor: %.6f vs %.6f\n" % [stats_1["mean_decay"], stats_2["mean_decay"]]
		+ "Mean retention factor: %.6f vs %.6f\n" % [stats_1["mean_retention"], stats_2["mean_retention"]]
		+ "Mean Hebbian plasticity rate: %.6f vs %.6f\n" % [stats_1["mean_hebbian"], stats_2["mean_hebbian"]]
		+ "Mean modulatory release factor: %.6f vs %.6f\n" % [stats_1["mean_modulatory_release"], stats_2["mean_modulatory_release"]]
		+ "Mean modulatory sensitivity: %.6f vs %.6f\n" % [stats_1["mean_modulatory_sensitivity"], stats_2["mean_modulatory_sensitivity"]]
		+ "Mean connection weight: %.6f vs %.6f\n\n" % [stats_1["mean_weight"], stats_2["mean_weight"]]
		+ "Refractory strength: %.6f vs %.6f\n" % [first_brain["refractory_strength"], second_brain["refractory_strength"]]
		+ "Exponential factor: %.6f vs %.6f\n" % [first_brain["exponential_factor"], second_brain["exponential_factor"]]
		+ "Modulatory persistence lambda: %.6f vs %.6f\n" % [first_brain["lambda"], second_brain["lambda"]]
		+ "Modulatory spread nu: %.6f vs %.6f\n" % [first_brain["nu"], second_brain["nu"]]
		+ "Trace persistence mu: %.6f vs %.6f\n" % [first_brain["mu"], second_brain["mu"]]
		+ "Input influence psi: %.6f vs %.6f\n" % [first_brain["input_gain"], second_brain["input_gain"]]
		+ "Structural plasticity: %.6f vs %.6f\n" % [first_brain["structural_plasticity"], second_brain["structural_plasticity"]]
		+ "Build time: %.3f s vs %.3f s" % [time_1, time_2]
	)


## Computes the overlap of directional connections between both brains.
func _connection_overlap(first_brain: Dictionary, second_brain: Dictionary) -> Dictionary:
	var neurons_1: Array = first_brain["neurons"]
	var neurons_2: Array = second_brain["neurons"]
	var shared := 0
	var total_1 := 0
	var total_2 := 0

	for neuron_index in neurons_1.size():
		var connections_1: Array = neurons_1[neuron_index]["connections"]
		var connections_2: Array = neurons_2[neuron_index]["connections"]
		total_1 += connections_1.size()
		total_2 += connections_2.size()

		var targets_1 := {}
		for connection in connections_1:
			targets_1[connection["target"]] = true

		for connection in connections_2:
			if targets_1.has(connection["target"]):
				shared += 1

	var union := total_1 + total_2 - shared
	var shared_percent := 100.0 if union == 0 else 100.0 * shared / union

	return {
		"shared": shared,
		"union": union,
		"shared_percent": shared_percent
	}


## Computes the structural statistics used in the paired summary.
func _brain_statistics(brain: Dictionary) -> Dictionary:
	var neurons: Array = brain["neurons"]
	var total_neurons := neurons.size()
	var total_connections := 0
	var excitatory := 0
	var inhibitory := 0
	var threshold_sum := 0.0
	var decay_sum := 0.0
	var retention_sum := 0.0
	var hebbian_sum := 0.0
	var modulatory_release_sum := 0.0
	var modulatory_sensitivity_sum := 0.0
	var weight_sum := 0.0
	var min_out_degree := 0
	var max_out_degree := 0
	var outgoing_neuron_count := 0
	var outgoing_connection_count := 0
	var first_outgoing_neuron := true

	for neuron in neurons:
		var connections: Array = neuron["connections"]
		total_connections += connections.size()
		threshold_sum += neuron["activation_threshold"]
		decay_sum += neuron["decay_factor"]
		retention_sum += neuron["retention_factor"]
		hebbian_sum += neuron["hebbian_plasticity_rate"]
		modulatory_release_sum += neuron["modulatory_release_factor"]
		modulatory_sensitivity_sum += neuron["modulatory_sensitivity"]

		if neuron["polarity_factor"] > 0:
			excitatory += 1
		else:
			inhibitory += 1

		for connection in connections:
			weight_sum += connection["weight"]

		if not neuron["output"]:
			var out_degree := connections.size()
			outgoing_neuron_count += 1
			outgoing_connection_count += out_degree

			if first_outgoing_neuron:
				min_out_degree = out_degree
				max_out_degree = out_degree
				first_outgoing_neuron = false
			else:
				min_out_degree = mini(min_out_degree, out_degree)
				max_out_degree = maxi(max_out_degree, out_degree)

	var neuron_denominator = max(1, total_neurons)
	var mean_outgoing := 0.0 if outgoing_neuron_count == 0 else float(outgoing_connection_count) / outgoing_neuron_count
	var mean_weight := 0.0 if total_connections == 0 else weight_sum / total_connections

	return {
		"total_neurons": total_neurons,
		"total_connections": total_connections,
		"mean_outgoing": mean_outgoing,
		"min_out_degree": min_out_degree,
		"max_out_degree": max_out_degree,
		"excitatory": excitatory,
		"inhibitory": inhibitory,
		"excitatory_percent": 100.0 * excitatory / neuron_denominator,
		"inhibitory_percent": 100.0 * inhibitory / neuron_denominator,
		"mean_activation": threshold_sum / neuron_denominator,
		"mean_decay": decay_sum / neuron_denominator,
		"mean_retention": retention_sum / neuron_denominator,
		"mean_hebbian": hebbian_sum / neuron_denominator,
		"mean_modulatory_release": modulatory_release_sum / neuron_denominator,
		"mean_modulatory_sensitivity": modulatory_sensitivity_sum / neuron_denominator,
		"mean_weight": mean_weight
	}


## Writes both genes and highlights every digit that differs.
func _update_genetic_parameters() -> void:
	if config_1.is_empty() or config_2.is_empty():
		genetic_parameters_label.text = "[font_size=22][b]Genetic Parameters[/b][/font_size]\nLoad both brains to compare their genomes."
		return

	var genome_1: Dictionary = config_1["genome"]
	var genome_2: Dictionary = config_2["genome"]
	var total_digits := 0
	var total_differences := 0

	for gene_name in GENE_ORDER:
		var gene_1: String = genome_1[gene_name]
		var gene_2: String = genome_2[gene_name]
		total_digits += gene_1.length()
		total_differences += _count_gene_differences(gene_1, gene_2)

	var text := (
		"[font_size=22][b]Genetic Parameters[/b][/font_size]\n"
		+ "Different digits: [b]%d / %d[/b]\n" % [total_differences, total_digits]
		+ "[font_size=13][color=#8f8f8f]Equal digits[/color]  "
		+ "[color=#ffb86c]Brain 1 difference[/color]  "
		+ "[color=#8be9fd]Brain 2 difference[/color][/font_size]\n\n"
	)

	for gene_name in GENE_ORDER:
		var gene_1: String = genome_1[gene_name]
		var gene_2: String = genome_2[gene_name]
		var mismatch_count := _count_gene_differences(gene_1, gene_2)
		var mismatch_text := "[color=#8f8f8f]identical[/color]" if mismatch_count == 0 else "[color=#ff6b6b]%d changed digits[/color]" % mismatch_count

		text += "[b]%s[/b]  %s\n" % [GENE_LABELS[gene_name], mismatch_text]
		text += "B1  %s\n" % _colored_gene(gene_1, gene_2, true)
		text += "B2  %s\n\n" % _colored_gene(gene_2, gene_1, false)

	genetic_parameters_label.text = text


## Counts how many positions differ between two equal-length genes.
func _count_gene_differences(first_gene: String, second_gene: String) -> int:
	var count := 0

	for index in first_gene.length():
		if first_gene[index] != second_gene[index]:
			count += 1

	return count


## Returns one gene as BBCode, emphasizing positions that differ.
func _colored_gene(gene: String, other_gene: String, brain_1_gene: bool) -> String:
	var text := ""
	var difference_color := "#ffb86c" if brain_1_gene else "#8be9fd"
	var difference_background := "#4a2c12" if brain_1_gene else "#12394a"

	for index in gene.length():
		var digit := gene[index]

		if digit == other_gene[index]:
			text += "[color=#8f8f8f]%s[/color]" % digit
		else:
			text += "[bgcolor=%s][color=%s][b]%s[/b][/color][/bgcolor]" % [difference_background, difference_color, digit]

		if (index + 1) % 8 == 0 and index + 1 < gene.length():
			text += " "

	return text


## Returns neurons indexed by their spatial position.
func _neurons_by_position(brain: Dictionary) -> Dictionary:
	var result := {}

	for neuron in brain["neurons"]:
		result[neuron["position"]] = neuron

	return result


## Returns incoming degree for every spatial neuron index.
func _incoming_counts(brain: Dictionary) -> PackedInt32Array:
	var I: int = brain["size"].x
	var J: int = brain["size"].y
	var K: int = brain["size"].z
	var IJ := I * J
	var counts := PackedInt32Array()
	counts.resize(IJ * K)

	for neuron in brain["neurons"]:
		for connection in neuron["connections"]:
			var target: Vector3i = connection["target"]
			counts[target.x + I * target.y + IJ * target.z] += 1

	return counts


## Converts one neuron's connections to target -> total weight.
func _connection_weight_map(neuron: Dictionary) -> Dictionary:
	var weights := {}

	for connection in neuron["connections"]:
		var target: Vector3i = connection["target"]
		weights[target] = weights.get(target, 0.0) + float(connection["weight"])

	return weights


## Returns the histogram bin for a value in a symmetric difference range.
func _difference_bin(value: float, max_abs: float) -> int:
	if max_abs <= 0.0:
		return int(BINS / 2.0)

	var normalized := (value + max_abs) / (2.0 * max_abs)
	return clampi(int(floor(normalized * BINS)), 0, BINS - 1)


## Finds the largest absolute connection-weight difference over the union of edges.
func _find_weight_max_abs_difference(first_brain: Dictionary, second_brain: Dictionary) -> float:
	var neurons_2 := _neurons_by_position(second_brain)
	var max_abs := 0.0

	for neuron_1 in first_brain["neurons"]:
		var neuron_2: Dictionary = neurons_2[neuron_1["position"]]
		var weights_1 := _connection_weight_map(neuron_1)
		var weights_2 := _connection_weight_map(neuron_2)

		for target in weights_1:
			var difference := float(weights_1[target]) - float(weights_2.get(target, 0.0))
			max_abs = maxf(max_abs, absf(difference))

		for target in weights_2:
			if weights_1.has(target):
				continue

			max_abs = maxf(max_abs, absf(float(weights_2[target])))

	return max_abs


## Creates empty whole-brain and per-slice bins for one difference parameter.
func _initialize_histogram_parameter(parameter_key: String, max_abs: float, K: int, integer_range: bool) -> void:
	var limit := max_abs if max_abs > 0.0 else 1.0
	var slices: Array = []
	slices.resize(K)

	for k in K:
		var bins := PackedInt32Array()
		bins.resize(BINS)
		slices[k] = bins

	var all_bins := PackedInt32Array()
	all_bins.resize(BINS)

	histogram_data[parameter_key] = {
		"range": Vector2(-limit, limit),
		"integer_range": integer_range,
		"slices": slices,
		"all": all_bins
	}


## Adds one Brain 1 - Brain 2 value to the whole-brain and corresponding slice histogram.
func _add_histogram_difference(parameter_key: String, value: float, k: int) -> void:
	var parameter_data: Dictionary = histogram_data[parameter_key]
	var value_range: Vector2 = parameter_data["range"]
	var max_abs := maxf(absf(value_range.x), absf(value_range.y))
	var bin := _difference_bin(value, max_abs)

	parameter_data["slices"][k][bin] += 1
	parameter_data["all"][bin] += 1


## Builds all Brain 1 - Brain 2 histograms using one common parameter selector.
func _build_histogram_data(first_brain: Dictionary, second_brain: Dictionary) -> void:
	var I: int = first_brain["size"].x
	var J: int = first_brain["size"].y
	var K: int = first_brain["size"].z
	var IJ := I * J
	var neurons_2 := _neurons_by_position(second_brain)
	var incoming_1 := _incoming_counts(first_brain)
	var incoming_2 := _incoming_counts(second_brain)

	var max_abs_values := {
		"connection_weight": _find_weight_max_abs_difference(first_brain, second_brain),
		"incoming_connections": 0.0,
		"outgoing_connections": 0.0,
		"activation_threshold": 0.0,
		"decay_factor": 0.0,
		"retention_factor": 0.0,
		"hebbian_plasticity_rate": 0.0,
		"modulatory_release_factor": 0.0,
		"modulatory_sensitivity": 0.0,
		"polarity_factor": 0.0,
		"connection_energy": 0.0,
		"neuron_role": 3.0
	}

	for neuron_1 in first_brain["neurons"]:
		var pos: Vector3i = neuron_1["position"]
		var neuron_2: Dictionary = neurons_2[pos]
		var spatial_index := pos.x + I * pos.y + IJ * pos.z

		max_abs_values["incoming_connections"] = maxf(
			float(max_abs_values["incoming_connections"]),
			absf(float(incoming_1[spatial_index] - incoming_2[spatial_index]))
		)
		max_abs_values["outgoing_connections"] = maxf(
			float(max_abs_values["outgoing_connections"]),
			absf(float(neuron_1["connections"].size() - neuron_2["connections"].size()))
		)
		max_abs_values["activation_threshold"] = maxf(
			float(max_abs_values["activation_threshold"]),
			absf(float(neuron_1["activation_threshold"]) - float(neuron_2["activation_threshold"]))
		)
		max_abs_values["decay_factor"] = maxf(
			float(max_abs_values["decay_factor"]),
			absf(float(neuron_1["decay_factor"]) - float(neuron_2["decay_factor"]))
		)
		max_abs_values["retention_factor"] = maxf(
			float(max_abs_values["retention_factor"]),
			absf(float(neuron_1["retention_factor"]) - float(neuron_2["retention_factor"]))
		)
		max_abs_values["hebbian_plasticity_rate"] = maxf(
			float(max_abs_values["hebbian_plasticity_rate"]),
			absf(float(neuron_1["hebbian_plasticity_rate"]) - float(neuron_2["hebbian_plasticity_rate"]))
		)
		max_abs_values["modulatory_release_factor"] = maxf(
			float(max_abs_values["modulatory_release_factor"]),
			absf(float(neuron_1["modulatory_release_factor"]) - float(neuron_2["modulatory_release_factor"]))
		)
		max_abs_values["modulatory_sensitivity"] = maxf(
			float(max_abs_values["modulatory_sensitivity"]),
			absf(float(neuron_1["modulatory_sensitivity"]) - float(neuron_2["modulatory_sensitivity"]))
		)
		max_abs_values["polarity_factor"] = maxf(
			float(max_abs_values["polarity_factor"]),
			absf(float(neuron_1["polarity_factor"]) - float(neuron_2["polarity_factor"]))
		)
		max_abs_values["connection_energy"] = maxf(
			float(max_abs_values["connection_energy"]),
			absf(float(neuron_1["connections_value"]) - float(neuron_2["connections_value"]))
		)

	histogram_data.clear()

	for parameter_key in HISTOGRAM_PARAMETER_KEYS:
		_initialize_histogram_parameter(
			parameter_key,
			float(max_abs_values[parameter_key]),
			K,
			parameter_key in ["incoming_connections", "outgoing_connections", "neuron_role"]
		)

	for neuron_1 in first_brain["neurons"]:
		var pos: Vector3i = neuron_1["position"]
		var k: int = pos.z
		var neuron_2: Dictionary = neurons_2[pos]
		var spatial_index := pos.x + I * pos.y + IJ * pos.z

		_add_histogram_difference("incoming_connections", incoming_1[spatial_index] - incoming_2[spatial_index], k)
		_add_histogram_difference("outgoing_connections", neuron_1["connections"].size() - neuron_2["connections"].size(), k)
		_add_histogram_difference("activation_threshold", float(neuron_1["activation_threshold"]) - float(neuron_2["activation_threshold"]), k)
		_add_histogram_difference("decay_factor", float(neuron_1["decay_factor"]) - float(neuron_2["decay_factor"]), k)
		_add_histogram_difference("retention_factor", float(neuron_1["retention_factor"]) - float(neuron_2["retention_factor"]), k)
		_add_histogram_difference("hebbian_plasticity_rate", float(neuron_1["hebbian_plasticity_rate"]) - float(neuron_2["hebbian_plasticity_rate"]), k)
		_add_histogram_difference("modulatory_release_factor", float(neuron_1["modulatory_release_factor"]) - float(neuron_2["modulatory_release_factor"]), k)
		_add_histogram_difference("modulatory_sensitivity", float(neuron_1["modulatory_sensitivity"]) - float(neuron_2["modulatory_sensitivity"]), k)
		_add_histogram_difference("polarity_factor", float(neuron_1["polarity_factor"]) - float(neuron_2["polarity_factor"]), k)
		_add_histogram_difference("connection_energy", float(neuron_1["connections_value"]) - float(neuron_2["connections_value"]), k)
		_add_histogram_difference("neuron_role", _role_transition_class(_neuron_role(neuron_1), _neuron_role(neuron_2)), k)

		var weights_1 := _connection_weight_map(neuron_1)
		var weights_2 := _connection_weight_map(neuron_2)

		for target in weights_1:
			_add_histogram_difference(
				"connection_weight",
				float(weights_1[target]) - float(weights_2.get(target, 0.0)),
				k
			)

		for target in weights_2:
			if weights_1.has(target):
				continue

			_add_histogram_difference("connection_weight", -float(weights_2[target]), k)


## Prepares the shared difference histogram and its slice and parameter selectors.
func _setup_histogram(first_brain: Dictionary, second_brain: Dictionary) -> void:
	_build_histogram_data(first_brain, second_brain)
	_setup_slice_selector(histogram_slice_selector, first_brain["size"].z, true)

	histogram_parameter_selector.clear()
	for label in HISTOGRAM_PARAMETER_LABELS:
		histogram_parameter_selector.add_item(label)

	histogram_parameter_selector.select(0)
	_update_histogram()

	histogram_label.show()
	histogram_slice_selector.show()
	histogram_parameter_selector.show()
	histogram.show()


## Updates the histogram using the selected slice and difference parameter.
func _update_histogram() -> void:
	if brain_1.is_empty() or brain_2.is_empty():
		return

	var parameter_index: int = histogram_parameter_selector.selected
	var slice_index: int = histogram_slice_selector.selected

	if parameter_index < 0 or slice_index < 0:
		return

	var parameter_key: String = HISTOGRAM_PARAMETER_KEYS[parameter_index]
	var parameter_data: Dictionary = histogram_data[parameter_key]
	var value_range: Vector2 = parameter_data["range"]

	histogram_label.text = "%s Distribution" % HISTOGRAM_PARAMETER_LABELS[parameter_index]
	histogram.set_x_range(value_range.x, value_range.y, parameter_data["integer_range"])

	if slice_index == 0:
		histogram.set_bins(parameter_data["all"])
	else:
		histogram.set_bins(parameter_data["slices"][slice_index - 1])


## Fills one slice selector with an optional all-slices entry.
func _setup_slice_selector(selector: OptionButton, K: int, include_all: bool) -> void:
	selector.clear()

	if include_all:
		selector.add_item("All slices")

	for k in K:
		selector.add_item("k = %d" % k)

	selector.select(0)


## Builds all spatial Brain 1 - Brain 2 scalar fields used by the heatmap.
func _build_heatmap_data(first_brain: Dictionary, second_brain: Dictionary) -> void:
	var I: int = first_brain["size"].x
	var J: int = first_brain["size"].y
	var K: int = first_brain["size"].z
	var IJ := I * J
	var total_neurons := IJ * K
	var neurons_2 := _neurons_by_position(second_brain)
	var incoming_1 := _incoming_counts(first_brain)
	var incoming_2 := _incoming_counts(second_brain)

	var activation_values := PackedFloat64Array()
	var decay_values := PackedFloat64Array()
	var retention_values := PackedFloat64Array()
	var hebbian_values := PackedFloat64Array()
	var modulatory_release_values := PackedFloat64Array()
	var modulatory_sensitivity_values := PackedFloat64Array()
	var polarity_values := PackedFloat64Array()
	var connection_energy_values := PackedFloat64Array()
	var role_values := PackedFloat64Array()
	var incoming_values := PackedFloat64Array()

	activation_values.resize(total_neurons)
	decay_values.resize(total_neurons)
	retention_values.resize(total_neurons)
	hebbian_values.resize(total_neurons)
	modulatory_release_values.resize(total_neurons)
	modulatory_sensitivity_values.resize(total_neurons)
	polarity_values.resize(total_neurons)
	connection_energy_values.resize(total_neurons)
	role_values.resize(total_neurons)
	incoming_values.resize(total_neurons)

	var max_abs_activation := 0.0
	var max_abs_decay := 0.0
	var max_abs_retention := 0.0
	var max_abs_hebbian := 0.0
	var max_abs_modulatory_release := 0.0
	var max_abs_modulatory_sensitivity := 0.0
	var max_abs_polarity := 0.0
	var max_abs_connection_energy := 0.0
	var max_abs_incoming := 0.0

	for neuron_1 in first_brain["neurons"]:
		var pos: Vector3i = neuron_1["position"]
		var neuron_2: Dictionary = neurons_2[pos]
		var spatial_index := pos.x + I * pos.y + IJ * pos.z

		var activation := float(neuron_1["activation_threshold"]) - float(neuron_2["activation_threshold"])
		var decay := float(neuron_1["decay_factor"]) - float(neuron_2["decay_factor"])
		var retention := float(neuron_1["retention_factor"]) - float(neuron_2["retention_factor"])
		var hebbian := float(neuron_1["hebbian_plasticity_rate"]) - float(neuron_2["hebbian_plasticity_rate"])
		var modulatory_release := float(neuron_1["modulatory_release_factor"]) - float(neuron_2["modulatory_release_factor"])
		var modulatory_sensitivity := float(neuron_1["modulatory_sensitivity"]) - float(neuron_2["modulatory_sensitivity"])
		var polarity := float(neuron_1["polarity_factor"]) - float(neuron_2["polarity_factor"])
		var connection_energy := float(neuron_1["connections_value"]) - float(neuron_2["connections_value"])
		var incoming := float(incoming_1[spatial_index] - incoming_2[spatial_index])
		var role_transition := _role_transition_class(_neuron_role(neuron_1), _neuron_role(neuron_2))

		activation_values[spatial_index] = activation
		decay_values[spatial_index] = decay
		retention_values[spatial_index] = retention
		hebbian_values[spatial_index] = hebbian
		modulatory_release_values[spatial_index] = modulatory_release
		modulatory_sensitivity_values[spatial_index] = modulatory_sensitivity
		polarity_values[spatial_index] = polarity
		connection_energy_values[spatial_index] = connection_energy
		role_values[spatial_index] = role_transition
		incoming_values[spatial_index] = incoming

		max_abs_activation = maxf(max_abs_activation, absf(activation))
		max_abs_decay = maxf(max_abs_decay, absf(decay))
		max_abs_retention = maxf(max_abs_retention, absf(retention))
		max_abs_hebbian = maxf(max_abs_hebbian, absf(hebbian))
		max_abs_modulatory_release = maxf(max_abs_modulatory_release, absf(modulatory_release))
		max_abs_modulatory_sensitivity = maxf(max_abs_modulatory_sensitivity, absf(modulatory_sensitivity))
		max_abs_polarity = maxf(max_abs_polarity, absf(polarity))
		max_abs_connection_energy = maxf(max_abs_connection_energy, absf(connection_energy))
		max_abs_incoming = maxf(max_abs_incoming, absf(incoming))

	heatmap_data = {
		"activation_threshold": activation_values,
		"decay_factor": decay_values,
		"retention_factor": retention_values,
		"hebbian_plasticity_rate": hebbian_values,
		"modulatory_release_factor": modulatory_release_values,
		"modulatory_sensitivity": modulatory_sensitivity_values,
		"polarity_factor": polarity_values,
		"connection_energy": connection_energy_values,
		"neuron_role": role_values,
		"incoming_connections": incoming_values
	}

	heatmap_ranges = {
		"activation_threshold": _symmetric_range(max_abs_activation),
		"decay_factor": _symmetric_range(max_abs_decay),
		"retention_factor": _symmetric_range(max_abs_retention),
		"hebbian_plasticity_rate": _symmetric_range(max_abs_hebbian),
		"modulatory_release_factor": _symmetric_range(max_abs_modulatory_release),
		"modulatory_sensitivity": _symmetric_range(max_abs_modulatory_sensitivity),
		"polarity_factor": _symmetric_range(max_abs_polarity),
		"connection_energy": _symmetric_range(max_abs_connection_energy),
		"neuron_role": Vector2(-3.0, 3.0),
		"incoming_connections": _symmetric_range(max_abs_incoming)
	}


## Returns a symmetric range around zero, preserving a useful range when all values are zero.
func _symmetric_range(max_abs: float) -> Vector2:
	var limit := max_abs if max_abs > 0.0 else 1.0
	return Vector2(-limit, limit)

## Returns -1 for input, 0 for hidden, and 1 for output neurons.
func _neuron_role(neuron: Dictionary) -> int:
	if neuron["input"]:
		return -1

	if neuron["output"]:
		return 1

	return 0

## Encodes categorical Brain 1 -> Brain 2 role changes in seven discrete classes.
func _role_transition_class(role_1: int, role_2: int) -> int:
	if role_1 == role_2:
		return 0

	if role_1 == -1 and role_2 == 1:
		return -3
	if role_1 == -1 and role_2 == 0:
		return -2
	if role_1 == 0 and role_2 == 1:
		return -1
	if role_1 == 0 and role_2 == -1:
		return 1
	if role_1 == 1 and role_2 == 0:
		return 2

	return 3

## Prepares the difference heatmap and its selectors.
func _setup_heatmap(first_brain: Dictionary, second_brain: Dictionary) -> void:
	_build_heatmap_data(first_brain, second_brain)
	_setup_slice_selector(heatmap_slice_selector, first_brain["size"].z, false)

	heatmap_parameter_selector.clear()
	for label in HEATMAP_PARAMETER_LABELS:
		heatmap_parameter_selector.add_item(label)

	heatmap_parameter_selector.select(0)
	_update_heatmap()
	heatmap_label.show()
	heatmap_slice_selector.show()
	heatmap_parameter_selector.show()
	neuron_heatmap.show()

## Updates the heatmap using the selected slice and difference parameter.
func _update_heatmap() -> void:
	if brain_1.is_empty() or brain_2.is_empty():
		return

	var parameter_index: int = heatmap_parameter_selector.selected
	var k: int = heatmap_slice_selector.selected

	if parameter_index < 0 or k < 0:
		return

	var parameter_key: String = HEATMAP_PARAMETER_KEYS[parameter_index]
	var I: int = brain_1["size"].x
	var J: int = brain_1["size"].y
	var slice_size := I * J
	var slice_start := k * slice_size
	var all_values: PackedFloat64Array = heatmap_data[parameter_key]
	var slice_values: PackedFloat64Array = all_values.slice(slice_start, slice_start + slice_size)
	var value_range: Vector2 = heatmap_ranges[parameter_key]

	heatmap_label.text = "%s Heatmap" % HEATMAP_PARAMETER_LABELS[parameter_index]
	neuron_heatmap.set_data(slice_values, Vector2i(I, J), value_range.x, value_range.y)

## Hides all plots until a valid difference has been computed.
func _hide_analysis() -> void:
	histogram_label.hide()
	histogram_slice_selector.hide()
	histogram_parameter_selector.hide()
	histogram.hide()
	heatmap_label.hide()
	heatmap_slice_selector.hide()
	heatmap_parameter_selector.hide()
	neuron_heatmap.hide()


## Updates the histogram when either selector changes.
func _on_histogram_selection_changed(_index: int) -> void:
	_update_histogram()


## Updates the heatmap when either selector changes.
func _on_heatmap_selection_changed(_index: int) -> void:
	_update_heatmap()
