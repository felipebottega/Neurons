extends Node2D


const GENE_SPECS = {
	"max_connections": 8,
	"beta": 8,
	"connections": 64,
	"initial_search_region": 64,
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
	"initial_search_region",
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
	"initial_search_region": "Initial search region",
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
	"connections_density",
	"num_inputs",
	"num_outputs"
]

const EXTERNAL_PARAMETER_LABELS = {
	"size_x": "Size X",
	"size_y": "Size Y",
	"size_z": "Size Z",
	"connections_density": "Connections density",
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
	"Neuron role"
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
	"Neuron role",
	"Incoming connections"
]


var config_1: Dictionary = {}
var config_2: Dictionary = {}
var current_brain: Dictionary = {}
var current_offspring_genome: Dictionary = {}

var load_brain_1_default_modulate := Color.WHITE
var load_brain_2_default_modulate := Color.WHITE

var summary_label: Label
var load_dialog: FileDialog
var save_genome_dialog: FileDialog
var load_target := 1

var generation_thread: Thread
var generation_progress_mutex = Mutex.new()
var generation_progress := 0
var displayed_generation_progress := -1
var displayed_generation_stage := ""
var preparing_brain := false
var generation_start_time := 0
var generation_mutation_probability := 0.0

var histogram_data: Dictionary = {}
var heatmap_data: Dictionary = {}
var heatmap_ranges: Dictionary = {}

@onready var genetic_parameters_label: RichTextLabel = $Label
@onready var generate_offspring_button: Button = $GenerateOffspring
@onready var load_brain_1_button: Button = $LoadBrain1
@onready var load_brain_2_button: Button = $LoadBrain2
@onready var mutation_probability_field: LineEdit = $LineEdit
@onready var save_brain_button: Button = $SaveBrain

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
	_setup_save_genome_dialog()
	_setup_genetic_parameters_label()
	_setup_mutation_probability_field()
	_hide_analysis()

	load_brain_1_button.pressed.connect(_open_load_dialog.bind(1))
	load_brain_2_button.pressed.connect(_open_load_dialog.bind(2))
	generate_offspring_button.pressed.connect(_generate_offspring)
	save_brain_button.pressed.connect(_open_save_genome_dialog)
	mutation_probability_field.text_changed.connect(_on_mutation_probability_changed)

	histogram_slice_selector.item_selected.connect(_on_histogram_selection_changed)
	histogram_parameter_selector.item_selected.connect(_on_histogram_selection_changed)
	heatmap_slice_selector.item_selected.connect(_on_heatmap_selection_changed)
	heatmap_parameter_selector.item_selected.connect(_on_heatmap_selection_changed)

	_update_generate_button()


## Creates the offspring summary area on the left side of the screen.
func _setup_summary() -> void:
	var title = Label.new()
	title.position = Vector2(30, 10)
	title.add_theme_font_size_override("font_size", 22)
	title.text = "Offspring Summary"
	add_child(title)

	summary_label = Label.new()
	summary_label.position = Vector2(30, 45)
	summary_label.size = Vector2(800, 500)
	summary_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	summary_label.add_theme_font_size_override("font_size", 14)
	summary_label.text = "Load Brain 1 and Brain 2 to generate an offspring."
	add_child(summary_label)


## Configures the existing RichTextLabel used to compare both parent genomes.
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


## Creates the native file chooser used to save the generated offspring genome.
func _setup_save_genome_dialog() -> void:
	save_genome_dialog = FileDialog.new()
	save_genome_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	save_genome_dialog.access = FileDialog.ACCESS_FILESYSTEM
	save_genome_dialog.use_native_dialog = true
	save_genome_dialog.filters = PackedStringArray(["*.json ; JSON files"])
	save_genome_dialog.current_file = "brain.json"
	save_genome_dialog.file_selected.connect(_save_genome_to_file)
	add_child(save_genome_dialog)


## Configures the mutation probability field.
func _setup_mutation_probability_field() -> void:
	if mutation_probability_field.text.strip_edges().is_empty():
		mutation_probability_field.text = "0.001"

	mutation_probability_field.tooltip_text = "Mutation probability per digit. Must be between 0 and 1."


## Opens the load dialog for Brain 1 or Brain 2.
## [param target] must be 1 or 2.
func _open_load_dialog(target: int) -> void:
	if _is_generating():
		return

	load_target = target
	load_dialog.popup_centered()


## Opens the operating system file chooser to save the current offspring configuration.
func _open_save_genome_dialog() -> void:
	if save_brain_button.disabled:
		return

	save_genome_dialog.popup_centered()


## Saves the generated offspring using the same JSON structure as BrainExplorer.
## [param path] is the file selected in the save dialog.
func _save_genome_to_file(path: String) -> void:
	if current_offspring_genome.is_empty() or config_1.is_empty():
		return

	var save_path := path if path.get_extension().to_lower() == "json" else path + ".json"
	var external_parameters: Dictionary = config_1["external_parameters"].duplicate(true)
	var genome_data := {}

	for gene_name in GENE_ORDER:
		var gene: Array = current_offspring_genome[gene_name]
		genome_data[gene_name] = _gene_to_text(gene)

	var data := {
		"format_version": 3,
		"external_parameters": external_parameters,
		"genome": genome_data
	}

	var file := FileAccess.open(save_path, FileAccess.WRITE)

	if file == null:
		summary_label.text = "Could not save brain configuration. Error: %s" % error_string(FileAccess.get_open_error())
		return

	file.store_string(JSON.stringify(data, "\t"))
	summary_label.text = "Brain configuration saved to:\n%s" % save_path


## Loads one parent configuration saved by BrainExplorer.
## [param path] is the JSON file selected by the user.
func _load_brain_configuration(path: String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)

	if file == null:
		summary_label.text = "Could not load brain configuration. Error: %s" % error_string(FileAccess.get_open_error())
		return

	var data = JSON.parse_string(file.get_as_text())

	_migrate_legacy_retention_factor_key(data)
	_migrate_legacy_initial_search_region(data)
	_migrate_legacy_connections_density(data)

	if not _is_valid_genome_file(data):
		summary_label.text = "Invalid brain configuration file:\n%s" % path
		return

	if load_target == 1:
		config_1 = data.duplicate(true)
		load_brain_1_button.tooltip_text = path
		load_brain_1_button.modulate = LOADED_BUTTON_MODULATE
	else:
		config_2 = data.duplicate(true)
		load_brain_2_button.tooltip_text = path
		load_brain_2_button.modulate = LOADED_BUTTON_MODULATE

	current_brain.clear()
	summary_label.modulate = Color.WHITE

	# Discard the previous child genome after changing either parent.
	current_offspring_genome.clear()

	_hide_analysis()
	_update_genetic_parameters()
	_update_loaded_summary()
	_update_generate_button()


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


## Older brain JSONs did not contain the initial search region gene.
## Use the connection gene as a deterministic legacy fallback.
func _migrate_legacy_initial_search_region(data: Variant) -> void:
	if not (data is Dictionary) or not data.has("genome") or not (data["genome"] is Dictionary):
		return
	var genes: Dictionary = data["genome"]
	if not genes.has("initial_search_region") and genes.has("connections"):
		genes["initial_search_region"] = genes["connections"]


## Legacy brains were generated with eta = 1 and have no density field.
func _migrate_legacy_connections_density(data: Variant) -> void:
	if data is Dictionary and data.has("external_parameters") and data["external_parameters"] is Dictionary:
		var external: Dictionary = data["external_parameters"]
		if not external.has("connections_density"):
			external["connections_density"] = "1"


## Returns true when the parsed JSON contains every field required by BrainBuilderGPU.
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

		if parameter_name == "connections_density":
			if not parameter_text.is_valid_float():
				return false
			var eta := float(parameter_text)
			if is_nan(eta) or is_inf(eta) or eta <= 0.0:
				return false
		elif parameter_text.is_empty() or _filter_characters(parameter_text, "0123456789") != parameter_text:
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


## Updates the summary before an offspring is generated.
func _update_loaded_summary() -> void:
	if config_1.is_empty() and config_2.is_empty():
		summary_label.text = "Load Brain 1 and Brain 2 to generate an offspring."
		return

	if config_1.is_empty():
		summary_label.text = "Brain 2 loaded.\nLoad Brain 1 to continue."
		return

	if config_2.is_empty():
		summary_label.text = "Brain 1 loaded.\nLoad Brain 2 to continue."
		return

	if not _external_parameters_match():
		var differences := _external_parameter_differences()
		summary_label.text = (
			"Both brains are loaded, but their external parameters are incompatible.\n\n"
			+ "The offspring needs one common brain geometry and input/output configuration.\n"
			+ "\n".join(differences)
		)
		return

	var external: Dictionary = config_1["external_parameters"]

	if not _is_valid_mutation_probability():
		summary_label.text = (
			"Dimensions: %d x %d x %d\n" % [int(external["size_x"]), int(external["size_y"]), int(external["size_z"])]
			+ "Inputs: %d    Outputs: %d\n\n" % [int(external["num_inputs"]), int(external["num_outputs"])]
			+ "Mutation probability must be a value between 0 and 1."
		)
		return

	summary_label.text = (
		"Dimensions: %d x %d x %d\n" % [int(external["size_x"]), int(external["size_y"]), int(external["size_z"])]
		+ "Inputs: %d    Outputs: %d\n" % [int(external["num_inputs"]), int(external["num_outputs"])]
		+ "Mutation probability per digit: %.6f\n\n" % float(mutation_probability_field.text)
		+ "Ready to generate offspring."
	)


## Returns a readable list of external parameters that differ between the parents.
func _external_parameter_differences() -> Array[String]:
	var differences: Array[String] = []

	if config_1.is_empty() or config_2.is_empty():
		return differences

	var external_1: Dictionary = config_1["external_parameters"]
	var external_2: Dictionary = config_2["external_parameters"]

	for parameter_name in EXTERNAL_PARAMETER_NAMES:
		if parameter_name == "connections_density":
			var eta_1 := float(external_1[parameter_name])
			var eta_2 := float(external_2[parameter_name])
			if eta_1 != eta_2:
				differences.append("Connections density: %s x %s" % [str(eta_1), str(eta_2)])
		else:
			var value_1 := int(external_1[parameter_name])
			var value_2 := int(external_2[parameter_name])
			if value_1 != value_2:
				differences.append(
					"%s: %d x %d" % [EXTERNAL_PARAMETER_LABELS[parameter_name], value_1, value_2]
				)

	return differences


## Returns true when all external parameters are numerically identical.
func _external_parameters_match() -> bool:
	if config_1.is_empty() or config_2.is_empty():
		return false

	var external_1: Dictionary = config_1["external_parameters"]
	var external_2: Dictionary = config_2["external_parameters"]

	for parameter_name in EXTERNAL_PARAMETER_NAMES:
		if parameter_name == "connections_density":
			if float(external_1[parameter_name]) != float(external_2[parameter_name]):
				return false
		elif int(external_1[parameter_name]) != int(external_2[parameter_name]):
			return false

	return true


## Enables offspring generation only when both compatible parent configurations are loaded.
func _update_generate_button() -> void:
	var can_generate := (
		not config_1.is_empty()
		and not config_2.is_empty()
		and _external_parameters_match()
		and _is_valid_mutation_probability()
	)
	var generating := _is_generating()

	generate_offspring_button.disabled = not can_generate or generating
	load_brain_1_button.disabled = generating
	load_brain_2_button.disabled = generating
	mutation_probability_field.editable = not generating
	save_brain_button.disabled = current_offspring_genome.is_empty() or current_brain.is_empty() or not bool(current_brain.get("valid", false)) or generating


## Imports a saved parent genome into the existing GDScript Genome class.
func _genome_from_config(config: Dictionary) -> Genome:
	var parent := Genome.new()
	var fields: Dictionary = config["genome"]
	for gene_name in GENE_ORDER:
		parent.set(gene_name, _gene_from_text(fields[gene_name]))
	return parent


func _gene_from_text(text: String) -> Array[int]:
	var gene: Array[int] = []

	for character in text:
		gene.append(int(character))

	return gene


## Converts an integer gene array into the text format used by saved brain configurations.
func _gene_to_text(gene: Array) -> String:
	var text := ""

	for digit in gene:
		text += str(digit)

	return text


## Returns true when the mutation probability field contains a value from 0 to 1.
func _is_valid_mutation_probability() -> bool:
	var text := mutation_probability_field.text.strip_edges()

	if text.is_empty() or not text.is_valid_float():
		return false

	var probability := float(text)
	return probability >= 0.0 and probability <= 1.0


## Revalidates offspring generation when the mutation probability changes.
func _on_mutation_probability_changed(_new_text: String) -> void:
	_update_generate_button()


## Starts generation of a new offspring and its brain.
func _generate_offspring() -> void:
	if generate_offspring_button.disabled or _is_generating():
		return
	if not _external_parameters_match():
		_update_loaded_summary()
		_update_generate_button()
		return

	var external: Dictionary = config_1["external_parameters"]
	var size := Vector3i(int(external["size_x"]), int(external["size_y"]), int(external["size_z"]))
	var num_inputs := int(external["num_inputs"])
	var num_outputs := int(external["num_outputs"])
	var connections_density := float(external["connections_density"])
	var parent_1: Genome = _genome_from_config(config_1)
	var parent_2: Genome = _genome_from_config(config_2)
	var mutation_probability: float = float(mutation_probability_field.text)
	var child: Genome = Offspring.new().generate(parent_1, parent_2, mutation_probability)
	# Only the child's phenotype is constructed. Parents are temporary gene sources.
	parent_1.free()
	parent_2.free()
	if child == null:
		summary_label.text = "Could not generate the offspring genome."
		return

	var genome: Dictionary = {}
	for gene_name in GENE_ORDER:
		var gene_values: Array = child.get(gene_name)
		genome[gene_name] = gene_values.duplicate()
	child.free()
	current_offspring_genome = genome
	current_brain.clear()
	generation_mutation_probability = mutation_probability
	generation_progress_mutex.lock()
	generation_progress = 0
	generation_progress_mutex.unlock()
	displayed_generation_progress = -1
	displayed_generation_stage = ""
	preparing_brain = false
	generation_start_time = Time.get_ticks_msec()

	_hide_analysis()
	summary_label.modulate = Color.WHITE
	summary_label.text = "Generating offspring brain... 0%"
	generate_offspring_button.disabled = true
	generation_thread = Thread.new()
	var error := generation_thread.start(_build_brain_thread.bind(genome, size, num_inputs, num_outputs, connections_density))
	_update_generate_button()
	if error != OK:
		generation_thread = null
		current_offspring_genome.clear()
		summary_label.text = "Could not start brain-generation thread. Error: %s" % error_string(error)
		_update_generate_button()


func _build_brain_thread(genome: Dictionary, size: Vector3i, num_inputs: int, num_outputs: int, connections_density: float) -> Dictionary:
	var builder := BrainBuilderGPU.new()
	return builder.build(genome, size, num_inputs, num_outputs, _set_generation_progress_from_thread, false, null, connections_density, BrainBuilderGPU.STATS_WEIGHT_HISTOGRAM)


func _set_generation_progress_from_thread(percent: int) -> void:
	generation_progress_mutex.lock()
	generation_progress = percent
	generation_progress_mutex.unlock()


## Keeps the UI responsive while an offspring brain is being generated.
func _process(_delta: float) -> void:
	if generation_thread == null:
		return

	generation_progress_mutex.lock()
	var builder_progress: int = generation_progress
	generation_progress_mutex.unlock()
	var progress: int = int(94.0 * float(builder_progress) / 100.0)
	var stage := _gpu_generation_stage(builder_progress)
	if progress != displayed_generation_progress or stage != displayed_generation_stage:
		_set_generation_status(progress, stage)

	if not generation_thread.is_alive():
		current_brain = generation_thread.wait_to_finish()
		generation_thread = null
		var elapsed_seconds: float = (Time.get_ticks_msec() - generation_start_time) / 1000.0
		if current_brain.is_empty():
			_update_generation_failure(elapsed_seconds)
			_update_generate_button()
			return
		preparing_brain = true
		call_deferred("_finish_brain_generation", elapsed_seconds)


func _gpu_generation_stage(builder_progress: int) -> String:
	if builder_progress >= 91:
		return "Reducing GPU connection statistics"
	if builder_progress >= 83:
		return "Finalizing synaptic weights on GPU"
	if builder_progress >= 75:
		return "Validating and rescuing connections"
	if builder_progress >= 21:
		return "Generating connections on GPU"
	if builder_progress >= 16:
		return "Assigning input and output neurons"
	return "Generating neuronal fields on GPU"


func _set_generation_status(percent: int, stage: String) -> void:
	displayed_generation_progress = percent
	displayed_generation_stage = stage
	summary_label.text = "%s... %d%%" % [stage, percent]


func _finish_brain_generation(build_seconds: float) -> void:
	_set_generation_status(95, "Calculating offspring statistics")
	await get_tree().process_frame
	_update_summary(current_brain, build_seconds)
	var final_summary: String = summary_label.text
	var final_color: Color = summary_label.modulate

	_set_generation_status(97, "Preparing histograms")
	await get_tree().process_frame
	_setup_histogram(current_brain)

	_set_generation_status(99, "Preparing heatmaps")
	await get_tree().process_frame
	_setup_heatmap(current_brain)

	var total_seconds: float = (Time.get_ticks_msec() - generation_start_time) / 1000.0
	summary_label.modulate = final_color
	summary_label.text = "Offspring brain ready (100%)\n\n" + final_summary + "\nTotal preparation time: %.3f s" % total_seconds
	displayed_generation_progress = 100
	displayed_generation_stage = "Complete"
	preparing_brain = false
	_update_generate_button()


func _is_generating() -> bool:
	return preparing_brain or (generation_thread != null and generation_thread.is_started())


func _update_generation_failure(elapsed_seconds: float) -> void:
	summary_label.modulate = Color(1.0, 0.35, 0.1)
	summary_label.text = (
		"Offspring brain construction could not complete.\n"
		+ "Check the Godot debugger for GPU/shader errors.\n"
		+ "Mutation probability per digit: %.6f\n" % generation_mutation_probability
		+ "Generation time: %.3f s" % elapsed_seconds
	)
	_hide_analysis()


func _update_summary(brain: Dictionary, elapsed_seconds: float) -> void:
	var neurons: Array = brain["neurons"]
	var total_neurons: int = neurons.size()
	var total_connections: int = brain["connection_total"]
	var out_degrees: PackedInt32Array = brain["outgoing_counts"]
	var I: int = brain["size"].x
	var J: int = brain["size"].y
	var excitatory := 0
	var inhibitory := 0
	var threshold_sum := 0.0
	var decay_sum := 0.0
	var retention_sum := 0.0
	var hebbian_sum := 0.0
	var modulatory_release_factor_sum := 0.0
	var modulatory_sensitivity_sum := 0.0
	var connection_energy_sum := 0.0

	if brain.get("valid", true):
		summary_label.modulate = Color.WHITE
	else:
		summary_label.modulate = Color(1.0, 0.35, 0.1)

	var min_out_degree := 0
	var max_out_degree := 0
	var outgoing_neuron_count := 0
	var outgoing_connection_count := 0
	var first_outgoing_neuron := true

	for neuron in neurons:
		var pos: Vector3i = neuron["position"]
		var spatial: int = pos.x + I * (pos.y + J * pos.z)
		var out_degree: int = out_degrees[spatial]
		threshold_sum += neuron["activation_threshold"]
		decay_sum += neuron["decay_factor"]
		retention_sum += neuron["retention_factor"]
		hebbian_sum += neuron["hebbian_plasticity_rate"]
		modulatory_release_factor_sum += neuron["modulatory_release_factor"]
		modulatory_sensitivity_sum += neuron["modulatory_sensitivity"]
		connection_energy_sum += neuron["connections_value"]
		if neuron["polarity_factor"] > 0:
			excitatory += 1
		else:
			inhibitory += 1
		if neuron["output"]:
			continue
		outgoing_neuron_count += 1
		outgoing_connection_count += out_degree
		if first_outgoing_neuron:
			min_out_degree = out_degree
			max_out_degree = out_degree
			first_outgoing_neuron = false
		else:
			min_out_degree = mini(min_out_degree, out_degree)
			max_out_degree = maxi(max_out_degree, out_degree)

	var mean_connections: int = 0 if outgoing_neuron_count == 0 else int(float(outgoing_connection_count) / outgoing_neuron_count)
	var mean_weight: float = 0.0 if total_connections == 0 else float(brain["connection_weight_sum"]) / float(total_connections)
	var neuron_denominator: int = maxi(1, total_neurons)
	var isolated: Dictionary = brain.get("isolated_before_rescue", {})
	var rescue_input: int = int(isolated.get("input", 0))
	var rescue_internal: int = int(isolated.get("internal", 0))
	var rescue_output: int = int(isolated.get("output", 0))
	var rescue_evaluations: int = int(brain.get("rescue_candidates_total", 0))
	var status_text := ""
	if not brain.get("valid", true):
		status_text = "Brain generation failed.\nIsolated neurons.\n\n"

	summary_label.text = (
		status_text
		+ "Dimensions: %d x %d x %d\n" % [brain["size"].x, brain["size"].y, brain["size"].z]
		+ "Mutation probability per digit: %.6f\n" % generation_mutation_probability
		+ "Beta: %.6f\n" % brain["beta"]
		+ "Neurons: %d\n" % total_neurons
		+ "Maximum connections per neuron: %d\n" % brain["max_connections"]
		+ "Input influence psi: %.6f\n" % brain.get("input_gain", 0.0)
		+ "Existing directional connections: %d\n" % total_connections
		+ "Initial connections (1st/2nd pass): %d / %d\n" % [int(brain["initial_local_connections"]), int(brain["initial_additional_connections"])]
		+ "Mean outgoing connections: %d\n" % mean_connections
		+ "Outgoing degree range: %d - %d\n" % [min_out_degree, max_out_degree]
		+ "Isolated before rescue (input/internal/output): %d / %d / %d\n" % [rescue_input, rescue_internal, rescue_output]
		+ "Rescue candidates evaluated (H): %d\n" % rescue_evaluations
		+ "Excitatory neurons: %d (%.2f%%)\n" % [excitatory, 100.0 * excitatory / neuron_denominator]
		+ "Inhibitory neurons: %d (%.2f%%)\n\n" % [inhibitory, 100.0 * inhibitory / neuron_denominator]
		+ "Mean activation threshold: %.6f\n" % (threshold_sum / neuron_denominator)
		+ "Mean decay factor: %.6f\n" % (decay_sum / neuron_denominator)
		+ "Mean retention factor: %.6f\n" % (retention_sum / neuron_denominator)
		+ "Mean Hebbian plasticity rate: %.6f\n" % (hebbian_sum / neuron_denominator)
		+ "Mean modulatory release factor: %.6f\n" % (modulatory_release_factor_sum / neuron_denominator)
		+ "Mean modulatory sensitivity: %.6f\n" % (modulatory_sensitivity_sum / neuron_denominator)
		+ "Mean connection energy: %.6f\n" % (connection_energy_sum / neuron_denominator)
		+ "Mean connection weight: %.6f\n\n" % mean_weight
		+ "Maximum fatigue: %.6f\n" % brain["refractory_strength"]
		+ "Exponential factor: %.6f\n" % brain["exponential_factor"]
		+ "Modulatory persistence lambda: %.6f\n" % brain["lambda"]
		+ "Modulatory spread nu: %.6f\n" % brain["nu"]
		+ "Trace persistence mu: %.6f\n" % brain["mu"]
		+ "Structural plasticity: %.6f\n" % brain["structural_plasticity"]
		+ "Build time: %.3f s" % elapsed_seconds
	)

## Displays differences between the parent genomes.
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


## Counts how many positions differ between two equal length genes.
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


## Builds histogram data for every supported parameter, both globally and per brain slice.
## [param brain] contains the brain structure and neuron properties.
## Returns nothing.
func _build_histogram_data(brain: Dictionary) -> void:
	var neurons: Array = brain["neurons"]
	var I: int = brain["size"].x
	var J: int = brain["size"].y
	var K: int = brain["size"].z
	var incoming: PackedInt32Array = brain["incoming_counts"]
	var degrees: PackedInt32Array = brain["outgoing_counts"]
	var weight_slices: Array = brain["connection_weight_bins"]
	var incoming_max_degree := 1
	var outgoing_max_degree := 1
	for spatial in incoming.size():
		incoming_max_degree = maxi(incoming_max_degree, incoming[spatial])
		outgoing_max_degree = maxi(outgoing_max_degree, degrees[spatial])

	histogram_data.clear()
	for parameter_key in HISTOGRAM_PARAMETER_KEYS:
		var value_range := _histogram_range(parameter_key, incoming_max_degree, outgoing_max_degree)
		var slices: Array = []
		slices.resize(K)
		for k in K:
			var bins := PackedInt32Array()
			bins.resize(BINS)
			slices[k] = bins
		var all_bins := PackedInt32Array()
		all_bins.resize(BINS)
		histogram_data[parameter_key] = {
			"slices": slices,
			"all": all_bins,
			"range": value_range,
			"integer_range": parameter_key in ["incoming_connections", "outgoing_connections"]
		}

	# The packed builder already computed the weight distribution in one pass.
	var weight_all: PackedInt32Array = histogram_data["connection_weight"]["all"]
	for slice_bins in weight_slices:
		var bins: PackedInt32Array = slice_bins
		for i in BINS:
			weight_all[i] += bins[i]
	histogram_data["connection_weight"]["slices"] = weight_slices
	histogram_data["connection_weight"]["all"] = weight_all

	for neuron in neurons:
		var pos: Vector3i = neuron["position"]
		var spatial: int = pos.x + I * (pos.y + J * pos.z)
		var k: int = pos.z
		_add_histogram_value("incoming_connections", incoming[spatial], k)
		_add_histogram_value("outgoing_connections", degrees[spatial], k)
		_add_histogram_value("activation_threshold", neuron["activation_threshold"], k)
		_add_histogram_value("decay_factor", neuron["decay_factor"], k)
		_add_histogram_value("retention_factor", neuron["retention_factor"], k)
		_add_histogram_value("hebbian_plasticity_rate", neuron["hebbian_plasticity_rate"], k)
		_add_histogram_value("modulatory_release_factor", neuron["modulatory_release_factor"], k)
		_add_histogram_value("modulatory_sensitivity", neuron["modulatory_sensitivity"], k)
		_add_histogram_value("polarity_factor", neuron["polarity_factor"], k)
		_add_histogram_value("connection_energy", neuron["connections_value"], k)
		var role := 0.0
		if neuron["input"]:
			role = -1.0
		elif neuron["output"]:
			role = 1.0
		_add_histogram_value("neuron_role", role, k)


## Returns the common x-axis range used by a histogram parameter across all slices.


func _histogram_range(parameter_key: String, incoming_max_degree: int, outgoing_max_degree: int) -> Vector2:
	match parameter_key:
		"connection_weight", "activation_threshold", "retention_factor", "modulatory_release_factor":
			return Vector2(0.0, 1.0)
		"decay_factor":
			return Vector2(0.5, 2.5)
		"hebbian_plasticity_rate":
			return Vector2(0.0, 0.5)
		"modulatory_sensitivity", "polarity_factor", "connection_energy", "neuron_role":
			return Vector2(-1.0, 1.0)
		"incoming_connections":
			return Vector2(0.0, incoming_max_degree)
		"outgoing_connections":
			return Vector2(0.0, outgoing_max_degree)

	return Vector2(0.0, 1.0)


## Adds one value to both the whole-brain and corresponding slice histogram.
func _add_histogram_value(parameter_key: String, value: float, k: int) -> void:
	var parameter_data: Dictionary = histogram_data[parameter_key]
	var value_range: Vector2 = parameter_data["range"]
	var normalized := 0.0

	if value_range.y > value_range.x:
		normalized = (value - value_range.x) / (value_range.y - value_range.x)

	var bin := clampi(int(normalized * BINS), 0, BINS - 1)
	parameter_data["slices"][k][bin] += 1
	parameter_data["all"][bin] += 1


## Prepares the shared histogram and its slice and parameter selectors.
func _setup_histogram(brain: Dictionary) -> void:
	_build_histogram_data(brain)

	histogram_slice_selector.clear()
	histogram_slice_selector.add_item("All slices")

	for k in brain["size"].z:
		histogram_slice_selector.add_item("k = %d" % k)

	histogram_parameter_selector.clear()
	for label in HISTOGRAM_PARAMETER_LABELS:
		histogram_parameter_selector.add_item(label)

	histogram_slice_selector.select(0)
	histogram_parameter_selector.select(0)
	_update_histogram()

	histogram_label.show()
	histogram_slice_selector.show()
	histogram_parameter_selector.show()
	histogram.show()


## Updates the histogram using the currently selected slice and parameter.
func _update_histogram() -> void:
	if current_brain.is_empty():
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


## Builds spatial scalar fields used by the neuron heatmap.
## [param brain] contains the brain structure and neuron properties.
## Returns nothing.
func _build_heatmap_data(brain: Dictionary) -> void:
	var I: int = brain["size"].x
	var J: int = brain["size"].y
	var K: int = brain["size"].z
	var IJ: int = I * J
	var total_neurons: int = IJ * K

	var activation_values := PackedFloat64Array()
	var decay_values := PackedFloat64Array()
	var retention_values := PackedFloat64Array()
	var hebbian_values := PackedFloat64Array()
	var modulatory_release_factor_values := PackedFloat64Array()
	var modulatory_sensitivity_values := PackedFloat64Array()
	var polarity_values := PackedFloat64Array()
	var connection_energy_values := PackedFloat64Array()
	var role_values := PackedFloat64Array()
	var incoming_values := PackedFloat64Array()
	activation_values.resize(total_neurons)
	decay_values.resize(total_neurons)
	retention_values.resize(total_neurons)
	hebbian_values.resize(total_neurons)
	modulatory_release_factor_values.resize(total_neurons)
	modulatory_sensitivity_values.resize(total_neurons)
	polarity_values.resize(total_neurons)
	connection_energy_values.resize(total_neurons)
	role_values.resize(total_neurons)
	incoming_values.resize(total_neurons)

	# Incoming degrees were calculated by the GPU builder, including rescue edges.
	var incoming_counts: PackedInt32Array = brain["incoming_counts"]

	var activation_min := INF
	var activation_max := -INF
	var decay_min := INF
	var decay_max := -INF
	var retention_min := INF
	var retention_max := -INF
	var hebbian_min := INF
	var hebbian_max := -INF
	var incoming_min := INF
	var incoming_max := -INF

	for neuron in brain["neurons"]:
		var pos: Vector3i = neuron["position"]
		var spatial_index: int = pos.x + I * pos.y + IJ * pos.z

		var activation: float = neuron["activation_threshold"]
		var decay: float = neuron["decay_factor"]
		var retention: float = neuron["retention_factor"]
		var hebbian: float = neuron["hebbian_plasticity_rate"]
		var modulatory_release_factor: float = neuron["modulatory_release_factor"]
		var modulatory_sensitivity: float = neuron["modulatory_sensitivity"]
		var polarity: float = neuron["polarity_factor"]
		var connection_energy: float = neuron["connections_value"]
		var role := 0.0
		var incoming: float = incoming_counts[spatial_index]

		if neuron["input"]:
			role = -1.0
		elif neuron["output"]:
			role = 1.0

		activation_values[spatial_index] = activation
		decay_values[spatial_index] = decay
		retention_values[spatial_index] = retention
		hebbian_values[spatial_index] = hebbian
		modulatory_release_factor_values[spatial_index] = modulatory_release_factor
		modulatory_sensitivity_values[spatial_index] = modulatory_sensitivity
		polarity_values[spatial_index] = polarity
		connection_energy_values[spatial_index] = connection_energy
		role_values[spatial_index] = role
		incoming_values[spatial_index] = incoming

		activation_min = minf(activation_min, activation)
		activation_max = maxf(activation_max, activation)
		decay_min = minf(decay_min, decay)
		decay_max = maxf(decay_max, decay)
		retention_min = minf(retention_min, retention)
		retention_max = maxf(retention_max, retention)
		hebbian_min = minf(hebbian_min, hebbian)
		hebbian_max = maxf(hebbian_max, hebbian)
		incoming_min = minf(incoming_min, incoming)
		incoming_max = maxf(incoming_max, incoming)

	heatmap_data = {
		"activation_threshold": activation_values,
		"decay_factor": decay_values,
		"retention_factor": retention_values,
		"hebbian_plasticity_rate": hebbian_values,
		"modulatory_release_factor": modulatory_release_factor_values,
		"modulatory_sensitivity": modulatory_sensitivity_values,
		"polarity_factor": polarity_values,
		"connection_energy": connection_energy_values,
		"neuron_role": role_values,
		"incoming_connections": incoming_values
	}

	heatmap_ranges = {
		"activation_threshold": Vector2(activation_min, activation_max),
		"decay_factor": Vector2(decay_min, decay_max),
		"retention_factor": Vector2(retention_min, retention_max),
		"hebbian_plasticity_rate": Vector2(hebbian_min, hebbian_max),
		"modulatory_release_factor": Vector2(0.0, 1.0),
		"modulatory_sensitivity": Vector2(-1.0, 1.0),
		"polarity_factor": Vector2(-1.0, 1.0),
		"connection_energy": Vector2(-1.0, 1.0),
		"neuron_role": Vector2(-1.0, 1.0),
		"incoming_connections": Vector2(incoming_min, incoming_max)
	}


## Prepares the neuron heatmap and its selectors.
## [param brain] contains the brain structure and neuron properties.
## Returns nothing.


func _setup_heatmap(brain: Dictionary) -> void:
	_build_heatmap_data(brain)

	heatmap_slice_selector.clear()
	for k in brain["size"].z:
		heatmap_slice_selector.add_item("k = %d" % k)

	heatmap_parameter_selector.clear()
	for label in HEATMAP_PARAMETER_LABELS:
		heatmap_parameter_selector.add_item(label)

	heatmap_slice_selector.select(0)
	heatmap_parameter_selector.select(0)
	_update_heatmap()

	heatmap_label.show()
	heatmap_slice_selector.show()
	heatmap_parameter_selector.show()
	neuron_heatmap.show()


## Updates the heatmap using the currently selected slice and neuron parameter.
func _update_heatmap() -> void:
	if current_brain.is_empty():
		return

	var parameter_index: int = heatmap_parameter_selector.selected
	var k: int = heatmap_slice_selector.selected

	if parameter_index < 0 or k < 0:
		return

	var parameter_key: String = HEATMAP_PARAMETER_KEYS[parameter_index]
	var I: int = current_brain["size"].x
	var J: int = current_brain["size"].y
	var slice_size: int = I * J
	var slice_start: int = k * slice_size
	var all_values: PackedFloat64Array = heatmap_data[parameter_key]
	var slice_values: PackedFloat64Array = all_values.slice(slice_start, slice_start + slice_size)
	var value_range: Vector2 = heatmap_ranges[parameter_key]

	heatmap_label.text = "%s Heatmap" % HEATMAP_PARAMETER_LABELS[parameter_index]
	neuron_heatmap.set_data(slice_values, Vector2i(I, J), value_range.x, value_range.y)


## Hides every graph until a valid offspring brain has been generated.
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


## Updates the neuron heatmap when either selector changes.
func _on_heatmap_selection_changed(_index: int) -> void:
	_update_heatmap()
