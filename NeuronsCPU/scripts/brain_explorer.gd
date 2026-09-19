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

const GENE_FIELD_NAMES = {
	"max_connections": "LineEdit6",
	"connections": "LineEdit7",
	"activation_threshold": "LineEdit8",
	"exponential_factor": "LineEdit9",
	"decay_factor": "LineEdit10",
	"retention_factor": "LineEdit11",
	"polarity_factor": "LineEdit12",
	"hebbian_plasticity_rate": "LineEdit13",
	"structural_plasticity": "LineEdit14",
	"beta": "LineEdit15",
	"modulatory_release_factor": "LineEdit16",
	"modulatory_sensitivity": "LineEdit17",
	"modulatory_dynamics": "LineEdit18",
	"refractory_strength": "LineEdit19",
	"input_influence": "LineEdit20"
}


const EXTERNAL_PARAMETER_FIELD_NAMES = {
	"size_x": "LineEdit",
	"size_y": "LineEdit2",
	"size_z": "LineEdit3",
	"num_inputs": "LineEdit4",
	"num_outputs": "LineEdit5"
}

const BINS := 20

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


var current_brain: Dictionary = {}
var summary_label: Label

var generation_thread: Thread
var generation_progress_mutex = Mutex.new()
var generation_progress = 0
var displayed_generation_progress = -1
var generation_start_time = 0
var histogram_data: Dictionary = {}
var heatmap_data: Dictionary = {}
var heatmap_ranges: Dictionary = {}
var save_genome_dialog: FileDialog
var load_genome_dialog: FileDialog

@onready var histogram_label: Label = $HistogramLabel
@onready var histogram: Histogram = $Histogram
@onready var histogram_slice_selector: OptionButton = $HistogramSliceSelector
@onready var histogram_parameter_selector: OptionButton = $HistogramParameterSelector
@onready var heatmap_label: Label = $HeatmapLabel
@onready var neuron_heatmap: Heatmap = $Heatmap
@onready var heatmap_slice_selector: OptionButton = $HeatmapSliceSelector
@onready var heatmap_parameter_selector: OptionButton = $HeatmapParameterSelector


func _ready() -> void:
	histogram_slice_selector.item_selected.connect(_on_histogram_selection_changed)
	histogram_parameter_selector.item_selected.connect(_on_histogram_selection_changed)
	heatmap_slice_selector.item_selected.connect(_on_heatmap_selection_changed)
	heatmap_parameter_selector.item_selected.connect(_on_heatmap_selection_changed)
	histogram_label.hide()
	histogram_slice_selector.hide()
	histogram_parameter_selector.hide()
	histogram.hide()
	heatmap_label.hide()
	heatmap_slice_selector.hide()
	heatmap_parameter_selector.hide()
	neuron_heatmap.hide()
	_setup_fields()
	_setup_gene_randomize_buttons()
	_setup_summary()
	_setup_genome_file_dialogs()

	$RandomizeGenome.pressed.connect(_randomize_genome)
	$GenerateBrain.pressed.connect(_generate_brain)
	$SaveBrain.pressed.connect(_open_save_genome_dialog)
	$LoadBrain.pressed.connect(_open_load_genome_dialog)

	_validate_form()


## Configures the external-parameter and gene fields.
func _setup_fields() -> void:
	for parameter_name in EXTERNAL_PARAMETER_FIELD_NAMES:
		var field: LineEdit = get_node(EXTERNAL_PARAMETER_FIELD_NAMES[parameter_name])
		field.tooltip_text = field.placeholder_text
		field.text_changed.connect(_on_external_parameter_changed.bind(field))

	for gene_name in GENE_ORDER:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		var gene_size: int = GENE_SPECS[gene_name]
		field.max_length = gene_size
		field.tooltip_text = field.placeholder_text + " (%d digits)" % gene_size
		field.text_changed.connect(_on_gene_changed.bind(field))

## Creates one small randomization button beside each gene field.
func _setup_gene_randomize_buttons() -> void:
	var button_template: Button = $Button
	button_template.hide()

	for gene_name in GENE_SPECS:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		var button: Button = button_template.duplicate()
		button.show()
		button.position = field.position + Vector2(field.size.x + 10.0, -3.0)
		button.tooltip_text = "Randomize " + field.placeholder_text
		button.pressed.connect(_randomize_gene.bind(field, GENE_SPECS[gene_name]))
		add_child(button)

## Creates the preliminary brain summary area on the left side of the screen.
func _setup_summary() -> void:
	var title = Label.new()
	title.position = Vector2(30, 10)
	title.add_theme_font_size_override("font_size", 22)
	title.text = "Brain Summary"
	add_child(title)

	summary_label = $BrainInfoLabel
	summary_label.position = Vector2(30, 45)
	summary_label.size = Vector2(800, 720)
	summary_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	summary_label.text = "Fill the parameters and generate a brain."


## Creates native file dialogs for saving and loading brain configuration JSON files.
func _setup_genome_file_dialogs() -> void:
	save_genome_dialog = FileDialog.new()
	save_genome_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	save_genome_dialog.access = FileDialog.ACCESS_FILESYSTEM
	save_genome_dialog.use_native_dialog = true
	save_genome_dialog.filters = PackedStringArray(["*.json ; JSON files"])
	save_genome_dialog.current_file = "brain.json"
	save_genome_dialog.file_selected.connect(_save_genome_to_file)
	add_child(save_genome_dialog)

	load_genome_dialog = FileDialog.new()
	load_genome_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	load_genome_dialog.access = FileDialog.ACCESS_FILESYSTEM
	load_genome_dialog.use_native_dialog = true
	load_genome_dialog.filters = PackedStringArray(["*.json ; JSON files"])
	load_genome_dialog.file_selected.connect(_load_genome_from_file)
	add_child(load_genome_dialog)


## Opens the operating system file chooser to save the current brain configuration.
func _open_save_genome_dialog() -> void:
	if $SaveBrain.disabled:
		return

	save_genome_dialog.popup_centered()


## Opens the operating system file chooser to load a brain configuration.
func _open_load_genome_dialog() -> void:
	if $LoadBrain.disabled:
		return

	load_genome_dialog.popup_centered()


## Saves every field shown in the interface to a JSON file.
## [param path] is the file selected in the save dialog.
func _save_genome_to_file(path: String) -> void:
	var save_path := path if path.get_extension().to_lower() == "json" else path + ".json"
	var external_parameters := {}
	var genome_data := {}

	for parameter_name in EXTERNAL_PARAMETER_FIELD_NAMES:
		var field: LineEdit = get_node(EXTERNAL_PARAMETER_FIELD_NAMES[parameter_name])
		external_parameters[parameter_name] = field.text

	for gene_name in GENE_ORDER:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		genome_data[gene_name] = field.text

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


## Loads every field shown in the interface from a JSON file.
## [param path] is the file selected in the open dialog.
func _load_genome_from_file(path: String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)

	if file == null:
		summary_label.text = "Could not load brain configuration. Error: %s" % error_string(FileAccess.get_open_error())
		return

	var data = JSON.parse_string(file.get_as_text())

	_migrate_legacy_retention_factor_key(data)

	if not _is_valid_genome_file(data):
		summary_label.text = "Invalid brain configuration file:\n%s" % path
		return

	var external_parameters: Dictionary = data["external_parameters"]
	var genome_data: Dictionary = data["genome"]

	for parameter_name in EXTERNAL_PARAMETER_FIELD_NAMES:
		var field: LineEdit = get_node(EXTERNAL_PARAMETER_FIELD_NAMES[parameter_name])
		field.text = external_parameters[parameter_name]

	for gene_name in GENE_ORDER:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		field.text = genome_data[gene_name]

	_validate_form()
	summary_label.text = "Brain configuration loaded from:\n%s" % path


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


## Returns true when a parsed JSON value contains every valid interface field.
func _is_valid_genome_file(data: Variant) -> bool:
	if not (data is Dictionary):
		return false

	if not data.has("external_parameters") or not (data["external_parameters"] is Dictionary):
		return false

	if not data.has("genome") or not (data["genome"] is Dictionary):
		return false

	var external_parameters: Dictionary = data["external_parameters"]
	var genome_data: Dictionary = data["genome"]

	for parameter_name in EXTERNAL_PARAMETER_FIELD_NAMES:
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

## Keeps external parameters restricted to non-negative integers.
func _on_external_parameter_changed(new_text: String, field: LineEdit) -> void:
	var filtered = _filter_characters(new_text, "0123456789")

	if filtered != new_text:
		var caret = field.caret_column
		field.text = filtered
		field.caret_column = mini(caret, filtered.length())

	_validate_form()

## Keeps gene fields restricted to the alphabet {0, 1, 2, 3}.
func _on_gene_changed(new_text: String, field: LineEdit) -> void:
	var filtered = _filter_characters(new_text, "0123")

	if filtered != new_text:
		var caret = field.caret_column
		field.text = filtered
		field.caret_column = mini(caret, filtered.length())

	_validate_form()

## Returns only characters contained in allowed_characters.
func _filter_characters(text: String, allowed_characters: String) -> String:
	var filtered = ""

	for character in text:
		if allowed_characters.contains(character):
			filtered += character

	return filtered

## Randomizes all genetic parameters without changing external parameters.
func _randomize_genome() -> void:
	for gene_name in GENE_SPECS:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		field.text = _random_gene_text(GENE_SPECS[gene_name])

	_validate_form()

## Randomizes a single gene field.
func _randomize_gene(field: LineEdit, gene_size: int) -> void:
	field.text = _random_gene_text(gene_size)
	_validate_form()

## Generates a random gene string with digits in {0, 1, 2, 3}.
func _random_gene_text(gene_size: int) -> String:
	var gene = ""

	for _i in gene_size:
		gene += str(randi_range(0, 3))

	return gene

## Validates all fields and enables brain generation only for a valid configuration.
func _validate_form() -> void:
	var valid = true

	for parameter_name in EXTERNAL_PARAMETER_FIELD_NAMES:
		var field: LineEdit = get_node(EXTERNAL_PARAMETER_FIELD_NAMES[parameter_name])
		if field.text.is_empty():
			valid = false

	for gene_name in GENE_SPECS:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		if field.text.length() != GENE_SPECS[gene_name]:
			valid = false

	if valid:
		var size = Vector3i(
			int($LineEdit.text),
			int($LineEdit2.text),
			int($LineEdit3.text)
		)
		var num_inputs = int($LineEdit4.text)
		var num_outputs = int($LineEdit5.text)

		if size.x < 2 or size.y < 2 or size.z < 2:
			valid = false
		elif num_inputs + num_outputs > size.x * size.y * size.z:
			valid = false

	var generating := _is_generating()
	$GenerateBrain.disabled = not valid or generating
	$SaveBrain.disabled = not valid or generating
	$LoadBrain.disabled = generating

## Builds a Genome object using exactly the genes written in the interface.
func _genome_from_fields() -> Genome:
	var genome = Genome.new()

	for gene_name in GENE_ORDER:
		var field: LineEdit = get_node(GENE_FIELD_NAMES[gene_name])
		genome.set(gene_name, _gene_from_text(field.text))

	return genome

## Converts a gene string into the integer array expected by Genome.
func _gene_from_text(text: String) -> Array[int]:
	var gene: Array[int] = []

	for character in text:
		gene.append(int(character))

	return gene

## Generates the brain corresponding exactly to the current fields in a worker thread.
func _generate_brain() -> void:
	if $GenerateBrain.disabled or _is_generating():
		return

	var size = Vector3i(int($LineEdit.text), int($LineEdit2.text), int($LineEdit3.text))
	var num_inputs = int($LineEdit4.text)
	var num_outputs = int($LineEdit5.text)
	var genome = _genome_from_fields()

	generation_progress_mutex.lock()
	generation_progress = 0
	generation_progress_mutex.unlock()
	displayed_generation_progress = -1
	generation_start_time = Time.get_ticks_msec()

	summary_label.text = "Generating brain... 0%"
	$GenerateBrain.disabled = true

	generation_thread = Thread.new()
	var error = generation_thread.start(_build_brain_thread.bind(genome, size, num_inputs, num_outputs))
	_validate_form()

	if error != OK:
		generation_thread = null
		summary_label.text = "Could not start brain-generation thread. Error: %s" % error_string(error)
		_validate_form()


## Worker-thread entry point. It must not access the scene tree or UI nodes.
func _build_brain_thread(genome: Genome, size: Vector3i, num_inputs: int, num_outputs: int) -> Dictionary:
	var builder = BrainBuilder.new()
	return builder.build(genome, size, num_inputs, num_outputs, _set_generation_progress_from_thread)


## Called by BrainBuilder from the worker thread. Only shared progress state is touched here.
func _set_generation_progress_from_thread(percent: int) -> void:
	generation_progress_mutex.lock()
	generation_progress = percent
	generation_progress_mutex.unlock()


## Keeps the UI responsive while a brain is being generated and prints progress to Output.
func _process(_delta: float) -> void:
	if not _is_generating():
		return

	generation_progress_mutex.lock()
	var progress = generation_progress
	generation_progress_mutex.unlock()

	if progress != displayed_generation_progress:
		displayed_generation_progress = progress
		summary_label.text = "Generating brain... %d%%" % progress

	if not generation_thread.is_alive():
		current_brain = generation_thread.wait_to_finish()
		generation_thread = null

		if displayed_generation_progress != 100:
			displayed_generation_progress = 100

		var elapsed_seconds = (Time.get_ticks_msec() - generation_start_time) / 1000.0
		
		if current_brain.is_empty():
			_update_generation_failure(elapsed_seconds)
			_validate_form()
			return
	
		_update_summary(current_brain, elapsed_seconds)
		_setup_histogram(current_brain)
		_setup_heatmap(current_brain)
		_validate_form()

## Returns true while the worker thread is still owned by this scene.
func _is_generating() -> bool:
	return generation_thread != null and generation_thread.is_started()

## Displays the result of a failed brain generation.
func _update_generation_failure(elapsed_seconds: float) -> void:
	summary_label.modulate = Color(1.0, 0.35, 0.1)
	summary_label.text = "Brain generation failed.\nIsolated neurons were found.\nGeneration time: %.3f s" % elapsed_seconds
	histogram_label.hide()
	histogram_slice_selector.hide()
	histogram_parameter_selector.hide()
	histogram.hide()
	heatmap_label.hide()
	heatmap_slice_selector.hide()
	heatmap_parameter_selector.hide()
	neuron_heatmap.hide()

## Displays a first set of structural statistics for exploratory analysis.
func _update_summary(brain: Dictionary, elapsed_seconds: float) -> void:
	var neurons: Array = brain["neurons"]
	var total_neurons = neurons.size()
	var total_connections = 0
	var excitatory = 0
	var inhibitory = 0
	var threshold_sum = 0.0
	var decay_sum = 0.0
	var retention_sum = 0.0
	var hebbian_sum = 0.0
	var modulatory_release_factor_sum = 0.0
	var modulatory_sensitivity_sum = 0.0
	var connection_energy_sum = 0.0
	var weight_sum = 0.0
	var incoming = {}

	if brain.get("valid", true):
		summary_label.modulate = Color.WHITE
	else:
		summary_label.modulate = Color(1.0, 0.35, 0.1)

	for neuron in neurons:
		incoming[neuron["position"]] = 0

	for neuron in neurons:
		var connections: Array = neuron["connections"]
		total_connections += connections.size()
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

		for connection in connections:
			weight_sum += connection["weight"]
			incoming[connection["target"]] += 1

	var min_out_degree = 0
	var max_out_degree = 0
	var outgoing_neuron_count = 0
	var outgoing_connection_count = 0
	var first_outgoing_neuron = true

	for neuron in neurons:
		if neuron["output"]:
			continue

		var out_degree = neuron["connections"].size()
		outgoing_neuron_count += 1
		outgoing_connection_count += out_degree

		if first_outgoing_neuron:
			min_out_degree = out_degree
			max_out_degree = out_degree
			first_outgoing_neuron = false
		else:
			min_out_degree = mini(min_out_degree, out_degree)
			max_out_degree = maxi(max_out_degree, out_degree)

	var mean_connections = 0 if outgoing_neuron_count == 0 else int(float(outgoing_connection_count) / outgoing_neuron_count)
	var mean_weight = 0.0 if total_connections == 0 else weight_sum / total_connections
	var neuron_denominator = max(1, total_neurons)
	var status_text = ""

	if not brain.get("valid", true):
		var isolated_after: Dictionary = brain.get("isolated_after_rescue", {})
		status_text = "Brain generation failed.\nIsolated neurons.\n\n"

	summary_label.text = (
		status_text
		+ "Dimensions: %d x %d x %d\n" % [brain["size"].x, brain["size"].y, brain["size"].z]
		+ "Beta: %.6f\n" % brain["beta"]
		+ "Neurons: %d\n" % total_neurons
		+ "Maximum connections per neuron: %d\n" % brain["max_connections"]
		+ "Input influence psi: %.6f\n" % brain.get("input_gain", 0.0)
		+ "Existing directional connections: %d\n" % total_connections
		+ "Mean outgoing connections: %d\n" % mean_connections
		+ "Outgoing degree range: %d - %d\n" % [min_out_degree, max_out_degree]
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

## Builds histogram data for every supported parameter, both globally and per brain slice.
## [param brain] contains the brain structure and neuron properties.
## Returns nothing.
func _build_histogram_data(brain: Dictionary) -> void:
	var neurons: Array = brain["neurons"]
	var K: int = brain["size"].z
	var incoming := {}

	for neuron in neurons:
		incoming[neuron["position"]] = 0

	for neuron in neurons:
		for connection in neuron["connections"]:
			incoming[connection["target"]] += 1

	var incoming_max_degree := 1
	var outgoing_max_degree := 1

	for neuron in neurons:
		incoming_max_degree = maxi(incoming_max_degree, incoming[neuron["position"]])
		outgoing_max_degree = maxi(outgoing_max_degree, neuron["connections"].size())

	histogram_data.clear()

	for parameter_key in HISTOGRAM_PARAMETER_KEYS:
		var value_range = _histogram_range(parameter_key, incoming_max_degree, outgoing_max_degree)
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

	for neuron in neurons:
		var k: int = neuron["position"].z

		_add_histogram_value("incoming_connections", incoming[neuron["position"]], k)
		_add_histogram_value("outgoing_connections", neuron["connections"].size(), k)
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

		for connection in neuron["connections"]:
			_add_histogram_value("connection_weight", connection["weight"], k)


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

	var incoming_counts := PackedInt32Array()
	incoming_counts.resize(total_neurons)

	for neuron in brain["neurons"]:
		for connection in neuron["connections"]:
			var target: Vector3i = connection["target"]
			var target_index: int = target.x + I * target.y + IJ * target.z
			incoming_counts[target_index] += 1

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

## Updates the histogram when either selector changes.
func _on_histogram_selection_changed(_index: int) -> void:
	_update_histogram()

## Updates the neuron heatmap when either selector changes.
func _on_heatmap_selection_changed(_index: int) -> void:
	_update_heatmap()
	
