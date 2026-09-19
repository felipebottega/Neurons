extends Node3D


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

const EXTERNAL_PARAMETER_NAMES = [
	"size_x",
	"size_y",
	"size_z",
	"num_inputs",
	"num_outputs"
]

const HEATMAP_PARAMETER_KEYS = [
	"signal",
	"signal_0_5",
	"signal_0_1",
	"activation_threshold",
	"firing",
	"activity_trace",
	"fatigue",
	"decay_factor",
	"retention_factor",
	"hebbian_plasticity_rate",
	"modulatory_release_factor",
	"modulatory_sensitivity",
	"modulatory_release",
	"modulatory_field",
	"effective_modulation",
	"polarity_factor",
	"connection_energy",
	"neuron_role",
	"incoming_connections"
]

const HEATMAP_PARAMETER_LABELS = [
	"Signal",
	"Signal (max 0.5)",
	"Signal (max 0.1)",
	"Activation threshold",
	"Firing",
	"Activity trace R_t",
	"Fatigue",
	"Decay factor",
	"Retention factor",
	"Hebbian plasticity rate",
	"Modulatory release factor",
	"Modulatory sensitivity",
	"Modulatory release",
	"Modulatory field",
	"Effective modulation",
	"Polarity factor",
	"Connection energy",
	"Neuron role",
	"Incoming connections"
]

const HEATMAP_PALETTE: Array[Color] = [
	Color("#0000BB"),
	Color("#0000F6"),
	Color("#0020FF"),
	Color("#0054FF"),
	Color("#008CFF"),
	Color("#00C0FF"),
	Color("#0FF8E7"),
	Color("#39FFBE"),
	Color("#66FF90"),
	Color("#90FF66"),
	Color("#BEFF39"),
	Color("#E7FF0F"),
	Color("#FFD300"),
	Color("#FFA300"),
	Color("#FF6F00"),
	Color("#FF3F00"),
	Color("#F60B00"),
	Color("#BB0000"),
]

const COLORBAR_WIDTH := 300.0
const COLORBAR_HEIGHT := 12.0
const COLORBAR_BOTTOM_MARGIN := 28.0
const COLORBAR_LEFT_MARGIN := 30.0
const COLORBAR_LABEL_GAP := 4.0
const COLORBAR_FONT_SIZE := 12

const NEURON_SPACING := 1.0
const BASE_NEURON_RADIUS := 0.12

const SELECTED_OUTGOING_COLOR = Color(1.0, 0.0, 0.0, 1.0)
const SELECTED_INCOMING_COLOR = Color(0.0, 0.6, 1.0, 1.0)

const SELECTION_NONE = 0
const SELECTION_BOTH = 1
const SELECTION_OUTGOING = 2
const SELECTION_INCOMING = 3


var current_brain: Dictionary = {}
var brain_dynamics := BrainDynamics.new()
var simulation_running := false
var single_step_requested = false
var random_input_running = false
var random_input_elapsed = 0.0
var random_input_rng = RandomNumberGenerator.new()

var runtime_build_time_seconds := 0.0
var runtime_excitatory := 0
var runtime_inhibitory := 0
var runtime_mean_activation_threshold := 0.0
var runtime_mean_decay_factor := 0.0
var runtime_mean_retention_factor := 0.0
var runtime_mean_hebbian_plasticity_rate := 0.0
var runtime_mean_modulatory_release_factor := 0.0
var runtime_mean_modulatory_sensitivity := 0.0
var runtime_reciprocal_directional_connections = 0
var runtime_reciprocal_connection_rate = 0.0

var load_genome_dialog: FileDialog
var generation_thread: Thread
var generation_progress_mutex := Mutex.new()
var generation_progress := 0
var displayed_generation_progress := -1
var generation_start_time := 0

var camera: Camera3D
var camera_target := Vector3.ZERO
var camera_distance := 10.0
var camera_yaw := deg_to_rad(45.0)
var camera_pitch := deg_to_rad(25.0)
var orbiting := false
var panning := false

var neuron_mesh: SphereMesh
var connection_material: StandardMaterial3D
var selected_connection_material: StandardMaterial3D
var selected_connections_instance: MeshInstance3D
var selected_neuron_index: int = -1
var selected_neuron_state: int = SELECTION_NONE

var heatmap_ranges: Dictionary = {}
var incoming_counts := PackedInt32Array()

var colorbar_container: Control
var colorbar_texture_rect: TextureRect
var colorbar_min_label: Label
var colorbar_mid_label: Label
var colorbar_max_label: Label
var selection_legend_label: Label
var selected_neuron_info_label: Label
var neuron_role_legend_label: Label


@onready var neurons_instance: MultiMeshInstance3D = $Neurons
@onready var connections_instance: MeshInstance3D = $Connections
@onready var directional_light: DirectionalLight3D = $DirectionalLight3D
@onready var world_environment: WorldEnvironment = $WorldEnvironment
@onready var load_brain_button: Button = $CanvasLayer/UI/LoadBrainButton
@onready var show_neurons: CheckButton = $CanvasLayer/UI/ShowNeurons
@onready var show_connections: CheckButton = $CanvasLayer/UI/ShowConnections
@onready var neuron_size_slider: HSlider = $CanvasLayer/UI/NeuronSizeSlider
@onready var connection_opacity_slider: HSlider = $CanvasLayer/UI/ConnectionOpacitySlider
@onready var reset_brain_button: Button = $CanvasLayer/UI/ResetBrainButton
@onready var brain_info_label: Label = $CanvasLayer/UI/BrainInfoLabel
@onready var heatmap_selector: OptionButton = $CanvasLayer/UI/HeatmapSelector
@onready var play_button: Button = $CanvasLayer/UI/Play
@onready var next_iteration_button: Button = $CanvasLayer/UI/NextIteration
@onready var fire_input_button: Button = $CanvasLayer/UI/FireInput
@onready var fire_input_option_button: OptionButton = $CanvasLayer/UI/FireInputOptionButton
@onready var fire_input_strength: HSlider = $CanvasLayer/UI/FireInputStrength
@onready var fire_random_button: Button = $CanvasLayer/UI/FireRandom
@onready var fire_random_delay: HSlider = $CanvasLayer/UI/FireRandomDelay
@onready var fire_input_strength_label: Label = $CanvasLayer/UI/FireInputStrength/Label
@onready var fire_random_delay_label: Label = $CanvasLayer/UI/FireRandomDelay/Label


func _ready() -> void:
	_setup_file_dialog()
	_setup_controls()
	_setup_camera()
	_setup_environment()
	_setup_connection_material()
	_setup_selected_connection_visualization()
	_setup_selection_legend()
	_setup_selected_neuron_info()
	_setup_neuron_role_legend()
	_setup_colorbar()

	load_brain_button.pressed.connect(_open_load_genome_dialog)
	show_neurons.toggled.connect(_on_show_neurons_toggled)
	show_connections.toggled.connect(_on_show_connections_toggled)
	neuron_size_slider.value_changed.connect(_on_neuron_size_changed)
	connection_opacity_slider.value_changed.connect(_on_connection_opacity_changed)
	reset_brain_button.pressed.connect(_reset_brain)
	heatmap_selector.item_selected.connect(_on_heatmap_selection_changed)
	play_button.pressed.connect(_on_play_pressed)
	next_iteration_button.pressed.connect(_on_next_iteration_pressed)
	fire_input_button.pressed.connect(_on_fire_input_pressed)
	fire_input_option_button.item_selected.connect(_on_fire_input_selected)
	fire_random_button.pressed.connect(_on_fire_random_pressed)
	fire_input_strength.value_changed.connect(_on_fire_input_strength_changed)
	fire_random_delay.value_changed.connect(_on_fire_random_delay_changed)

	_on_fire_input_strength_changed(fire_input_strength.value)
	_on_fire_random_delay_changed(fire_random_delay.value)

	random_input_rng.randomize()
	brain_info_label.text = ""


## Creates the native file dialog used to load the same brain configuration files as BrainExplorer.
func _setup_file_dialog() -> void:
	load_genome_dialog = FileDialog.new()
	load_genome_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	load_genome_dialog.access = FileDialog.ACCESS_FILESYSTEM
	load_genome_dialog.use_native_dialog = true
	load_genome_dialog.filters = PackedStringArray(["*.json ; JSON files"])
	load_genome_dialog.file_selected.connect(_load_genome_from_file)
	add_child(load_genome_dialog)


## Configures the scene controls with useful defaults.
func _setup_controls() -> void:
	show_neurons.button_pressed = true
	show_connections.button_pressed = true

	neuron_size_slider.min_value = 0.25
	neuron_size_slider.max_value = 3.0
	neuron_size_slider.step = 0.05
	neuron_size_slider.value = 1.0

	connection_opacity_slider.min_value = 0.02
	connection_opacity_slider.max_value = 1.0
	connection_opacity_slider.step = 0.01
	connection_opacity_slider.value = 1.0

	heatmap_selector.clear()
	for label in HEATMAP_PARAMETER_LABELS:
		heatmap_selector.add_item(label)
	heatmap_selector.select(0)
	heatmap_selector.disabled = true
	play_button.text = "Play"
	play_button.disabled = true
	next_iteration_button.disabled = true
	fire_input_button.text = "Fire Input"
	fire_input_button.disabled = true
	fire_input_option_button.clear()
	fire_input_option_button.disabled = true
	fire_random_button.text = "Fire Random"
	fire_random_button.disabled = true

	brain_info_label.size = Vector2(800.0, 720.0)
	brain_info_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART


## Creates the shared material used by every connection line.
## Connection weight is encoded by the vertex color while the opacity slider acts as a global multiplier.
func _setup_connection_material() -> void:
	connection_material = StandardMaterial3D.new()
	connection_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	connection_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	connection_material.vertex_color_use_as_albedo = true
	connection_material.albedo_color = Color(1.0, 1.0, 1.0, connection_opacity_slider.value)


## Creates the separate mesh and material used to highlight one selected neuron's connections.
func _setup_selected_connection_visualization() -> void:
	selected_connection_material = StandardMaterial3D.new()
	selected_connection_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	selected_connection_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	selected_connection_material.vertex_color_use_as_albedo = true
	selected_connection_material.albedo_color = Color(1.0, 1.0, 1.0, connection_opacity_slider.value)

	selected_connections_instance = MeshInstance3D.new()
	selected_connections_instance.name = "SelectedConnections"
	selected_connections_instance.visible = false
	add_child(selected_connections_instance)


## Creates the top-center legend shown while a neuron is selected.
func _setup_selection_legend() -> void:
	selection_legend_label = Label.new()
	selection_legend_label.name = "NeuronSelectionLegend"
	selection_legend_label.text = "Selected neuron: RED = outgoing connections (0)    BLUE = incoming connections (0)"
	selection_legend_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	selection_legend_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	selection_legend_label.set_anchors_preset(Control.PRESET_CENTER_TOP)
	selection_legend_label.offset_left = -360.0
	selection_legend_label.offset_top = 12.0
	selection_legend_label.offset_right = 360.0
	selection_legend_label.offset_bottom = 42.0
	selection_legend_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	selection_legend_label.add_theme_font_size_override("font_size", 16)
	selection_legend_label.add_theme_color_override("font_color", Color.WHITE)
	selection_legend_label.add_theme_color_override("font_outline_color", Color.BLACK)
	selection_legend_label.add_theme_constant_override("outline_size", 6)
	selection_legend_label.hide()
	$CanvasLayer.add_child(selection_legend_label)


## Creates the live diagnostic label shown below the selected-neuron connection legend.
func _setup_selected_neuron_info() -> void:
	selected_neuron_info_label = Label.new()
	selected_neuron_info_label.name = "SelectedNeuronInfo"
	selected_neuron_info_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	selected_neuron_info_label.vertical_alignment = VERTICAL_ALIGNMENT_TOP
	selected_neuron_info_label.set_anchors_preset(Control.PRESET_CENTER_TOP)
	selected_neuron_info_label.offset_left = -200.0
	selected_neuron_info_label.offset_top = 38.0
	selected_neuron_info_label.offset_right = 900.0
	selected_neuron_info_label.offset_bottom = 245.0
	selected_neuron_info_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	selected_neuron_info_label.add_theme_font_size_override("font_size", 14)
	selected_neuron_info_label.add_theme_color_override("font_color", Color.WHITE)
	selected_neuron_info_label.add_theme_color_override("font_outline_color", Color.BLACK)
	selected_neuron_info_label.add_theme_constant_override("outline_size", 5)
	selected_neuron_info_label.hide()
	$CanvasLayer.add_child(selected_neuron_info_label)


## Creates the top-center legend shown while the Neuron role heatmap is selected.
func _setup_neuron_role_legend() -> void:
	neuron_role_legend_label = Label.new()
	neuron_role_legend_label.name = "NeuronRoleLegend"
	neuron_role_legend_label.text = "Neuron role: BLUE = internal    GREEN = input    RED = output"
	neuron_role_legend_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	neuron_role_legend_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	neuron_role_legend_label.set_anchors_preset(Control.PRESET_CENTER_TOP)
	neuron_role_legend_label.offset_left = -360.0
	neuron_role_legend_label.offset_top = 55.0
	neuron_role_legend_label.offset_right = 360.0
	neuron_role_legend_label.offset_bottom = 95.0
	neuron_role_legend_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	neuron_role_legend_label.add_theme_font_size_override("font_size", 16)
	neuron_role_legend_label.add_theme_color_override("font_color", Color.WHITE)
	neuron_role_legend_label.add_theme_color_override("font_outline_color", Color.BLACK)
	neuron_role_legend_label.add_theme_constant_override("outline_size", 6)
	neuron_role_legend_label.hide()
	$CanvasLayer.add_child(neuron_role_legend_label)


## Shows the neuron-role legend only while the corresponding heatmap is selected.
func _update_neuron_role_legend() -> void:
	if neuron_role_legend_label == null or current_brain.is_empty() or heatmap_selector.selected < 0:
		if neuron_role_legend_label != null:
			neuron_role_legend_label.hide()
		return

	var parameter_index = heatmap_selector.selected

	if parameter_index >= HEATMAP_PARAMETER_KEYS.size():
		neuron_role_legend_label.hide()
		return

	if HEATMAP_PARAMETER_KEYS[parameter_index] == "neuron_role":
		neuron_role_legend_label.show()
	else:
		neuron_role_legend_label.hide()


## Creates the horizontal heatmap colorbar centered at the bottom of the viewport.
func _setup_colorbar() -> void:
	colorbar_container = Control.new()
	colorbar_container.name = "HeatmapColorbar"
	colorbar_container.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	colorbar_container.offset_left = -COLORBAR_WIDTH / 2.0
	colorbar_container.offset_top = -(COLORBAR_BOTTOM_MARGIN + COLORBAR_HEIGHT + COLORBAR_LABEL_GAP + 18.0)
	colorbar_container.offset_right = COLORBAR_WIDTH / 2.0
	colorbar_container.offset_bottom = -COLORBAR_BOTTOM_MARGIN
	colorbar_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	colorbar_container.hide()
	$CanvasLayer.add_child(colorbar_container)

	colorbar_texture_rect = TextureRect.new()
	colorbar_texture_rect.position = Vector2.ZERO
	colorbar_texture_rect.size = Vector2(COLORBAR_WIDTH, COLORBAR_HEIGHT)
	colorbar_texture_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	colorbar_texture_rect.stretch_mode = TextureRect.STRETCH_SCALE
	colorbar_texture_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	colorbar_container.add_child(colorbar_texture_rect)

	colorbar_min_label = _create_colorbar_label(HORIZONTAL_ALIGNMENT_LEFT)
	colorbar_mid_label = _create_colorbar_label(HORIZONTAL_ALIGNMENT_CENTER)
	colorbar_max_label = _create_colorbar_label(HORIZONTAL_ALIGNMENT_RIGHT)

	colorbar_container.add_child(colorbar_min_label)
	colorbar_container.add_child(colorbar_mid_label)
	colorbar_container.add_child(colorbar_max_label)

	var label_y := COLORBAR_HEIGHT + COLORBAR_LABEL_GAP
	var label_height := 18.0
	var third := COLORBAR_WIDTH / 3.0

	colorbar_min_label.position = Vector2(0.0, label_y)
	colorbar_min_label.size = Vector2(third, label_height)

	colorbar_mid_label.position = Vector2(third, label_y)
	colorbar_mid_label.size = Vector2(third, label_height)

	colorbar_max_label.position = Vector2(2.0 * third, label_y)
	colorbar_max_label.size = Vector2(third, label_height)


## Creates one numeric label used by the heatmap colorbar.
## [param alignment] determines how the value is aligned inside its third of the bar.
## Returns the configured Label.
func _create_colorbar_label(alignment: HorizontalAlignment) -> Label:
	var label := Label.new()
	label.horizontal_alignment = alignment
	label.add_theme_font_size_override("font_size", COLORBAR_FONT_SIZE)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return label


## Creates the colorbar texture for the current value range.
## Constant ranges are displayed with the palette midpoint, except for the modulatory field,
## where an exactly zero range uses the darkest palette color.
func _update_colorbar_texture(value_min: float, value_max: float, parameter_key: String) -> void:
	var width = int(COLORBAR_WIDTH)
	var height = int(COLORBAR_HEIGHT)
	var bar_image = Image.create(width, height, false, Image.FORMAT_RGBA8)
	var value_span = value_max - value_min
	var collapsed_range: bool = value_span == 0.0 if parameter_key == "modulatory_field" else is_zero_approx(value_span)

	for x in width:
		var normalized = 0.5

		if collapsed_range:
			if parameter_key == "modulatory_field" and value_min == 0.0:
				normalized = 0.0
		else:
			normalized = float(x) / maxf(1.0, float(width - 1))

		var bar_color = _heatmap_palette_color(normalized)

		for y in height:
			bar_image.set_pixel(x, y, bar_color)

	colorbar_texture_rect.texture = ImageTexture.create_from_image(bar_image)


## Updates the colorbar limits to match the currently selected 3D heatmap.
func _update_colorbar() -> void:
	if current_brain.is_empty() or heatmap_selector.selected < 0:
		colorbar_container.hide()
		return

	var parameter_index := heatmap_selector.selected

	if parameter_index >= HEATMAP_PARAMETER_KEYS.size():
		colorbar_container.hide()
		return

	var parameter_key: String = HEATMAP_PARAMETER_KEYS[parameter_index]
	var value_range: Vector2 = heatmap_ranges[parameter_key]
	var value_min = value_range.x
	var value_max = value_range.y
	var value_span = value_max - value_min
	var collapsed_range: bool = value_span == 0.0 if parameter_key == "modulatory_field" else is_zero_approx(value_span)

	_update_colorbar_texture(value_min, value_max, parameter_key)

	if collapsed_range:
		colorbar_min_label.text = ""
		colorbar_mid_label.text = _format_colorbar_value(value_min, value_min, value_max)
		colorbar_max_label.text = ""
	else:
		colorbar_min_label.text = _format_colorbar_value(value_min, value_min, value_max)
		colorbar_mid_label.text = _format_colorbar_value((value_min + value_max) / 2.0, value_min, value_max)
		colorbar_max_label.text = _format_colorbar_value(value_max, value_min, value_max)

	colorbar_container.show()


## Formats colorbar values using the same compact rules as Heatmap.gd.
func _format_colorbar_value(value: float, value_min: float, value_max: float) -> String:
	var colorbar_scale = maxf(absf(value_min), absf(value_max))
	var span = absf(value_max - value_min)

	if colorbar_scale > 0.0 and colorbar_scale < 0.001:
		return String.num_scientific(value)

	if is_equal_approx(value, round(value)):
		return str(int(round(value)))

	if colorbar_scale >= 10000.0:
		return String.num_scientific(value)
	if span >= 10.0:
		return "%.1f" % value
	if span >= 1.0:
		return "%.2f" % value
	if span >= 0.1:
		return "%.3f" % value

	return "%.4f" % value


## Creates a Camera3D when the scene does not already contain one.
func _setup_camera() -> void:
	camera = get_node_or_null("Camera3D") as Camera3D

	if camera == null:
		camera = Camera3D.new()
		camera.name = "Camera3D"
		add_child(camera)

	camera.current = true
	camera.fov = 60.0
	camera.near = 0.05
	camera.far = 100000.0
	_update_camera_transform()


## Configures simple lighting and a dark background for the 3D graph.
func _setup_environment() -> void:
	directional_light.rotation_degrees = Vector3(-50.0, -35.0, 0.0)
	directional_light.light_energy = 1.15
	directional_light.shadow_enabled = false

	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.025, 0.03, 0.045, 1.0)
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color(0.55, 0.60, 0.75, 1.0)
	environment.ambient_light_energy = 0.75
	world_environment.environment = environment


## Opens the operating system file chooser.
func _open_load_genome_dialog() -> void:
	if _is_generating():
		return

	load_genome_dialog.popup_centered()


## Loads and validates a BrainExplorer JSON configuration, then generates the corresponding brain.
## [param path] is the configuration file selected in the open dialog.
func _load_genome_from_file(path: String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)

	if file == null:
		brain_info_label.text = "Could not load brain configuration.\n%s" % error_string(FileAccess.get_open_error())
		return

	var data = JSON.parse_string(file.get_as_text())

	_migrate_legacy_retention_factor_key(data)

	if not _is_valid_genome_file(data):
		brain_info_label.text = "Invalid brain configuration file."
		return

	var external_parameters: Dictionary = data["external_parameters"]
	var genome_data: Dictionary = data["genome"]

	var size := Vector3i(
		int(external_parameters["size_x"]),
		int(external_parameters["size_y"]),
		int(external_parameters["size_z"])
	)
	var num_inputs := int(external_parameters["num_inputs"])
	var num_outputs := int(external_parameters["num_outputs"])
	var genome := _genome_from_data(genome_data)

	_start_brain_generation(genome, size, num_inputs, num_outputs)


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


## Returns true when the parsed JSON follows the same configuration format accepted by BrainExplorer.
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

	var size_x := int(external_parameters["size_x"])
	var size_y := int(external_parameters["size_y"])
	var size_z := int(external_parameters["size_z"])
	var num_inputs := int(external_parameters["num_inputs"])
	var num_outputs := int(external_parameters["num_outputs"])

	if size_x < 2 or size_y < 2 or size_z < 2:
		return false

	if num_inputs + num_outputs > size_x * size_y * size_z:
		return false

	return true


## Returns only characters contained in allowed_characters.
func _filter_characters(text: String, allowed_characters: String) -> String:
	var filtered := ""

	for character in text:
		if allowed_characters.contains(character):
			filtered += character

	return filtered


## Creates a Genome object from the gene strings stored in a BrainExplorer configuration file.
func _genome_from_data(genome_data: Dictionary) -> Genome:
	var genome := Genome.new()

	for gene_name in GENE_ORDER:
		genome.set(gene_name, _gene_from_text(genome_data[gene_name]))

	return genome


## Converts a gene string into the integer array expected by Genome.
func _gene_from_text(text: String) -> Array[int]:
	var gene: Array[int] = []

	for character in text:
		gene.append(int(character))

	return gene


## Starts BrainBuilder outside the main thread so the viewer remains responsive.
func _start_brain_generation(genome: Genome, size: Vector3i, num_inputs: int, num_outputs: int) -> void:
	_pause_simulation()
	_stop_random_input()
	brain_dynamics.clear()
	play_button.disabled = true
	next_iteration_button.disabled = true

	if _is_generating():
		return

	_clear_visualization()

	generation_progress_mutex.lock()
	generation_progress = 0
	generation_progress_mutex.unlock()

	displayed_generation_progress = -1
	generation_start_time = Time.get_ticks_msec()
	load_brain_button.disabled = true
	brain_info_label.text = "Generating brain... 0%"

	generation_thread = Thread.new()
	var error := generation_thread.start(_build_brain_thread.bind(genome, size, num_inputs, num_outputs))

	if error != OK:
		generation_thread = null
		load_brain_button.disabled = false
		brain_info_label.text = "Could not start brain generation.\n%s" % error_string(error)


## Worker thread entry point. It must not access the scene tree or UI nodes.
func _build_brain_thread(genome: Genome, size: Vector3i, num_inputs: int, num_outputs: int) -> Dictionary:
	var builder := BrainBuilder.new()
	return builder.build(genome, size, num_inputs, num_outputs, _set_generation_progress_from_thread)


## Receives BrainBuilder progress from the worker thread.
func _set_generation_progress_from_thread(percent: int) -> void:
	generation_progress_mutex.lock()
	generation_progress = percent
	generation_progress_mutex.unlock()


## Updates generation progress and transfers the completed brain back to the main thread.
func _process(delta: float) -> void:
	if _is_generating():
		generation_progress_mutex.lock()
		var progress := generation_progress
		generation_progress_mutex.unlock()

		if progress != displayed_generation_progress:
			displayed_generation_progress = progress
			brain_info_label.text = "Generating brain... %d%%" % progress

		if not generation_thread.is_alive():
			current_brain = generation_thread.wait_to_finish()
			generation_thread = null
			load_brain_button.disabled = false

			var elapsed_seconds := (Time.get_ticks_msec() - generation_start_time) / 1000.0

			if current_brain.is_empty() or not bool(current_brain.get("valid", false)):
				brain_info_label.text = (
					"Brain generation failed.\n"
					+ "Isolated neurons were found after developmental rescue.\n"
					+ "Generation time: %.3f s" % elapsed_seconds
				)
				return

			runtime_build_time_seconds = elapsed_seconds
			brain_dynamics.set_brain(current_brain)
			_setup_input_controls()
			_prepare_runtime_summary(current_brain)
			_update_reciprocal_connection_statistics()
			_build_visualization(current_brain)
			play_button.disabled = false
			next_iteration_button.disabled = false
			_update_runtime_status()

		return

	if simulation_running or single_step_requested:
		if simulation_running:
			_update_random_input(delta)

		var removed_connections := brain_dynamics.advance(delta)
		var added_connections = brain_dynamics.added_connections_last_iteration
		var topology_changed = removed_connections > 0 or added_connections > 0
		single_step_requested = false

		if topology_changed:
			_update_reciprocal_connection_statistics()

		_update_runtime_status()

		if show_connections.button_pressed:
			_build_connections(current_brain)

		var selected_heatmap = HEATMAP_PARAMETER_KEYS[heatmap_selector.selected]

		if selected_heatmap == "modulatory_field":
			_update_modulatory_field_heatmap_range()

		if selected_heatmap == "signal" or selected_heatmap == "signal_0_5" or selected_heatmap == "signal_0_1" or selected_heatmap == "firing" or selected_heatmap == "activity_trace" or selected_heatmap == "fatigue" or selected_heatmap == "modulatory_release" or selected_heatmap == "modulatory_field" or selected_heatmap == "effective_modulation":
			_update_heatmap_colors()

		if topology_changed:
			_update_incoming_heatmap_metadata()

			if selected_neuron_index >= 0:
				_build_selected_connections()

			if selected_heatmap == "incoming_connections":
				_update_heatmap_colors()


## Returns true while the worker thread is still owned by this scene.
func _is_generating() -> bool:
	return generation_thread != null and generation_thread.is_started()


## Removes the currently displayed brain.
func _clear_visualization() -> void:
	_pause_simulation()
	_clear_neuron_selection()
	brain_dynamics.clear()
	current_brain = {}
	runtime_build_time_seconds = 0.0
	runtime_excitatory = 0
	runtime_inhibitory = 0
	runtime_mean_activation_threshold = 0.0
	runtime_mean_decay_factor = 0.0
	runtime_mean_retention_factor = 0.0
	runtime_mean_hebbian_plasticity_rate = 0.0
	runtime_mean_modulatory_release_factor = 0.0
	runtime_mean_modulatory_sensitivity = 0.0
	runtime_reciprocal_directional_connections = 0
	runtime_reciprocal_connection_rate = 0.0
	neurons_instance.multimesh = null
	connections_instance.mesh = null

	if selected_connections_instance != null:
		selected_connections_instance.mesh = null

	heatmap_ranges.clear()
	incoming_counts = PackedInt32Array()
	heatmap_selector.select(0)
	heatmap_selector.disabled = true
	play_button.disabled = true
	next_iteration_button.disabled = true
	fire_input_button.text = "Fire Input"
	fire_input_button.disabled = true
	fire_input_option_button.clear()
	fire_input_option_button.disabled = true
	_stop_random_input()
	fire_random_button.disabled = true

	if colorbar_container != null:
		colorbar_container.hide()

	if neuron_role_legend_label != null:
		neuron_role_legend_label.hide()


## Creates the neuron MultiMesh and the connection ArrayMesh.
func _build_visualization(brain: Dictionary) -> void:
	_build_heatmap_metadata(brain)
	_build_neurons(brain)
	_build_connections(brain)
	_update_heatmap_colors()
	heatmap_selector.disabled = false
	_reset_camera()


## Builds one low polygon sphere instance for every neuron.
func _build_neurons(brain: Dictionary) -> void:
	var neurons: Array = brain["neurons"]

	neuron_mesh = SphereMesh.new()
	neuron_mesh.radial_segments = 8
	neuron_mesh.rings = 4
	_update_neuron_mesh_size(neuron_size_slider.value)

	var neuron_material := StandardMaterial3D.new()
	neuron_material.vertex_color_use_as_albedo = true
	neuron_material.albedo_color = Color.WHITE
	neuron_material.roughness = 0.65
	neuron_mesh.material = neuron_material

	var multimesh := MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.use_colors = true
	multimesh.instance_count = neurons.size()
	multimesh.mesh = neuron_mesh

	for index in neurons.size():
		var neuron: Dictionary = neurons[index]
		var world_position := _brain_to_world(neuron["position"], brain["size"])
		multimesh.set_instance_transform(index, Transform3D(Basis.IDENTITY, world_position))
		multimesh.set_instance_color(index, Color.WHITE)

	neurons_instance.multimesh = multimesh
	neurons_instance.visible = show_neurons.button_pressed


## Builds all directional connections as line segments in a single ArrayMesh.
func _build_connections(brain: Dictionary) -> void:
	var neurons: Array = brain["neurons"]
	var total_connections := 0

	for neuron in neurons:
		total_connections += neuron["connections"].size()

	var vertices := PackedVector3Array()
	var colors := PackedColorArray()
	vertices.resize(total_connections * 2)
	colors.resize(total_connections * 2)

	var vertex_index := 0

	for neuron in neurons:
		var source_position := _brain_to_world(neuron["position"], brain["size"])

		for connection in neuron["connections"]:
			var weight: float = connection["weight"]
			var connection_color = _heatmap_palette_color(clampf(weight, 0.0, 1.0))

			vertices[vertex_index] = source_position
			vertices[vertex_index + 1] = _brain_to_world(connection["target"], brain["size"])
			colors[vertex_index] = connection_color
			colors[vertex_index + 1] = connection_color
			vertex_index += 2

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_COLOR] = colors

	var mesh := connections_instance.mesh as ArrayMesh

	if mesh == null:
		mesh = ArrayMesh.new()
	else:
		mesh.clear_surfaces()

	if not vertices.is_empty():
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINES, arrays)
		mesh.surface_set_material(0, connection_material)

	connections_instance.mesh = mesh
	connections_instance.visible = show_connections.button_pressed and selected_neuron_index < 0


## Builds the directional connections associated with the selected neuron according to the current selection state.
## State 1 shows outgoing connections in saturated red and incoming connections in saturated blue.
## State 2 shows only outgoing connections. State 3 shows only incoming connections.
func _build_selected_connections() -> void:
	if selected_neuron_index < 0 or selected_neuron_state == SELECTION_NONE or current_brain.is_empty():
		selected_connections_instance.mesh = null
		selected_connections_instance.visible = false
		return

	var neurons: Array = current_brain["neurons"]

	if selected_neuron_index >= neurons.size():
		_clear_neuron_selection()
		return

	var selected_neuron: Dictionary = neurons[selected_neuron_index]
	var selected_position = _brain_to_world(selected_neuron["position"], current_brain["size"])
	var show_outgoing: bool = selected_neuron_state == SELECTION_BOTH or selected_neuron_state == SELECTION_OUTGOING
	var show_incoming: bool = selected_neuron_state == SELECTION_BOTH or selected_neuron_state == SELECTION_INCOMING
	var outgoing_count: int = selected_neuron["connections"].size() if show_outgoing else 0
	var incoming_count: int = 0

	if show_incoming:
		for source_index in neurons.size():
			if source_index == selected_neuron_index:
				continue

			for connection in neurons[source_index]["connections"]:
				if int(connection.get("target_index", -1)) == selected_neuron_index:
					incoming_count += 1

	var vertices = PackedVector3Array()
	var colors = PackedColorArray()
	vertices.resize((outgoing_count + incoming_count) * 2)
	colors.resize((outgoing_count + incoming_count) * 2)

	var vertex_index: int = 0

	if show_outgoing:
		for connection in selected_neuron["connections"]:
			vertices[vertex_index] = selected_position
			vertices[vertex_index + 1] = _brain_to_world(connection["target"], current_brain["size"])
			colors[vertex_index] = SELECTED_OUTGOING_COLOR
			colors[vertex_index + 1] = SELECTED_OUTGOING_COLOR
			vertex_index += 2

	if show_incoming:
		for source_index in neurons.size():
			if source_index == selected_neuron_index:
				continue

			var source: Dictionary = neurons[source_index]

			for connection in source["connections"]:
				if int(connection.get("target_index", -1)) != selected_neuron_index:
					continue

				vertices[vertex_index] = _brain_to_world(source["position"], current_brain["size"])
				vertices[vertex_index + 1] = selected_position
				colors[vertex_index] = SELECTED_INCOMING_COLOR
				colors[vertex_index + 1] = SELECTED_INCOMING_COLOR
				vertex_index += 2

	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_COLOR] = colors

	var mesh = selected_connections_instance.mesh as ArrayMesh

	if mesh == null:
		mesh = ArrayMesh.new()
	else:
		mesh.clear_surfaces()

	if not vertices.is_empty():
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINES, arrays)
		mesh.surface_set_material(0, selected_connection_material)

	selected_connections_instance.mesh = mesh
	selected_connections_instance.visible = true
	connections_instance.visible = false
	_update_selection_legend()


## Updates the top legend with the total outgoing and incoming connection counts of the selected neuron.
func _update_selection_legend() -> void:
	if selection_legend_label == null or selected_neuron_index < 0 or current_brain.is_empty():
		return

	var neurons: Array = current_brain["neurons"]
	var selected_neuron: Dictionary = neurons[selected_neuron_index]
	var outgoing_count = selected_neuron["connections"].size()
	var incoming_count = brain_dynamics.incoming_connection_counts[selected_neuron_index]

	selection_legend_label.text = (
		"Selected neuron: RED = outgoing connections (%d)    BLUE = incoming connections (%d)"
		% [outgoing_count, incoming_count]
	)
	selection_legend_label.show()
	_update_selected_neuron_info()


## Updates every neuron-level diagnostic value shown for the selected neuron.
func _update_selected_neuron_info() -> void:
	if selected_neuron_info_label == null or selected_neuron_index < 0 or current_brain.is_empty():
		return

	var neurons: Array = current_brain["neurons"]

	if selected_neuron_index >= neurons.size():
		return

	var neuron: Dictionary = neurons[selected_neuron_index]
	var neuron_position: Vector3i = neuron["position"]
	var neuron_type = "Internal"

	if neuron["input"]:
		neuron_type = "Input"
	elif neuron["output"]:
		neuron_type = "Output"

	var state = brain_dynamics.neuron_states_used[selected_neuron_index]
	var normalized_signal = brain_dynamics.normalized_signals[selected_neuron_index]
	var incoming_signal = brain_dynamics.incoming_signals[selected_neuron_index]
	var firing = brain_dynamics.firing_states[selected_neuron_index]
	var fatigue = brain_dynamics.fatigues_used[selected_neuron_index]
	var activity_trace = brain_dynamics.activity_traces[selected_neuron_index]
	var activation_threshold = float(neuron["activation_threshold"])
	var effective_threshold = brain_dynamics.effective_thresholds_used[selected_neuron_index]
	var decay_factor = float(neuron["decay_factor"])
	var decay_rate = brain_dynamics.decay_rates[selected_neuron_index]
	var retention_factor = float(neuron["retention_factor"])
	var hebbian_rate = float(neuron["hebbian_plasticity_rate"])
	var polarity = int(neuron["polarity_factor"])
	var connection_energy = float(neuron["connections_value"])
	var modulatory_release_factor = float(neuron["modulatory_release_factor"])
	var modulatory_sensitivity = float(neuron["modulatory_sensitivity"])
	var modulatory_release = brain_dynamics.modulatory_releases[selected_neuron_index]
	var modulatory_field = brain_dynamics.modulatory_fields[selected_neuron_index]
	var effective_modulation = brain_dynamics.effective_modulations[selected_neuron_index]
	#var outgoing_count = neuron["connections"].size()
	#var incoming_count = brain_dynamics.incoming_connection_counts[selected_neuron_index]

	var role_specific = ""

	if neuron["input"]:
		var input_slot = brain_dynamics.input_neuron_indices.find(selected_neuron_index)
		var input_used = 0.0 if input_slot < 0 else brain_dynamics.input_signals_used[input_slot]
		role_specific = "\nInput used: %.3f    Input influence psi: %.3f    Minimum connection weight: %.3f" % [
			input_used,
			brain_dynamics.input_gain,
			brain_dynamics.input_min_weight
		]
	elif neuron["output"]:
		var output_slot = brain_dynamics.output_slot_by_neuron_index[selected_neuron_index]
		var output_signal = 0.0 if output_slot < 0 else brain_dynamics.output_signals[output_slot]
		role_specific = "\nOutput signal: %.3f" % output_signal

	selected_neuron_info_label.text = (
		"Neuron type: %s    Index: %d    Position: (%d, %d, %d)    Activity trace R_t: %.3f\n"
		+ "State s: %.3f    Normalized signal: %.3f    Incoming signal for next state: %.3f\n"
		+ "Firing: %.0f    Activation threshold: %.3f    Effective threshold: %.3f    Fatigue: %.3f\n"
		+ "Retention factor: %.3f    Decay factor: %.3f    Decay: %.6f\n"
		+ "Hebbian plasticity rate: %.3f    Polarity factor: %d    Connection energy: %.3f\n"
		+ "Modulatory release factor: %.3f    Modulatory sensitivity: %.3f\n"
		+ "Modulatory release: %.3f    Modulatory field: %.3f    Effective modulation: %.3f\n"
		+ role_specific
	) % [
		neuron_type,
		selected_neuron_index,
		neuron_position.x,
		neuron_position.y,
		neuron_position.z,
		activity_trace,
		state,
		normalized_signal,
		incoming_signal,
		firing,
		activation_threshold,
		effective_threshold,
		fatigue,
		retention_factor,
		decay_factor,
		decay_rate,
		hebbian_rate,
		polarity,
		connection_energy,
		modulatory_release_factor,
		modulatory_sensitivity,
		modulatory_release,
		modulatory_field,
		effective_modulation,
	]

	selected_neuron_info_label.show()


## Selects the nearest visible neuron intersected by the mouse ray.
## Repeated clicks on the selected neuron cycle through both, outgoing only, incoming only, and normal view.
func _select_neuron_at_screen_position(mouse_position: Vector2) -> void:
	if current_brain.is_empty() or not show_neurons.button_pressed or neurons_instance.multimesh == null:
		return

	var ray_origin = camera.project_ray_origin(mouse_position)
	var ray_direction = camera.project_ray_normal(mouse_position).normalized()
	var local_radius = BASE_NEURON_RADIUS * neuron_size_slider.value
	var instance_scale = neurons_instance.global_transform.basis.get_scale()
	var selection_radius = local_radius * maxf(absf(instance_scale.x), maxf(absf(instance_scale.y), absf(instance_scale.z)))
	var radius_squared = selection_radius * selection_radius
	var best_index: int = -1
	var best_hit_distance = INF
	var neurons: Array = current_brain["neurons"]

	for neuron_index in neurons.size():
		var local_position = _brain_to_world(neurons[neuron_index]["position"], current_brain["size"])
		var world_position = neurons_instance.to_global(local_position)
		var to_neuron = world_position - ray_origin
		var projected_distance = to_neuron.dot(ray_direction)

		if projected_distance < 0.0:
			continue

		var distance_squared = to_neuron.length_squared() - projected_distance * projected_distance

		if distance_squared > radius_squared:
			continue

		var hit_distance = projected_distance - sqrt(maxf(0.0, radius_squared - distance_squared))

		if hit_distance < best_hit_distance:
			best_hit_distance = hit_distance
			best_index = neuron_index

	if best_index < 0:
		return

	if best_index != selected_neuron_index:
		selected_neuron_index = best_index
		selected_neuron_state = SELECTION_BOTH
		_build_selected_connections()
		return

	if selected_neuron_state == SELECTION_BOTH:
		selected_neuron_state = SELECTION_OUTGOING
		_build_selected_connections()
	elif selected_neuron_state == SELECTION_OUTGOING:
		selected_neuron_state = SELECTION_INCOMING
		_build_selected_connections()
	else:
		_clear_neuron_selection()


## Clears the selected neuron and restores the normal connection visibility state.
func _clear_neuron_selection() -> void:
	selected_neuron_index = -1
	selected_neuron_state = SELECTION_NONE

	if selected_connections_instance != null:
		selected_connections_instance.mesh = null
		selected_connections_instance.visible = false

	if selection_legend_label != null:
		selection_legend_label.hide()

	if selected_neuron_info_label != null:
		selected_neuron_info_label.hide()

	if connections_instance != null:
		connections_instance.visible = show_connections.button_pressed and not current_brain.is_empty()


## Converts an integer tensor coordinate into a centered Godot world coordinate.
func _brain_to_world(brain_position: Vector3i, size: Vector3i) -> Vector3:
	return Vector3(
		(float(brain_position.x) - 0.5 * float(size.x - 1)) * NEURON_SPACING,
		(float(brain_position.y) - 0.5 * float(size.y - 1)) * NEURON_SPACING,
		(float(brain_position.z) - 0.5 * float(size.z - 1)) * NEURON_SPACING
	)


## Builds the global value ranges used by the 3D heatmap and caches incoming degrees.
## [param brain] contains the brain structure and neuron properties.
## Returns nothing.
func _build_heatmap_metadata(brain: Dictionary) -> void:
	var I: int = brain["size"].x
	var J: int = brain["size"].y
	var IJ: int = I * J
	var total_neurons: int = brain["neurons"].size()

	incoming_counts = PackedInt32Array()
	incoming_counts.resize(total_neurons)

	for neuron in brain["neurons"]:
		for connection in neuron["connections"]:
			var target: Vector3i = connection["target"]
			var target_index := target.x + I * target.y + IJ * target.z
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
		var spatial_index := pos.x + I * pos.y + IJ * pos.z

		var activation: float = neuron["activation_threshold"]
		var decay: float = neuron["decay_factor"]
		var retention: float = neuron["retention_factor"]
		var hebbian: float = neuron["hebbian_plasticity_rate"]
		var incoming: float = incoming_counts[spatial_index]

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

	heatmap_ranges = {
		"activation_threshold": Vector2(activation_min, activation_max),
		"signal": Vector2(0.0, 1.0),
		"signal_0_5": Vector2(0.0, 0.5),
		"signal_0_1": Vector2(0.0, 0.1),
		"firing": Vector2(0.0, 1.0),
		"activity_trace": Vector2(0.0, 1.0),
		"fatigue": Vector2(0.0, float(brain["refractory_strength"])),
		"decay_factor": Vector2(decay_min, decay_max),
		"retention_factor": Vector2(retention_min, retention_max),
		"hebbian_plasticity_rate": Vector2(hebbian_min, hebbian_max),
		"modulatory_release_factor": Vector2(0.0, 1.0),
		"modulatory_sensitivity": Vector2(-1.0, 1.0),
		"modulatory_release": Vector2(0.0, 1.0),
		"modulatory_field": Vector2(0.0, 2.0),
		"effective_modulation": Vector2(-1.0, 1.0),
		"polarity_factor": Vector2(-1.0, 1.0),
		"connection_energy": Vector2(-1.0, 1.0),
		"neuron_role": Vector2(-1.0, 1.0),
		"incoming_connections": Vector2(incoming_min, incoming_max)
	}


## Keeps the modulatory-field heatmap range fixed between 0 and 2.
func _update_modulatory_field_heatmap_range() -> void:
	heatmap_ranges["modulatory_field"] = Vector2(0.0, 2.0)


## Refreshes the incoming connection counts after the live topology changes.
func _update_incoming_heatmap_metadata() -> void:
	var I: int = current_brain["size"].x
	var J: int = current_brain["size"].y
	var IJ: int = I * J
	var total_neurons: int = current_brain["neurons"].size()

	incoming_counts = PackedInt32Array()
	incoming_counts.resize(total_neurons)

	for neuron in current_brain["neurons"]:
		for connection in neuron["connections"]:
			var target: Vector3i = connection["target"]
			incoming_counts[target.x + I * target.y + IJ * target.z] += 1

	var incoming_min := INF
	var incoming_max := -INF

	for neuron in current_brain["neurons"]:
		var pos: Vector3i = neuron["position"]
		var spatial_index := pos.x + I * pos.y + IJ * pos.z
		var incoming: float = incoming_counts[spatial_index]
		incoming_min = minf(incoming_min, incoming)
		incoming_max = maxf(incoming_max, incoming)

	var previous_range: Vector2 = heatmap_ranges["incoming_connections"]
	incoming_min = minf(incoming_min, previous_range.x)
	incoming_max = maxf(incoming_max, previous_range.y)
	heatmap_ranges["incoming_connections"] = Vector2(incoming_min, incoming_max)


## Recolors every neuron using the heatmap parameter currently selected in the UI.
func _update_heatmap_colors() -> void:
	if current_brain.is_empty() or neurons_instance.multimesh == null:
		return

	var parameter_index := heatmap_selector.selected

	if parameter_index < 0 or parameter_index >= HEATMAP_PARAMETER_KEYS.size():
		return

	var parameter_key: String = HEATMAP_PARAMETER_KEYS[parameter_index]
	var value_range: Vector2 = heatmap_ranges[parameter_key]
	var value_span := value_range.y - value_range.x
	var I: int = current_brain["size"].x
	var J: int = current_brain["size"].y
	var IJ: int = I * J
	var neurons: Array = current_brain["neurons"]
	var multimesh := neurons_instance.multimesh

	for index in neurons.size():
		var neuron: Dictionary = neurons[index]
		var pos: Vector3i = neuron["position"]
		var spatial_index := pos.x + I * pos.y + IJ * pos.z
		var value = _heatmap_value(neuron, parameter_key, index, spatial_index)
		var normalized = 0.5
		var collapsed_range: bool = value_span == 0.0 if parameter_key == "modulatory_field" else is_zero_approx(value_span)

		if collapsed_range:
			if parameter_key == "modulatory_field" and value_range.x == 0.0:
				normalized = 0.0
		else:
			normalized = (value - value_range.x) / value_span

		multimesh.set_instance_color(index, _heatmap_palette_color(clampf(normalized, 0.0, 1.0)))

	_update_colorbar()
	_update_neuron_role_legend()


## Returns the scalar value represented by one neuron for the requested heatmap parameter.
func _heatmap_value(neuron: Dictionary, parameter_key: String, neuron_index: int, spatial_index: int) -> float:
	match parameter_key:
		"activation_threshold":
			return neuron["activation_threshold"]
		"signal", "signal_0_5", "signal_0_1":
			return brain_dynamics.normalized_signals[neuron_index]
		"firing":
			return brain_dynamics.firing_states[neuron_index]
		"activity_trace":
			return brain_dynamics.activity_traces[neuron_index]
		"fatigue":
			return brain_dynamics.fatigues[neuron_index]
		"decay_factor":
			return neuron["decay_factor"]
		"retention_factor":
			return neuron["retention_factor"]
		"hebbian_plasticity_rate":
			return neuron["hebbian_plasticity_rate"]
		"modulatory_release_factor":
			return neuron["modulatory_release_factor"]
		"modulatory_sensitivity":
			return neuron["modulatory_sensitivity"]
		"modulatory_release":
			return brain_dynamics.modulatory_releases[neuron_index]
		"modulatory_field":
			return brain_dynamics.modulatory_fields[neuron_index]
		"effective_modulation":
			return brain_dynamics.effective_modulations[neuron_index]
		"polarity_factor":
			return neuron["polarity_factor"]
		"connection_energy":
			return neuron["connections_value"]
		"neuron_role":
			if neuron["input"]:
				return 0.0
			if neuron["output"]:
				return 1.0
			return -1.0
		"incoming_connections":
			return incoming_counts[spatial_index]
		_:
			return 0.0


## Returns the same interpolated palette color used by Heatmap.gd.
func _heatmap_palette_color(value: float) -> Color:
	if HEATMAP_PALETTE.size() == 1:
		return HEATMAP_PALETTE[0]

	var scaled := value * (HEATMAP_PALETTE.size() - 1)
	var left := floori(scaled)
	var right := mini(left + 1, HEATMAP_PALETTE.size() - 1)
	var fraction := scaled - left

	return HEATMAP_PALETTE[left].lerp(HEATMAP_PALETTE[right], fraction)


## Recolors the 3D neuron field when the selected heatmap parameter changes.
func _on_heatmap_selection_changed(_index: int) -> void:
	if not current_brain.is_empty() and HEATMAP_PARAMETER_KEYS[heatmap_selector.selected] == "modulatory_field":
		_update_modulatory_field_heatmap_range()

	_update_heatmap_colors()


## Shows or hides every neuron instance.
func _on_show_neurons_toggled(pressed: bool) -> void:
	neurons_instance.visible = pressed


## Shows or hides the connection mesh.
func _on_show_connections_toggled(pressed: bool) -> void:
	if pressed and not current_brain.is_empty():
		_build_connections(current_brain)

	connections_instance.visible = pressed and selected_neuron_index < 0


## Changes the shared sphere mesh size without moving neuron positions.
func _on_neuron_size_changed(value: float) -> void:
	_update_neuron_mesh_size(value)


## Applies a new radius to the shared neuron mesh.
func _update_neuron_mesh_size(scale_factor: float) -> void:
	if neuron_mesh == null:
		return

	var radius := BASE_NEURON_RADIUS * scale_factor
	neuron_mesh.radius = radius
	neuron_mesh.height = 2.0 * radius


## Changes connection transparency without rebuilding the graph.
func _on_connection_opacity_changed(value: float) -> void:
	var color = connection_material.albedo_color
	color.a = value
	connection_material.albedo_color = color

	if selected_connection_material != null:
		var selected_color = selected_connection_material.albedo_color
		selected_color.a = value
		selected_connection_material.albedo_color = selected_color


## Calculates the static values of the BrainExplorer summary once for the current brain.
func _prepare_runtime_summary(brain: Dictionary) -> void:
	var neurons: Array = brain["neurons"]
	var total_neurons := neurons.size()
	var threshold_sum := 0.0
	var decay_sum := 0.0
	var retention_sum := 0.0
	var hebbian_sum := 0.0
	var modulatory_release_sum := 0.0
	var modulatory_sensitivity_sum := 0.0

	runtime_excitatory = 0
	runtime_inhibitory = 0

	for neuron in neurons:
		threshold_sum += neuron["activation_threshold"]
		decay_sum += neuron["decay_factor"]
		retention_sum += neuron["retention_factor"]
		hebbian_sum += neuron["hebbian_plasticity_rate"]
		modulatory_release_sum += neuron["modulatory_release_factor"]
		modulatory_sensitivity_sum += neuron["modulatory_sensitivity"]

		if neuron["polarity_factor"] > 0:
			runtime_excitatory += 1
		else:
			runtime_inhibitory += 1

	var neuron_denominator = max(1, total_neurons)
	runtime_mean_activation_threshold = threshold_sum / neuron_denominator
	runtime_mean_decay_factor = decay_sum / neuron_denominator
	runtime_mean_retention_factor = retention_sum / neuron_denominator
	runtime_mean_hebbian_plasticity_rate = hebbian_sum / neuron_denominator
	runtime_mean_modulatory_release_factor = modulatory_release_sum / neuron_denominator
	runtime_mean_modulatory_sensitivity = modulatory_sensitivity_sum / neuron_denominator


## Recalculates the fraction of directional connections whose reverse connection also exists.
## Each direction is counted separately, so one reciprocal pair contributes two reciprocal directional connections.
func _update_reciprocal_connection_statistics() -> void:
	var neurons: Array = current_brain["neurons"]
	var neuron_count = neurons.size()
	var connection_keys: Dictionary = {}
	var total_connections = 0

	for source_index in neuron_count:
		for connection in neurons[source_index]["connections"]:
			var target_index = int(connection["target_index"])
			connection_keys[source_index * neuron_count + target_index] = true
			total_connections += 1

	var reciprocal_connections = 0

	for source_index in neuron_count:
		for connection in neurons[source_index]["connections"]:
			var target_index = int(connection["target_index"])

			if connection_keys.has(target_index * neuron_count + source_index):
				reciprocal_connections += 1

	runtime_reciprocal_directional_connections = reciprocal_connections
	runtime_reciprocal_connection_rate = (
		0.0
		if total_connections == 0
		else 100.0 * reciprocal_connections / total_connections
	)


## Updates the shared label with the BrainExplorer summary plus live runtime information.
func _update_runtime_status() -> void:
	var total_neurons: int = current_brain["neurons"].size()
	var neuron_denominator = max(1, total_neurons)

	brain_info_label.text = (
		"Dimensions: %d x %d x %d\n" % [current_brain["size"].x, current_brain["size"].y, current_brain["size"].z]
		+ "Beta: %.3f\n" % current_brain["beta"]
		+ "Neurons: %d\n" % total_neurons
		+ "Maximum connections per neuron: %d\n" % current_brain["max_connections"]
		+ "Input influence psi: %.3f\n" % current_brain["input_gain"]
		+ "Existing directional connections: %d\n" % brain_dynamics.total_connections
		+ "Reciprocal directional connections: %d (%.2f%%)\n" % [runtime_reciprocal_directional_connections, runtime_reciprocal_connection_rate]
		+ "New connections formed by structural plasticity: %d\n" % brain_dynamics.total_structural_plasticity_connections_formed
		+ "New connections formed by reconnection: %d\n" % brain_dynamics.total_homeostatic_connections_formed
		+ "Connections destroyed: %d\n" % brain_dynamics.total_connections_destroyed
		+ "Mean outgoing connections: %d\n" % brain_dynamics.mean_outgoing_connections
		+ "Outgoing degree range: %d - %d\n" % [brain_dynamics.min_out_degree, brain_dynamics.max_out_degree]
		+ "Excitatory neurons: %d (%.1f%%)\n" % [runtime_excitatory, 100.0 * runtime_excitatory / neuron_denominator]
		+ "Inhibitory neurons: %d (%.1f%%)\n\n" % [runtime_inhibitory, 100.0 * runtime_inhibitory / neuron_denominator]
		+ "Mean signal: %.3f\n" % brain_dynamics.mean_signal
		+ "Mean activation threshold: %.3f\n" % runtime_mean_activation_threshold
		+ "Mean decay factor: %.3f\n" % runtime_mean_decay_factor
		+ "Mean retention factor: %.3f\n" % runtime_mean_retention_factor
		+ "Mean Hebbian plasticity rate: %.3f\n" % runtime_mean_hebbian_plasticity_rate
		+ "Mean modulatory release factor: %.3f\n" % runtime_mean_modulatory_release_factor
		+ "Mean modulatory sensitivity: %.3f\n" % runtime_mean_modulatory_sensitivity
		+ "Mean modulatory release: %.3f\n" % brain_dynamics.mean_modulatory_release
		+ "Mean modulatory field: %.3f\n" % brain_dynamics.mean_modulatory_field
		+ "Mean effective modulation: %.3f\n" % brain_dynamics.mean_effective_modulation
		+ "Mean eligibility trace: %.3f\n" % brain_dynamics.mean_eligibility_trace
		+ "Mean connection weight: %.3f\n\n" % brain_dynamics.mean_weight
		+ "Maximum fatigue: %.3f\n" % current_brain["refractory_strength"]
		+ "Exponential factor: %.3f\n" % current_brain["exponential_factor"]
		+ "Modulatory persistence lambda: %.3f\n" % current_brain["lambda"]
		+ "Modulatory spread nu: %.3f\n" % current_brain["nu"]
		+ "Trace persistence mu: %.3f\n" % current_brain["mu"]
		+ "Structural plasticity: %.3f\n" % current_brain["structural_plasticity"]
		+ "Build time: %.3f s\n" % runtime_build_time_seconds
		+ "Iteration: %d / " % brain_dynamics.iteration_count
		+ "Elapsed time: %.3f s" % (brain_dynamics.elapsed_time_seconds)
	)

	if selected_neuron_index >= 0:
		_update_selected_neuron_info()


## Starts or pauses the brain dynamics without disabling the viewer controls.
func _on_play_pressed() -> void:
	if simulation_running:
		_pause_simulation()
	else:
		single_step_requested = false
		simulation_running = true
		play_button.text = "Pause"


## Stops continuous execution on the current iteration.
## When already paused, requests exactly one iteration on the next process frame.
func _on_next_iteration_pressed() -> void:
	if current_brain.is_empty():
		return

	if simulation_running:
		_pause_simulation()
		return

	single_step_requested = true


## Pauses the brain while preserving its exact current state for inspection.
func _pause_simulation() -> void:
	simulation_running = false
	single_step_requested = false

	play_button.text = "Play"


## Populates the input selector using the current brain input-neuron count.
func _setup_input_controls() -> void:
	fire_input_option_button.clear()

	var input_count = brain_dynamics.input_neuron_indices.size()
	var has_inputs = input_count > 0

	if has_inputs:
		fire_input_option_button.add_item("All")

		for input_number in input_count:
			fire_input_option_button.add_item(str(input_number + 1))

		fire_input_option_button.select(0)
		fire_input_button.text = "Fire All Inputs"
	else:
		fire_input_button.text = "Fire Input"

	fire_input_option_button.disabled = not has_inputs
	fire_input_button.disabled = not has_inputs
	fire_random_button.disabled = not has_inputs


## Updates the fire button label to match the selected input number.
func _on_fire_input_selected(index: int) -> void:
	if index == 0:
		fire_input_button.text = "Fire All Inputs"
	else:
		fire_input_button.text = "Fire Input %d" % index


## Queues the selected external input, or all inputs simultaneously.
func _on_fire_input_pressed() -> void:
	var selected_index = fire_input_option_button.selected

	if selected_index < 0:
		return

	var input_count = brain_dynamics.input_neuron_indices.size()
	var input_strength = fire_input_strength.value

	if selected_index == 0:
		for input_index in input_count:
			add_input_signal(input_index, input_strength)
	else:
		add_input_signal(selected_index - 1, input_strength)


## Starts or stops automatic random input firing.
## One input is fired immediately when the mode is enabled, then subsequent inputs
## are fired one at a time according to FireRandomDelay while the simulation is running.
func _on_fire_random_pressed() -> void:
	if random_input_running:
		_stop_random_input()
		return

	if brain_dynamics.input_neuron_indices.is_empty():
		return

	random_input_running = true
	random_input_elapsed = 0.0
	fire_random_button.text = "Stop Random"
	_fire_one_random_input()


## Advances the automatic random-input timer.
## The delay is measured in seconds and follows the current HSlider value live.
## At most one random input is queued per simulation iteration.
func _update_random_input(delta_seconds: float) -> void:
	if not random_input_running:
		return

	var delay_seconds = fire_random_delay.value

	if delay_seconds <= 0.0:
		_fire_one_random_input()
		return

	random_input_elapsed += delta_seconds

	if random_input_elapsed >= delay_seconds:
		random_input_elapsed -= delay_seconds
		_fire_one_random_input()


## Queues the current FireInputStrength on one randomly selected available input.
func _fire_one_random_input() -> void:
	var input_count = brain_dynamics.input_neuron_indices.size()

	if input_count == 0:
		return

	var input_index = random_input_rng.randi_range(0, input_count - 1)
	add_input_signal(input_index, fire_input_strength.value)


## Stops automatic random input firing and resets its timer.
func _stop_random_input() -> void:
	random_input_running = false
	random_input_elapsed = 0.0

	if is_instance_valid(fire_random_button):
		fire_random_button.text = "Fire Random"


## Queues one encoded input scalar s = f(r) in [0, 1] for an input neuron.
## The input index follows the stable input-neuron order of the generated brain.
func add_input_signal(input_index: int, input_signal: float) -> void:
	brain_dynamics.add_input_signal(input_index, input_signal)


## Returns the normalized signals emitted by output neurons in the latest iteration.
func get_output_signals() -> PackedFloat64Array:
	return brain_dynamics.output_signals


## Restores the generated brain and camera to their initial state without running BrainBuilder again.
func _reset_brain() -> void:
	_reset_camera()

	if current_brain.is_empty():
		return

	_pause_simulation()
	_stop_random_input()
	_clear_neuron_selection()
	brain_dynamics.reset()
	_update_reciprocal_connection_statistics()
	_build_heatmap_metadata(current_brain)

	if show_connections.button_pressed:
		_build_connections(current_brain)

	_update_heatmap_colors()
	_update_runtime_status()


## Restores a camera position that frames the entire brain.
func _reset_camera() -> void:
	camera_target = Vector3.ZERO
	camera_yaw = deg_to_rad(45.0)
	camera_pitch = deg_to_rad(25.0)

	if current_brain.is_empty():
		camera_distance = 10.0
	else:
		var size: Vector3i = current_brain["size"]
		var extent := Vector3(
			float(maxi(0, size.x - 1)) * NEURON_SPACING,
			float(maxi(0, size.y - 1)) * NEURON_SPACING,
			float(maxi(0, size.z - 1)) * NEURON_SPACING
		)
		var radius := maxf(1.0, 0.5 * extent.length())
		camera_distance = maxf(3.0, 1.25 * radius / tan(deg_to_rad(camera.fov * 0.5)))

	_update_camera_transform()


## Handles orbit, pan and zoom when the mouse event was not consumed by the UI.
func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
			_select_neuron_at_screen_position(event.position)
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			orbiting = event.pressed
		elif event.button_index == MOUSE_BUTTON_MIDDLE:
			panning = event.pressed
		elif event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_distance = maxf(0.5, camera_distance * 0.88)
			_update_camera_transform()
		elif event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_distance *= 1.14
			_update_camera_transform()

	elif event is InputEventMouseMotion:
		if orbiting:
			camera_yaw -= event.relative.x * 0.008
			camera_pitch = clampf(
				camera_pitch + event.relative.y * 0.008,
				deg_to_rad(-89.0),
				deg_to_rad(89.0)
			)
			_update_camera_transform()

		elif panning:
			var right := camera.global_transform.basis.x.normalized()
			var up := camera.global_transform.basis.y.normalized()
			var pan_scale := camera_distance * 0.0015
			camera_target += (-right * event.relative.x + up * event.relative.y) * pan_scale
			_update_camera_transform()


## Places the camera on a sphere around camera_target and points it toward the brain.
func _update_camera_transform() -> void:
	if camera == null:
		return

	var horizontal := cos(camera_pitch)
	var offset := Vector3(
		horizontal * sin(camera_yaw),
		sin(camera_pitch),
		horizontal * cos(camera_yaw)
	) * camera_distance

	camera.global_position = camera_target + offset
	camera.look_at(camera_target, Vector3.UP)


## Ensures the generation worker is joined before this scene is destroyed.
func _exit_tree() -> void:
	if generation_thread != null and generation_thread.is_started():
		generation_thread.wait_to_finish()


func _on_fire_input_strength_changed(value: float) -> void:
	fire_input_strength_label.text = str(value)


func _on_fire_random_delay_changed(value: float) -> void:
	fire_random_delay_label.text = str(value)
