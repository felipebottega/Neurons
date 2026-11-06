extends Node2D


const AgentClass = preload("res://scripts/simulations/agent.gd")
const TargetClass = preload("res://scripts/simulations/target.gd")
const MazeGeneratorClass = preload("res://scripts/simulations/maze_generator.gd")
const BRAIN_PREVIEW_SCENE: PackedScene = preload("res://scenes/agent_brain_preview.tscn")


enum ExperimentState {
	IDLE,
	GENERATING,
	RUNNING,
	PAUSED,
	OFFSPRING,
	FINISHED,
}


# The SpinBox in the scene owns its range, step and initial value.
var agent_count: int = 0
const SAVE_FORMAT_VERSION: int = 3
const SAVE_EXTENSION: String = "neuronsim"
const BRAIN_WORKER_BATCHES: int = 8
var gpu_backend: BrainPopulationGPU
# A complete neural/environment step always represents 1/60 s of simulated time.
# A slowdown uses both frame skipping and real-time pacing: with uncapped FPS,
# merely skipping frames does NOT reduce the number of neural steps per second.
const SIMULATION_STEP_SECONDS: float = 1.0 / 60.0
const PROCESS_FRAME_DIVISORS = [1, 2, 4, 8]  # raw, /2, /4, /8
const SPEED_STATS_WINDOW_US: int = 2000000
const RUNTIME_UI_UPDATE_INTERVAL_US: int = 16667  # UI labels capped at 60 Hz; steps are not decimated.
const NON_SIMULATION_UI_UPDATE_INTERVAL_US: int = 250000  # 4 Hz maximum for labels and other UI.
const BRAIN_PREVIEW_UPDATE_INTERVAL_US: int = 100000  # 10 Hz, on-demand GPU readback only.
const SIMULATION_4_COLLISION_EPSILON: float = 0.001
const SIMULATION_4_TOUCH_TOLERANCE: float = 0.05
const SIMULATION_4_ANGULAR_SECTOR_COUNT: int = 12
const SIMULATION_4_DIFFICULTY_BASE_THRESHOLD: float = 0.7
const SIMULATION_4_DIFFICULTY_THRESHOLD_STEP: float = 0.02
const SIMULATION_5_MAX_DIFFICULTY: int = 10
const SIMULATION_5_ADVANCE_THRESHOLD: float = 0.8
const SIMULATION_5_NUM_TESTS: int = 3
const SIMULATION_5_WALL_THICKNESS: float = 18.0
# Fixed maximum range of each directional proximity sensor. 0 = no nearby
# wall within range; 1 = collision at the current position.
const SIMULATION_5_SENSOR_RANGE: float = 220.0
# Simulation 6 shares Simulation 4's environment, collision solver, fitness and
# curriculum, but senses obstacles in eight fixed world-space directions.
# A ray measures distance from the agent center to the first collision surface
# (inflated obstacle or arena boundary). 0 = out of range; 1 = contact.
const SIMULATION_6_SENSOR_RANGE: float = 600.0
const SIMULATION_6_DIAGONAL_COMPONENT: float = 0.7071067811865476
const SIMULATION_6_SENSOR_DIRECTIONS = [
	Vector2.LEFT, Vector2.RIGHT, Vector2.UP, Vector2.DOWN,
	Vector2(-SIMULATION_6_DIAGONAL_COMPONENT, -SIMULATION_6_DIAGONAL_COMPONENT), Vector2(SIMULATION_6_DIAGONAL_COMPONENT, -SIMULATION_6_DIAGONAL_COMPONENT),
	Vector2(-SIMULATION_6_DIAGONAL_COMPONENT, SIMULATION_6_DIAGONAL_COMPONENT), Vector2(SIMULATION_6_DIAGONAL_COMPONENT, SIMULATION_6_DIAGONAL_COMPONENT),
]

const BRAIN_EXPLORER_GENE_ORDER = [
	"max_connections",
	"connections",
	"initial_search_region",
	"exponential_factor",
	"polarity_factor",
	"beta",
	"input_influence",
]

const HEATMAP_PHENOTYPE_KEYS = [
	"max_connections",
	"beta",
	"exponential_factor",
	"input_gain",
	"mean_polarity_factor",
	"mean_connection_energy",
	"mean_connections_per_neuron",
	"mean_connection_weight",
]
const HEATMAP_PHENOTYPE_LABELS = {
	"max_connections": "N_max",
	"beta": "Beta",
	"exponential_factor": "Exponential factor",
	"input_gain": "Input influence",
	"mean_polarity_factor": "Mean polarity",
	"mean_connection_energy": "Mean connection energy",
	"mean_connections_per_neuron": "Mean connections per neuron",
	"mean_connection_weight": "Mean connection weight",
}
const HEATMAP_RESOLUTIONS = [10, 30, 100, 300]

const HEATMAP_PHENOTYPE_INDEX = {
	"max_connections": 0,
	"beta": 1,
	"exponential_factor": 2,
	"input_gain": 3,
	"mean_polarity_factor": 4,
	"mean_connection_energy": 5,
	"mean_connections_per_neuron": 6,
	"mean_connection_weight": 7,
}

@export_group("Simulation")
@export var agent_speed: float = 600.0
@export var simulation_size: Vector2 = Vector2(644.0, 559.0)
@export var agent_square_size: float = 7.0
@export var target_square_size: float = 12.0
@export_range(1, 1000, 1) var max_brain_generation_attempts: int = 100
@export_range(0.05, 2.0, 0.05) var summary_update_interval: float = 0.25
@export_range(1.0, 60.0, 1.0) var agent_info_updates_per_second: float = 10.0

@export_group("Simulation 4: Obstacle")
@export_range(20.0, 400.0, 1.0) var simulation_4_min_target_distance: float = 190.0
@export_range(20.0, 300.0, 1.0) var simulation_4_obstacle_length: float = 140.0
@export_range(5.0, 80.0, 1.0) var simulation_4_obstacle_thickness: float = 24.0
@export_range(0.0, 0.4, 0.01) var simulation_4_obstacle_size_variation: float = 0.12


@export_group("Appearance")
@export var simulation_background_color: Color = Color(0.02, 0.18, 0.14, 1.0)
@export var simulation_border_color: Color = Color(0.55, 0.75, 0.70, 1.0)
@export var simulation_4_obstacle_color: Color = Color(0.64, 0.29, 0.20, 1.0)
@export var simulation_4_obstacle_border_color: Color = Color(1.0, 0.67, 0.45, 1.0)
@export var evolution_graph_size: Vector2 = Vector2(644.0, 135.0)
@export var evolution_graph_gap: float = 16.0
@export var evolution_graph_background_color: Color = Color(0.015, 0.11, 0.09, 1.0)
@export var evolution_graph_border_color: Color = Color(0.55, 0.75, 0.70, 1.0)
@export var evolution_graph_mean_color: Color = Color(0.25, 0.85, 1.0, 1.0)
@export var evolution_graph_best_color: Color = Color(1.0, 0.82, 0.2, 1.0)
@export var simulation_agent_color: Color = Color.WHITE
@export var simulation_elite_agent_color: Color = Color(1.0, 0.18, 0.18, 1.0)
@export var genetic_diversity_elite_color: Color = Color(1.0, 0.65, 0.3, 1.0)


@onready var dimension_i_field: LineEdit = $LineEdit
@onready var dimension_j_field: LineEdit = $LineEdit2
@onready var dimension_k_field: LineEdit = $LineEdit3
@onready var connections_density_field: LineEdit = $LineEdit6
@onready var generation_time_field: LineEdit = $LineEdit4
@onready var mutation_probability_field: LineEdit = $LineEdit5
@onready var start_button: Button = $Start
@onready var pause_button: Button = $Pause
@onready var single_elite_button: Button = $Pause2
@onready var save_button: Button = $Save
@onready var load_button: Button = $Load
@onready var save_brain_button: Button = $SaveBrain
@onready var free_mode_button: Button = $FreeMode
@onready var agent_info_label: Label = $AgentInfo
@onready var agent_info_label_2: Label = $AgentInfo2
@onready var agent_info_label_3: Label = $AgentInfo3
@onready var show_brain_button: Button = $ShowBrain
@onready var summary_label: Label = $Summary
@onready var simulation_space: Node2D = $SimulationSpace
@onready var simulation_4_num_tests_label: Label = $SimulationSpace/NumTestsLabel
@onready var speed_selector: OptionButton = $Speed
@onready var simulation_selector: OptionButton = $SimulationSelect
var active_simulation_index: int = 0
var simulation_change_dialog: AcceptDialog
@onready var evolution_graph_selector: OptionButton = $OptionButton
@onready var genetic_diversity_graph_selector: OptionButton = $OptionButton2
@onready var heatmap: Heatmap = $Heatmap
@onready var heatmap_x_selector: OptionButton = $GeneX
@onready var heatmap_y_selector: OptionButton = $GeneY
@onready var heatmap_resolution_selector: OptionButton = $Resolution
@onready var heatmap_tracking_button: Button = $HeatmapTracking


var experiment_state: ExperimentState = ExperimentState.IDLE
var experiment_duration_seconds: float = 0.0
var experiment_elapsed_seconds: float = 0.0
var summary_elapsed_seconds: float = 0.0
var last_runtime_ui_update_us: int = 0
var last_slow_ui_update_us: int = 0
var last_agent_info_update_us: int = 0
var last_brain_preview_update_us: int = 0
var preview_agent_index: int = -1
var preview_backend: BrainPopulationGPU
var agent_brain_preview: SubViewportContainer
var last_generation_ui_update_us: int = 0
# The renderer runs uncapped in raw. Slower modes preserve the measured raw
# frame rate as an upper bound so their skipped frames do not run arbitrarily
# fast and cancel out the intended slowdown.
var process_frame_divisor: int = 1
var process_frames_to_skip: int = 0
var raw_step_period_us: float = 0.0
var previous_raw_step_start_us: int = 0
var next_step_due_us: int = 0
var speed_stats_window_start_us: int = 0
var speed_stats_window_steps: int = 0
var effective_steps_per_second: float = 0.0
var previous_fps_limit: int = 0
var previous_vsync_mode: int = DisplayServer.VSYNC_ENABLED
var current_brain_size: Vector3i = Vector3i.ZERO
var mutation_probability: float = 0.001
var connections_density: float = 1.0

var num_inputs: int = 0
var num_outputs: int = 0
var target_test_count: int = 0
var target_position_change_interval_seconds: float = 0.0
var simulation_3_target_speed: float = 0.0
var evaluation_behavior: Callable
var fitness_reset_behavior: Callable
var fitness_accumulation_behavior: Callable
var fitness_behavior: Callable
var agent_info_behavior: Callable

var agents: Array = []
var target = null

var population_brains: Array = []
var population_genomes: Array = []
var population_from_load: bool = false
var loaded_population_size: Vector3i = Vector3i.ZERO
var loaded_population_density: float = 1.0
var loaded_population_num_inputs: int = -1
var loaded_population_num_outputs: int = -1

var agent_dynamics: Array = []
var fitness_integrals: PackedFloat64Array = PackedFloat64Array()
# These are reused across frames, avoiding allocations in the hot loop.
var _cached_evaluation_scores: PackedFloat64Array = PackedFloat64Array()
var _packed_gpu_inputs: PackedFloat32Array = PackedFloat32Array()
var simulation_4_obstacle_center: Vector2 = Vector2.ZERO
var simulation_4_obstacle_size: Vector2 = Vector2.ZERO
var simulation_4_obstacle_angle: float = 0.0
var simulation_4_target_sector_order: Array[int] = []
var simulation_4_difficulty: int = 3
var simulation_5_difficulty: int = 1
var simulation_6_difficulty: int = 3
var simulation_5_maze = MazeGeneratorClass.new()

var free_mode_active: bool = false
var target_dragging: bool = false
var target_drag_offset: Vector2 = Vector2.ZERO
var selected_agent_index: int = -1
var best_score_ever: float = 0.0
var best_distance_ever: float = 1.0e30
var current_mean_score: float = 0.0
var current_best_score: float = 0.0
var current_best_score_agent: int = -1
var current_mean_distance: float = 0.0
var current_min_distance: float = 0.0
var current_closest_agent: int = -1

var generation_number: int = 0
var completed_generation_count: int = 0
var evolution_mean_score_sum: float = 0.0
var evolution_mean_fitness_sum: float = 0.0
var evolution_best_mean_score: float = -1.0
var evolution_best_mean_score_generation: int = 0
var evolution_best_agent_score: float = -1.0
var evolution_best_agent_score_generation: int = 0
var evolution_best_mean_fitness: float = -1.0
var evolution_best_mean_fitness_generation: int = 0
var evolution_best_agent_fitness: float = -1.0
var evolution_best_agent_fitness_generation: int = 0
var evolution_mean_fitness_history: PackedFloat64Array = PackedFloat64Array()
var evolution_best_fitness_history: PackedFloat64Array = PackedFloat64Array()
var genetic_diversity_history: PackedFloat64Array = PackedFloat64Array()
var elite_genetic_diversity_history: PackedFloat64Array = PackedFloat64Array()
var current_elite_turnover_count: int = 0
var previous_generation_elite_signatures: Array[String] = []
var evolution_graph_mean_points_cache: PackedVector2Array = PackedVector2Array()
var evolution_graph_best_points_cache: PackedVector2Array = PackedVector2Array()
var evolution_graph_minimum_value_cache: float = 0.0
var evolution_graph_maximum_value_cache: float = 1.0
var genetic_diversity_graph_points_cache: PackedVector2Array = PackedVector2Array()
var elite_genetic_diversity_graph_points_cache: PackedVector2Array = PackedVector2Array()
var genetic_diversity_minimum_exponent_cache: int = -1
var genetic_diversity_excluded_agent_indices: Array[int] = []
var previous_generation_elite_indices: Array[int] = []
@onready var elite_button: Button = $Elite
@onready var num_agents_spinbox: SpinBox = $NumAgents
var history_graphs_layer: Node2D

var phenotypic_fitness_history_samples: Array = []
var phenotypic_fitness_history_values: PackedFloat64Array = PackedFloat64Array()
var generation_heatmap_phenotypes: Array = []
var phenotypic_heatmap_bin_sums: PackedFloat64Array = PackedFloat64Array()
var phenotypic_heatmap_bin_counts: PackedInt32Array = PackedInt32Array()
var heatmap_resolution: int = 30
var heatmap_tracking_enabled: bool = true

var target_test_index: int = 0
var generation_fitness_sums: PackedFloat64Array = PackedFloat64Array()
var generation_mean_final_score_sum: float = 0.0
var generation_best_final_score: float = -1.0
var generation_real_start_us: int = 0
var generation_real_time_seconds: float = 0.0

# Local RenderingDevice resources must remain on their creation thread.
# Generation therefore advances cooperatively on the main thread, yielding
# between brain attempts. Do not hand the device to a short-lived Thread.
var generation_active: bool = false
var _generation_rd: RenderingDevice
var _generation_partial_result: Dictionary = {}
var generation_progress_mutex: Mutex = Mutex.new()
var generation_agent_number: int = 0
var generation_brain_progress: int = 0
var generation_attempt_number: int = 0
var generation_cancel_requested: bool = false
var reset_pending: bool = false

var save_dialog: FileDialog
var load_dialog: FileDialog
var save_brain_dialog: FileDialog
var last_generation_best_brain_data: Dictionary = {}



func _ready() -> void:
	# _process runs once per rendered frame. Remove Godot's rendering caps while
	# this scene is active so "raw" is not stuck at the monitor refresh rate.
	# Restore both settings when leaving, to avoid affecting other scenes.
	previous_fps_limit = Engine.max_fps
	previous_vsync_mode = DisplayServer.window_get_vsync_mode()
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	history_graphs_layer = Node2D.new()
	history_graphs_layer.draw.connect(_draw_history_graphs.bind(history_graphs_layer))
	add_child(history_graphs_layer)

	# Use the value and restrictions already configured in the Inspector.
	agent_count = int(num_agents_spinbox.value)
	num_agents_spinbox.value_changed.connect(_on_num_agents_value_changed)
	elite_button.toggle_mode = true
	elite_button.button_pressed = false
	elite_button.text = "OFF"
	elite_button.toggled.connect(_on_elite_toggled)
	single_elite_button.toggle_mode = true
	single_elite_button.set_pressed_no_signal(false)
	single_elite_button.toggled.connect(_on_single_elite_toggled)
	_update_single_elite_button()
	show_brain_button.toggle_mode = true
	show_brain_button.button_pressed = false
	show_brain_button.text = "OFF"
	show_brain_button.toggled.connect(_on_show_brain_toggled)
	start_button.pressed.connect(_on_start_pressed)
	pause_button.pressed.connect(_on_pause_pressed)
	mutation_probability_field.text_changed.connect(_on_mutation_probability_text_changed)
	mutation_probability_field.text_submitted.connect(_on_mutation_probability_text_submitted)
	mutation_probability_field.focus_exited.connect(_on_mutation_probability_focus_exited)
	save_button.pressed.connect(_open_save_dialog)
	load_button.pressed.connect(_open_load_dialog)
	save_brain_button.pressed.connect(_open_save_brain_dialog)
	free_mode_button.pressed.connect(_on_free_mode_pressed)
	_setup_speed_selector()
	speed_selector.item_selected.connect(_on_speed_selected)
	simulation_selector.item_selected.connect(_on_simulation_selected)
	active_simulation_index = simulation_selector.selected
	simulation_change_dialog = AcceptDialog.new()
	simulation_change_dialog.title = "Incompatible simulation"
	simulation_change_dialog.dialog_text = ""
	add_child(simulation_change_dialog)
	_setup_evolution_graph_selector()
	evolution_graph_selector.item_selected.connect(_on_evolution_graph_selected)
	_setup_genetic_diversity_graph_selector()
	genetic_diversity_graph_selector.item_selected.connect(_on_genetic_diversity_graph_selected)
	_setup_heatmap_selectors()
	_setup_heatmap_resolution_selector()
	heatmap_x_selector.item_selected.connect(_on_heatmap_axis_selected)
	heatmap_y_selector.item_selected.connect(_on_heatmap_axis_selected)
	heatmap_resolution_selector.item_selected.connect(_on_heatmap_resolution_selected)
	heatmap_tracking_button.toggled.connect(_on_heatmap_tracking_toggled)
	_reset_phenotypic_fitness_history()
	heatmap.hide()

	_setup_simulation(simulation_selector.selected)
	mutation_probability_field.text = "0.001"
	connections_density_field.tooltip_text = "Connections density eta (> 0); 1 restores original connectivity."
	if connections_density_field.text.is_empty():
		connections_density_field.text = "1"
	_setup_file_dialogs()
	_refresh_buttons()
	summary_label.text = "Ready."
	agent_info_label.text = "Click an agent to inspect its neural signals."
	agent_info_label_2.text = ""
	agent_info_label_3.text = ""
	queue_redraw()


## Population size may change only outside active generation/simulation.
## A new population size invalidates previously built GPU brains and evolution data.
func _on_num_agents_value_changed(value: float) -> void:
	if _simulation_is_active() or reset_pending:
		num_agents_spinbox.set_value_no_signal(float(agent_count))
		return
	var selected_count: int = int(value)
	if selected_count == agent_count:
		return

	# Release any already loaded/generated GPU phenotypes before changing size.
	_clear_runtime_state()
	_reset_evolution_statistics()
	population_brains.clear()
	population_genomes.clear()
	population_from_load = false
	loaded_population_size = Vector3i.ZERO
	loaded_population_density = 1.0
	loaded_population_num_inputs = -1
	loaded_population_num_outputs = -1
	current_brain_size = Vector3i.ZERO
	agent_count = selected_count
	_setup_simulation(simulation_selector.selected)
	summary_label.text = "Ready (%d agents)." % agent_count
	_refresh_buttons()
	queue_redraw()


func _setup_speed_selector() -> void:
	# Only the scene owns the speed control; never modify Engine.time_scale or
	# physics_ticks_per_second. The neural core remains simulation-agnostic.
	speed_selector.clear()
	for speed_label in ["raw", "/2", "/4", "/8"]:
		speed_selector.add_item(speed_label)
	speed_selector.tooltip_text = (
		"raw: one complete neural/environment iteration each frame, uncapped. "
		+ "/2, /4, /8: target one half, one quarter, or one eighth of the measured "
		+ "raw iteration rate by skipping frames and pacing in real time. "
		+ "Neural timestep always remains 1/60 s."
	)
	speed_selector.select(0)
	_on_speed_selected(0)


func _on_speed_selected(index: int) -> void:
	if index < 0 or index >= PROCESS_FRAME_DIVISORS.size():
		return
	process_frame_divisor = int(PROCESS_FRAME_DIVISORS[index])
	process_frames_to_skip = 0
	next_step_due_us = 0
	previous_raw_step_start_us = 0
	speed_stats_window_start_us = 0
	speed_stats_window_steps = 0
	effective_steps_per_second = 0.0
	_apply_speed_render_limit()
	if experiment_state == ExperimentState.RUNNING:
		_update_summary()


func _apply_speed_render_limit() -> void:
	if process_frame_divisor == 1 or raw_step_period_us <= 0.0:
		# Never throttle the raw mode. When no baseline exists yet, determine it
		# from the first completed step before limiting FPS in a slower mode.
		Engine.max_fps = 0
		return
	# Keep roughly the raw frame cadence in slow modes. Their 1, 3 or 7
	# skipped frames must not become arbitrarily cheap microframes.
	Engine.max_fps = maxi(1, int(ceil(1000000.0 / raw_step_period_us)))


func _record_completed_speed_step(now_us: int) -> void:
	if speed_stats_window_start_us == 0:
		speed_stats_window_start_us = now_us
	speed_stats_window_steps += 1
	var window_us: int = now_us - speed_stats_window_start_us
	if window_us >= SPEED_STATS_WINDOW_US:
		effective_steps_per_second = float(speed_stats_window_steps) * 1000000.0 / float(window_us)
		speed_stats_window_start_us = now_us
		speed_stats_window_steps = 0


func _setup_evolution_graph_selector() -> void:
	evolution_graph_selector.clear()
	evolution_graph_selector.add_item("Raw")
	evolution_graph_selector.add_item("Average 10")
	evolution_graph_selector.add_item("Average 50")
	evolution_graph_selector.select(0)


func _on_evolution_graph_selected(_index: int) -> void:
	_rebuild_evolution_graph_cache()
	_queue_history_graph_redraw()


func _setup_genetic_diversity_graph_selector() -> void:
	genetic_diversity_graph_selector.clear()
	genetic_diversity_graph_selector.add_item("Raw")
	genetic_diversity_graph_selector.add_item("Average 10")
	genetic_diversity_graph_selector.add_item("Average 50")
	genetic_diversity_graph_selector.select(0)


func _on_genetic_diversity_graph_selected(_index: int) -> void:
	_rebuild_genetic_diversity_graph_cache()
	_queue_history_graph_redraw()


## Creates the heatmap selectors from generation-time brain phenotypes.
func _setup_heatmap_selectors() -> void:
	heatmap_x_selector.clear()
	heatmap_y_selector.clear()

	for phenotype_key in HEATMAP_PHENOTYPE_KEYS:
		var item_index: int = heatmap_x_selector.item_count
		var label: String = str(HEATMAP_PHENOTYPE_LABELS[phenotype_key])

		heatmap_x_selector.add_item(label)
		heatmap_x_selector.set_item_metadata(item_index, phenotype_key)
		heatmap_y_selector.add_item(label)
		heatmap_y_selector.set_item_metadata(item_index, phenotype_key)

	if heatmap_x_selector.item_count > 0:
		heatmap_x_selector.select(0)
		heatmap_y_selector.select(mini(1, heatmap_y_selector.item_count - 1))


## Creates the heatmap resolution selector and keeps 30 as the initial resolution.
func _setup_heatmap_resolution_selector() -> void:
	heatmap_resolution_selector.clear()

	for resolution in HEATMAP_RESOLUTIONS:
		var item_index: int = heatmap_resolution_selector.item_count
		heatmap_resolution_selector.add_item(str(resolution))
		heatmap_resolution_selector.set_item_metadata(item_index, resolution)
		if resolution == heatmap_resolution:
			heatmap_resolution_selector.select(item_index)


func _on_heatmap_axis_selected(_index: int) -> void:
	if heatmap_tracking_enabled:
		_rebuild_phenotypic_fitness_heatmap()


func _on_heatmap_resolution_selected(index: int) -> void:
	if not heatmap_tracking_enabled:
		return
	heatmap_resolution = int(heatmap_resolution_selector.get_item_metadata(index))
	_rebuild_phenotypic_fitness_heatmap()


## Pauses or resumes phenotype collection without discarding previously recorded generations.
func _on_heatmap_tracking_toggled(enabled: bool) -> void:
	heatmap_tracking_enabled = enabled
	heatmap_tracking_button.text = "ON" if enabled else "OFF"
	heatmap_x_selector.disabled = not enabled
	heatmap_y_selector.disabled = not enabled
	heatmap_resolution_selector.disabled = not enabled

	if not enabled:
		# Never record a partially tracked generation after switching back ON.
		generation_heatmap_phenotypes.clear()


## Clears the unsaved phenotype-fitness history and returns the heatmap to an empty state.
func _reset_phenotypic_fitness_history() -> void:
	phenotypic_fitness_history_samples.clear()
	phenotypic_fitness_history_values = PackedFloat64Array()
	generation_heatmap_phenotypes.clear()
	_initialize_phenotypic_heatmap_bins()
	_update_phenotypic_fitness_heatmap()


func _initialize_phenotypic_heatmap_bins() -> void:
	var bin_count: int = heatmap_resolution * heatmap_resolution
	phenotypic_heatmap_bin_sums = PackedFloat64Array()
	phenotypic_heatmap_bin_sums.resize(bin_count)
	phenotypic_heatmap_bin_counts = PackedInt32Array()
	phenotypic_heatmap_bin_counts.resize(bin_count)


## Captures only values fixed at brain generation, before lifetime dynamics begins.
func _capture_generation_heatmap_phenotypes() -> void:
	if not heatmap_tracking_enabled:
		return
	generation_heatmap_phenotypes.clear()

	# Only collect an initial phenotype from freshly constructed agents.
	if population_from_load or population_brains.size() != agent_count:
		return

	for brain_variant in population_brains:
		var brain: Dictionary = brain_variant
		generation_heatmap_phenotypes.append(_brain_heatmap_phenotypes(brain))


## Stores all phenotype-fitness pairs from the completed generation.
func _record_phenotypic_fitness_generation() -> void:
	if not heatmap_tracking_enabled or generation_heatmap_phenotypes.size() != agent_count:
		return

	for agent_index in agent_count:
		var fitness: float = generation_fitness_sums[agent_index] / float(target_test_count)
		var phenotypes: PackedFloat64Array = generation_heatmap_phenotypes[agent_index]
		phenotypic_fitness_history_samples.append(phenotypes.duplicate())
		phenotypic_fitness_history_values.append(fitness)

	# The observed ranges can expand when new samples arrive, so all historical
	# samples must be rebinned using the current minimum and maximum values.
	_rebuild_phenotypic_fitness_heatmap()


## Extracts the generation-time scalar phenotype used by every heatmap option.
func _brain_heatmap_phenotypes(brain: Dictionary) -> PackedFloat64Array:
	var values: PackedFloat64Array = PackedFloat64Array()
	values.resize(HEATMAP_PHENOTYPE_KEYS.size())

	values[int(HEATMAP_PHENOTYPE_INDEX["max_connections"])] = float(brain.get("max_connections", 0))
	values[int(HEATMAP_PHENOTYPE_INDEX["beta"])] = float(brain.get("beta", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["exponential_factor"])] = float(brain.get("exponential_factor", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["input_gain"])] = float(brain.get("input_gain", 0.0))

	var neurons: Array = brain.get("neurons", [])
	if neurons.is_empty():
		return values
	var polarity_factor_sum: float = 0.0
	var connection_energy_sum: float = 0.0
	var connection_weight_sum: float = 0.0
	var connection_count: int = 0

	for neuron_variant in neurons:
		var neuron: Dictionary = neuron_variant
		polarity_factor_sum += float(neuron.get("polarity_factor", 0.0))
		connection_energy_sum += float(neuron.get("connections_value", 0.0))

		var connections: Array = neuron.get("connections", [])
		connection_count += connections.size()
		for connection_variant in connections:
			var connection: Dictionary = connection_variant
			connection_weight_sum += float(connection.get("weight", 0.0))

	# GPU phenotypes never unpack their per-neuron synapses. Use the builder's
	# verified connection aggregates instead of silently reporting zero.
	connection_count = int(brain.get("connection_total", connection_count))
	connection_weight_sum = float(brain.get("connection_weight_sum", connection_weight_sum))
	var neuron_count: float = float(neurons.size())
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_polarity_factor"])] = polarity_factor_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_connection_energy"])] = connection_energy_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_connections_per_neuron"])] = float(connection_count) / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_connection_weight"])] = connection_weight_sum / float(connection_count) if connection_count > 0 else 0.0

	return values


## Rebuilds the selected two-phenotype projection from the complete in-memory history.
func _rebuild_phenotypic_fitness_heatmap() -> void:
	heatmap.show()
	_initialize_phenotypic_heatmap_bins()

	if heatmap_x_selector.item_count == 0 or heatmap_y_selector.item_count == 0:
		return

	if phenotypic_fitness_history_samples.is_empty():
		heatmap.set_axis_ranges()
		_update_phenotypic_fitness_heatmap()
		return

	var x_key: String = str(heatmap_x_selector.get_item_metadata(heatmap_x_selector.selected))
	var y_key: String = str(heatmap_y_selector.get_item_metadata(heatmap_y_selector.selected))
	var x_range: Vector2 = _heatmap_observed_range(x_key)
	var y_range: Vector2 = _heatmap_observed_range(y_key)

	for sample_index in phenotypic_fitness_history_samples.size():
		_accumulate_phenotypic_fitness_sample(
			phenotypic_fitness_history_samples[sample_index],
			phenotypic_fitness_history_values[sample_index],
			x_key,
			y_key,
			x_range,
			y_range
		)

	heatmap.set_axis_ranges(x_range.x, x_range.y, y_range.x, y_range.y)
	_update_phenotypic_fitness_heatmap()


func _accumulate_phenotypic_fitness_sample(
	phenotypes: PackedFloat64Array,
	fitness: float,
	x_key: String,
	y_key: String,
	x_range: Vector2,
	y_range: Vector2
) -> void:
	var x_value: float = phenotypes[int(HEATMAP_PHENOTYPE_INDEX[x_key])]
	var y_value: float = phenotypes[int(HEATMAP_PHENOTYPE_INDEX[y_key])]
	var x_coordinate: float = _heatmap_phenotype_coordinate(x_value, x_range)
	var y_coordinate: float = _heatmap_phenotype_coordinate(y_value, y_range)
	var x_bin: int = _heatmap_coordinate_to_bin(x_coordinate)
	var y_bin: int = heatmap_resolution - 1 - _heatmap_coordinate_to_bin(y_coordinate)
	var bin_index: int = x_bin + heatmap_resolution * y_bin

	phenotypic_heatmap_bin_sums[bin_index] += fitness
	phenotypic_heatmap_bin_counts[bin_index] += 1


## Maps an ordered phenotype linearly to the range currently observed in the history.
func _heatmap_phenotype_coordinate(value: float, value_range: Vector2) -> float:
	var span: float = value_range.y - value_range.x
	if span <= 0.0:
		return 0.5
	return clampf((value - value_range.x) / span, 0.0, 1.0)


## Returns the minimum and maximum values currently present in the stored history.
func _heatmap_observed_range(phenotype_key: String) -> Vector2:
	var phenotype_index: int = int(HEATMAP_PHENOTYPE_INDEX[phenotype_key])
	var minimum_value: float = INF
	var maximum_value: float = -INF

	for phenotypes_variant in phenotypic_fitness_history_samples:
		var phenotypes: PackedFloat64Array = phenotypes_variant
		var value: float = phenotypes[phenotype_index]
		minimum_value = minf(minimum_value, value)
		maximum_value = maxf(maximum_value, value)

	if minimum_value == INF:
		return Vector2(0.0, 1.0)

	if is_equal_approx(minimum_value, maximum_value):
		var padding: float = maxf(absf(minimum_value) * 0.05, 0.001)
		return Vector2(minimum_value - padding, maximum_value + padding)

	return Vector2(minimum_value, maximum_value)


func _heatmap_coordinate_to_bin(coordinate: float) -> int:
	return clampi(floori(clampf(coordinate, 0.0, 1.0) * heatmap_resolution), 0, heatmap_resolution - 1)


## Converts accumulated sums and counts to mean-fitness cells; unvisited cells remain NaN.
func _update_phenotypic_fitness_heatmap() -> void:
	if heatmap_x_selector.item_count == 0 or heatmap_y_selector.item_count == 0:
		return

	var bin_count: int = heatmap_resolution * heatmap_resolution
	var heatmap_values: PackedFloat64Array = PackedFloat64Array()
	heatmap_values.resize(bin_count)
	for bin_index in bin_count:
		if phenotypic_heatmap_bin_counts.size() == bin_count and phenotypic_heatmap_bin_counts[bin_index] > 0:
			var mean_fitness: float = phenotypic_heatmap_bin_sums[bin_index] / float(phenotypic_heatmap_bin_counts[bin_index])
			heatmap_values[bin_index] = mean_fitness
		else:
			heatmap_values[bin_index] = NAN

	var minimum_fitness: float = INF
	var maximum_fitness: float = -INF
	for value in heatmap_values:
		if not is_nan(value):
			minimum_fitness = minf(minimum_fitness, value)
			maximum_fitness = maxf(maximum_fitness, value)

	if minimum_fitness == INF:
		minimum_fitness = 0.0
		maximum_fitness = 1.0

	heatmap.set_data(
		heatmap_values,
		Vector2i(heatmap_resolution, heatmap_resolution),
		minimum_fitness,
		maximum_fitness
	)


func _on_simulation_selected(index: int) -> void:
	if index == active_simulation_index:
		return

	# Validate against the population actually being evolved, not merely the
	# previous dropdown value. Incremental GPU generation also has a fixed I/O
	# layout, even before the first finished genome is available.
	var required_inputs: int = 13 if index == 5 else (8 if index == 4 else (5 if index == 3 else 4))
	var required_outputs: int = 4
	var has_existing_population: bool = (
		_simulation_is_active()
		or population_genomes.size() == agent_count
		or population_brains.size() == agent_count
	)
	var population_inputs: int = num_inputs
	var population_outputs: int = num_outputs
	if population_from_load and population_genomes.size() == agent_count:
		population_inputs = loaded_population_num_inputs
		population_outputs = loaded_population_num_outputs
		# A loaded population may already be incompatible with the selected
		# simulation, but it can still be transferred to a compatible one.
		has_existing_population = true

	if has_existing_population and (population_inputs != required_inputs or population_outputs != required_outputs):
		simulation_selector.select(active_simulation_index)
		simulation_change_dialog.dialog_text = (
			"Cannot switch to simulation %d with the current population.\n\n" % (index + 1)
			+ "Population: %d inputs / %d outputs\n" % [population_inputs, population_outputs]
			+ "Simulation %d: %d inputs / %d outputs\n\n" % [index + 1, required_inputs, required_outputs]
			+ "The current simulation and population are unchanged."
		)
		simulation_change_dialog.popup_centered()
		return

	var was_running: bool = experiment_state == ExperimentState.RUNNING
	var was_paused: bool = experiment_state == ExperimentState.PAUSED
	var is_generating: bool = generation_active or experiment_state == ExperimentState.GENERATING or experiment_state == ExperimentState.OFFSPRING
	var previous_generation_number: int = generation_number
	var previous_elites: Array[int] = previous_generation_elite_indices.duplicate()
	var previous_excluded: Array[int] = genetic_diversity_excluded_agent_indices.duplicate()

	active_simulation_index = index
	_setup_simulation(index)
	# Fitness from different environments is not directly comparable. Keep the
	# evolved genomes and global generation number, but start fresh statistics
	# and phenotype/fitness correlations for the new task.
	_reset_evolution_statistics()
	generation_number = previous_generation_number
	previous_generation_elite_indices = previous_elites
	genetic_diversity_excluded_agent_indices = previous_excluded

	if was_running or was_paused:
		# Discard the unfinished test (without recording or selecting from its
		# fitness), and re-evaluate the SAME genetic population under the new
		# task. The existing GPU backend is reset, not destroyed or regenerated.
		target_test_index = 0
		generation_fitness_sums.resize(agent_count)
		generation_fitness_sums.fill(0.0)
		generation_mean_final_score_sum = 0.0
		generation_best_final_score = -1.0
		generation_real_start_us = Time.get_ticks_usec()
		if index == 2:
			simulation_3_target_speed = agent_speed * 0.5
		if index == 3 or index == 5:
			_simulation_4_shuffle_target_sectors()
		if heatmap_tracking_enabled:
			_capture_generation_heatmap_phenotypes()
		_start_experiment_from_population(false)
		if was_paused and experiment_state == ExperimentState.RUNNING:
			experiment_state = ExperimentState.PAUSED
			pause_button.text = "Continue"
			_update_summary()
			_refresh_buttons()
	elif is_generating:
		# The incremental worker has captured the same channel counts. Its
		# output will start in the newly selected simulation when ready.
		summary_label.text = "Simulation %d selected. The next evaluation will use this task." % (index + 1)
	else:
		# Inactive populations loaded from disk are retained as genotypes;
		# released GPU phenotypes can be rebuilt on the next Start.
		if population_genomes.size() == agent_count:
			if not population_from_load:
				population_from_load = true
				loaded_population_size = current_brain_size
				loaded_population_density = connections_density
				loaded_population_num_inputs = population_inputs
				loaded_population_num_outputs = population_outputs
			_clear_runtime_state()
			population_brains.clear()
			summary_label.text = "Compatible genomes preserved. Press Start to evolve in simulation %d." % (index + 1)
		else:
			summary_label.text = "Ready."

	_refresh_buttons()
	queue_redraw()


func _setup_simulation(index: int) -> void:
	_cached_evaluation_scores.resize(agent_count)
	match index:
		0:
			_setup_simulation_1()
		1:
			_setup_simulation_2()
		2:
			_setup_simulation_3()
		3:
			_setup_simulation_4()
		4:
			_setup_simulation_5()
		5:
			_setup_simulation_6()
		_:
			simulation_selector.select(0)
			_setup_simulation_1()
	# Keep all 13 input labels legible in the agent-info panel.
	agent_info_label.add_theme_font_size_override("font_size", 12 if index == 5 else 14)
	_update_simulation_4_num_tests_label()


func _setup_simulation_1() -> void:
	num_inputs = 4
	num_outputs = 4
	target_test_count = 3
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_1_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_1_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_1_fitness")
	agent_info_behavior = Callable(self, "_simulation_1_agent_info")

	agents.clear()
	for _agent_index in agent_count:
		agents.append(AgentClass.new(
			num_inputs,
			num_outputs,
			Callable(self, "_simulation_1_reset_agent"),
			Callable(self, "_simulation_1_update_agent_inputs"),
			Callable(self, "_simulation_1_apply_agent_outputs")
		))

	target = TargetClass.new(
		Callable(self, "_simulation_1_reset_target"),
		Callable(self, "_simulation_1_update_target")
	)


func _setup_simulation_2() -> void:
	num_inputs = 4
	num_outputs = 4
	target_test_count = 3
	target_position_change_interval_seconds = 5.0
	evaluation_behavior = Callable(self, "_simulation_1_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_1_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_1_fitness")
	agent_info_behavior = Callable(self, "_simulation_1_agent_info")

	agents.clear()
	for _agent_index in agent_count:
		agents.append(AgentClass.new(
			num_inputs,
			num_outputs,
			Callable(self, "_simulation_1_reset_agent"),
			Callable(self, "_simulation_1_update_agent_inputs"),
			Callable(self, "_simulation_1_apply_agent_outputs")
		))

	target = TargetClass.new(
		Callable(self, "_simulation_2_reset_target"),
		Callable(self, "_simulation_2_update_target")
	)


func _setup_simulation_3() -> void:
	num_inputs = 4
	num_outputs = 4
	simulation_3_target_speed = agent_speed * 0.5
	target_test_count = 3
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_1_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_1_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_1_fitness")
	agent_info_behavior = Callable(self, "_simulation_1_agent_info")

	agents.clear()
	for _agent_index in agent_count:
		agents.append(AgentClass.new(
			num_inputs,
			num_outputs,
			Callable(self, "_simulation_1_reset_agent"),
			Callable(self, "_simulation_1_update_agent_inputs"),
			Callable(self, "_simulation_1_apply_agent_outputs")
		))

	target = TargetClass.new(
		Callable(self, "_simulation_3_reset_target"),
		Callable(self, "_simulation_3_update_target")
	)


func _setup_simulation_4() -> void:
	# Obstacle avoidance with 5 sensory inputs and a sector-based curriculum.
	# Three target tests initially, progressing up to 12 tests.
	# Fitness continues to use only proximity during the final second.
	num_inputs = 5
	num_outputs = 4
	target_test_count = simulation_4_difficulty
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_4_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_4_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_4_fitness")
	agent_info_behavior = Callable(self, "_simulation_4_agent_info")

	agents.clear()
	for _agent_index in agent_count:
		agents.append(AgentClass.new(
			num_inputs,
			num_outputs,
			Callable(self, "_simulation_4_reset_agent"),
			Callable(self, "_simulation_4_update_agent_inputs"),
			Callable(self, "_simulation_4_apply_agent_outputs")
		))

	target = TargetClass.new(
		Callable(self, "_simulation_4_reset_target"),
		Callable(self, "_simulation_1_update_target")
	)


func _setup_simulation_5() -> void:
	# Four target displacement channels plus four local wall-distance sensors.
	# Unlike Simulation 4 (5 inputs), the maze task requires 8 inputs.
	# A single random layout is shared by all agents in each test.
	num_inputs = 8
	num_outputs = 4
	target_test_count = SIMULATION_5_NUM_TESTS
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_5_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_5_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_5_fitness")
	agent_info_behavior = Callable(self, "_simulation_5_agent_info")

	agents.clear()
	for _agent_index in agent_count:
		agents.append(AgentClass.new(
			num_inputs, num_outputs,
			Callable(self, "_simulation_5_reset_agent"),
			Callable(self, "_simulation_5_update_agent_inputs"),
			Callable(self, "_simulation_5_apply_agent_outputs")
		))
	target = TargetClass.new(
		Callable(self, "_simulation_5_reset_target"),
		Callable(self, "_simulation_1_update_target")
	)

func _setup_simulation_6() -> void:
	# Exact Simulation 4 experiment, with eight additional obstacle sensors.
	# Inputs 1-4 = target displacement, 5 = binary contact,
	# 6-13 = W, E, N, S, NW, NE, SW, SE obstacle proximity.
	num_inputs = 13
	num_outputs = 4
	target_test_count = simulation_6_difficulty
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_4_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_4_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_4_fitness")
	agent_info_behavior = Callable(self, "_simulation_6_agent_info")

	agents.clear()
	for _agent_index in agent_count:
		agents.append(AgentClass.new(
			num_inputs,
			num_outputs,
			Callable(self, "_simulation_4_reset_agent"),
			Callable(self, "_simulation_6_update_agent_inputs"),
			Callable(self, "_simulation_4_apply_agent_outputs")
		))
	target = TargetClass.new(
		Callable(self, "_simulation_4_reset_target"),
		Callable(self, "_simulation_1_update_target")
	)


func _exit_tree() -> void:
	Engine.max_fps = previous_fps_limit
	DisplayServer.window_set_vsync_mode(previous_vsync_mode)
	# The cooperative generator may be suspended on process_frame. Its GPU
	# objects still belong to this scene and must be freed before destruction.
	if _generation_rd != null:
		_dispose_generation_population(_generation_partial_result, _generation_rd)
	_generation_partial_result = {}
	_generation_rd = null
	generation_active = false
	_release_gpu_population()


func _process(_delta: float) -> void:
	if experiment_state == ExperimentState.RUNNING:
		_advance_simulation_on_process_frame()
		if experiment_state == ExperimentState.RUNNING:
			_update_runtime_ui_if_due()
			if show_brain_button.button_pressed:
				_update_brain_preview()

	if not generation_active:
		return
	var current_time_us: int = Time.get_ticks_usec()
	if current_time_us - last_generation_ui_update_us < NON_SIMULATION_UI_UPDATE_INTERVAL_US:
		return
	last_generation_ui_update_us = current_time_us
	generation_progress_mutex.lock()
	var agent_number: int = generation_agent_number
	var brain_progress: int = generation_brain_progress
	var attempt_number: int = generation_attempt_number
	generation_progress_mutex.unlock()
	var generation_title: String = "Generating offspring brains..." if experiment_state == ExperimentState.OFFSPRING else "Generating brains..."
	summary_label.text = (
		generation_title + "\n"
		+ "Agent: %d / %d\n" % [agent_number, agent_count]
		+ "Attempt: %d\n" % attempt_number
		+ "Brain generation: %d%%" % brain_progress
	)


func _run_initial_generation_async(size: Vector3i, maximum_attempts: int, selected_num_inputs: int, selected_num_outputs: int) -> void:
	var result: Dictionary = await _generate_population_incremental(size, maximum_attempts, selected_num_inputs, selected_num_outputs)
	_complete_incremental_generation(result)


func _run_offspring_generation_async(
		parent_genomes: Array, top_indices: Array[int], ranked_non_elite_indices: Array[int],
		size: Vector3i, maximum_attempts: int, selected_mutation_probability: float,
		selected_num_inputs: int, selected_num_outputs: int, selected_single_elite: bool) -> void:
	var result: Dictionary = await _generate_offspring_population_incremental(
		parent_genomes, top_indices, ranked_non_elite_indices, size, maximum_attempts,
		selected_mutation_probability, selected_num_inputs, selected_num_outputs, selected_single_elite
	)
	_complete_incremental_generation(result)


func _complete_incremental_generation(result: Dictionary) -> void:
	# This method and every other GPU operation now run on the same thread.
	generation_active = false
	if reset_pending:
		if _generation_rd != null:
			_dispose_generation_population(result, _generation_rd)
		_generation_partial_result = {}
		_generation_rd = null
		_complete_reset()
		return
	var error_message: String = str(result.get("error", ""))
	if not error_message.is_empty() or result.get("brains", []).size() != agent_count or result.get("genomes", []).size() != agent_count:
		if _generation_rd != null:
			_dispose_generation_population(result, _generation_rd)
		_generation_partial_result = {}
		_generation_rd = null
		population_brains.clear()
		population_genomes.clear()
		experiment_state = ExperimentState.IDLE
		summary_label.text = error_message if not error_message.is_empty() else "Brain generation failed: incomplete population."
		_refresh_buttons()
		queue_redraw()
		return
	# Ownership of the shared local RD passes to the simulation backend.
	_generation_partial_result = {}
	_generation_rd = null
	population_brains = result["brains"]
	population_genomes = result["genomes"]
	var excluded_indices_variant: Variant = result.get("genetic_diversity_excluded_agent_indices", [])
	genetic_diversity_excluded_agent_indices.clear()
	if excluded_indices_variant is Array:
		for index_variant in excluded_indices_variant:
			genetic_diversity_excluded_agent_indices.append(int(index_variant))
	population_from_load = false
	_start_experiment_from_population()


func _advance_simulation_on_process_frame() -> void:
	# Frame skipping alone has almost no effect with uncapped rendering: the
	# empty frames become much faster. Thus slow modes also require a real
	# elapsed-time gate based on the raw iteration cadence.
	if process_frames_to_skip > 0:
		process_frames_to_skip -= 1
		return
	var start_us: int = Time.get_ticks_usec()
	if process_frame_divisor > 1 and start_us < next_step_due_us:
		return

	if process_frame_divisor == 1:
		if previous_raw_step_start_us > 0:
			var sample_us: int = start_us - previous_raw_step_start_us
			# Exclude unusual stalls (dragging window, debugging, generation)
			# from the reference speed estimate.
			if sample_us > 0 and sample_us < 250000:
				if raw_step_period_us <= 0.0:
					raw_step_period_us = float(sample_us)
				else:
					raw_step_period_us = lerpf(raw_step_period_us, float(sample_us), 0.10)
		previous_raw_step_start_us = start_us

	process_frames_to_skip = process_frame_divisor - 1
	if not _execute_one_neural_environment_step():
		return

	var end_us: int = Time.get_ticks_usec()
	if raw_step_period_us <= 0.0:
		# Enables /2, /4 and /8 even if selected before running raw.
		raw_step_period_us = maxf(float(end_us - start_us), 1000.0)
		_apply_speed_render_limit()

	if process_frame_divisor > 1:
		# One complete environment -> neural -> environment cycle each time.
		# All skipped callbacks do *no* neural computations.
		next_step_due_us = start_us + int(ceil(raw_step_period_us * float(process_frame_divisor)))
	_record_completed_speed_step(end_us)


func _execute_one_neural_environment_step() -> bool:
	if free_mode_active:
		if not _advance_gpu_step(SIMULATION_STEP_SECONDS, false):
			return false
		summary_elapsed_seconds += SIMULATION_STEP_SECONDS
		return true

	var remaining_time: float = experiment_duration_seconds - experiment_elapsed_seconds
	if remaining_time <= 0.0:
		_finish_experiment()
		return false

	var step_delta: float = minf(SIMULATION_STEP_SECONDS, remaining_time)
	target.update(step_delta)
	if not _advance_gpu_step(step_delta, true):
		return false

	experiment_elapsed_seconds += step_delta
	summary_elapsed_seconds += step_delta

	if experiment_elapsed_seconds >= experiment_duration_seconds - 0.000001:
		_finish_experiment()
	return true


func _update_runtime_ui_if_due() -> void:
	var current_time_us: int = Time.get_ticks_usec()
	if current_time_us - last_runtime_ui_update_us < RUNTIME_UI_UPDATE_INTERVAL_US:
		return

	last_runtime_ui_update_us = current_time_us

	# Keep expensive aggregate UI metrics at up to 60 Hz. Agents are already
	# redrawn after every executed neural/environment step.
	_recalculate_current_spatial_metrics()
	queue_redraw()

	# Labels and other non-simulation UI only need to refresh twice per second.
	if current_time_us - last_slow_ui_update_us < NON_SIMULATION_UI_UPDATE_INTERVAL_US:
		return

	last_slow_ui_update_us = current_time_us
	var agent_info_update_interval_us: int = maxi(
		NON_SIMULATION_UI_UPDATE_INTERVAL_US,
		int(1_000_000.0 / agent_info_updates_per_second)
	)
	if not show_brain_button.button_pressed and current_time_us - last_agent_info_update_us >= agent_info_update_interval_us:
		last_agent_info_update_us = current_time_us
		_update_selected_agent_info()

	if summary_elapsed_seconds >= summary_update_interval:
		summary_elapsed_seconds = 0.0
		_update_summary()

func _on_single_elite_toggled(_enabled: bool) -> void:
	# The next offspring generation snapshots this setting before GPU construction.
	_update_single_elite_button()


func _update_single_elite_button() -> void:
	# The toggle state is also visible without opening the tooltip.
	single_elite_button.text = "Single: ON" if single_elite_button.button_pressed else "Single: OFF"


func _on_elite_toggled(enabled: bool) -> void:
	elite_button.text = "ON" if enabled else "OFF"
	if enabled and selected_agent_index >= 0 and selected_agent_index not in previous_generation_elite_indices:
		selected_agent_index = -1
		_update_selected_agent_info()
	queue_redraw()


# The preview never advances a second brain. It observes one agent in the
# population's existing GPU states. OFF performs no GPU readbacks or 3D updates.
func _on_show_brain_toggled(enabled: bool) -> void:
	show_brain_button.text = "ON" if enabled else "OFF"
	agent_info_label.visible = not enabled
	agent_info_label_2.visible = not enabled
	agent_info_label_3.visible = not enabled
	if enabled:
		# No SubViewport (or its 3D resources) exists while OFF.
		agent_brain_preview = BRAIN_PREVIEW_SCENE.instantiate() as SubViewportContainer
		add_child(agent_brain_preview)
		agent_brain_preview.request_render()
		_update_brain_preview(true)
	else:
		_invalidate_brain_preview()
		if is_instance_valid(agent_brain_preview):
			agent_brain_preview.queue_free()
		agent_brain_preview = null
		# Restore the original textual information and its update cadence.
		_update_selected_agent_info()


func _invalidate_brain_preview() -> void:
	preview_agent_index = -1
	preview_backend = null
	last_brain_preview_update_us = 0
	if agent_brain_preview != null and is_instance_valid(agent_brain_preview):
		agent_brain_preview.clear_brain()


func _update_brain_preview(force: bool = false) -> void:
	if not show_brain_button.button_pressed:
		return
	var valid_agent: bool = (
		gpu_backend != null
		and selected_agent_index >= 0
		and selected_agent_index < agents.size()
		and selected_agent_index < population_brains.size()
		and selected_agent_index < gpu_backend.population_size
	)
	if not valid_agent:
		if preview_agent_index != -1:
			_invalidate_brain_preview()
		return
	if preview_backend != gpu_backend or preview_agent_index != selected_agent_index:
		agent_brain_preview.set_brain(population_brains[selected_agent_index])
		preview_backend = gpu_backend
		preview_agent_index = selected_agent_index
		force = true
	var now_us: int = Time.get_ticks_usec()
	if not force and now_us - last_brain_preview_update_us < BRAIN_PREVIEW_UPDATE_INTERVAL_US:
		return
	last_brain_preview_update_us = now_us
	var signals: PackedFloat32Array = gpu_backend.read_visual_signals(selected_agent_index)
	if not signals.is_empty():
		agent_brain_preview.update_signals(signals)


func _agent_is_visible(agent_index: int) -> bool:
	return not elite_button.button_pressed or agent_index in previous_generation_elite_indices


func _draw() -> void:
	if not is_instance_valid(simulation_space):
		return

	var origin = simulation_space.position
	var simulation_rect = Rect2(origin, simulation_size)
	draw_rect(simulation_rect, simulation_background_color, true)
	draw_rect(simulation_rect, simulation_border_color, false, 2.0)

	if (
		experiment_state != ExperimentState.GENERATING
		and experiment_state != ExperimentState.RUNNING
		and experiment_state != ExperimentState.PAUSED
		and experiment_state != ExperimentState.OFFSPRING
		and experiment_state != ExperimentState.FINISHED
	):
		return

	if target == null:
		return

	if simulation_selector.selected == 3 or simulation_selector.selected == 5:
		_draw_simulation_4_obstacle(origin)
	elif simulation_selector.selected == 4:
		for wall in simulation_5_maze.walls:
			var world_rect: Rect2 = Rect2(origin + wall.position, wall.size)
			draw_rect(world_rect, simulation_4_obstacle_color, true)
			draw_rect(world_rect, simulation_4_obstacle_border_color, false, 2.0)
		# Show the unique corridor solution in red for visual inspection only.
		# Neither this path nor the fitness distance field enters neural inputs.
		if simulation_5_maze.solution_path.size() >= 2:
			var route_points := PackedVector2Array()
			for waypoint in simulation_5_maze.solution_path:
				route_points.append(origin + waypoint)
			draw_polyline(route_points, Color(1.0, 0.12, 0.12, 0.95), 3.0, true)

	var target_size = Vector2(target_square_size, target_square_size)
	var target_rect = Rect2(origin + target.position - target_size * 0.5, target_size)
	draw_rect(target_rect, Color.GREEN, true)

	var agent_size = Vector2(agent_square_size, agent_square_size)
	for agent_index in agents.size():
		if not _agent_is_visible(agent_index):
			continue
		var agent = agents[agent_index]
		if not agent.state.has("position"):
			continue
		var agent_position: Vector2 = agent.position
		var agent_rect = Rect2(origin + agent_position - agent_size * 0.5, agent_size)
		var agent_color: Color = simulation_elite_agent_color if agent_index in previous_generation_elite_indices else simulation_agent_color
		draw_rect(agent_rect, agent_color, true)
		if agent_index == selected_agent_index:
			draw_rect(agent_rect.grow(2.0), Color.YELLOW, false, 2.0)



func _draw_history_graphs(drawer: Node2D) -> void:
	if not is_instance_valid(simulation_space):
		return

	var origin = simulation_space.position
	_draw_evolution_graph(drawer, origin)
	_draw_genetic_diversity_graph(drawer, origin)


func _queue_history_graph_redraw() -> void:
	if is_instance_valid(history_graphs_layer):
		history_graphs_layer.queue_redraw()


func _draw_evolution_graph(drawer: Node2D, simulation_origin: Vector2) -> void:
	var graph_origin = Vector2(
		simulation_origin.x + (simulation_size.x - evolution_graph_size.x) * 0.5,
		simulation_origin.y + simulation_size.y + evolution_graph_gap
	)
	var graph_rect = Rect2(graph_origin, evolution_graph_size)
	drawer.draw_rect(graph_rect, evolution_graph_background_color, true)
	drawer.draw_rect(graph_rect, evolution_graph_border_color, false, 2.0)

	var margin_left: float = 46.0
	var margin_right: float = 14.0
	var margin_top: float = 18.0
	var margin_bottom: float = 28.0
	var plot_origin = graph_origin + Vector2(margin_left, margin_top)
	var plot_size = Vector2(
		evolution_graph_size.x - margin_left - margin_right,
		evolution_graph_size.y - margin_top - margin_bottom
	)

	drawer.draw_line(plot_origin, plot_origin + Vector2(0.0, plot_size.y), evolution_graph_border_color, 1.0)
	drawer.draw_line(plot_origin + Vector2(0.0, plot_size.y), plot_origin + plot_size, evolution_graph_border_color, 1.0)

	if evolution_graph_mean_points_cache.is_empty():
		return

	if evolution_graph_mean_points_cache.size() == 1:
		drawer.draw_circle(evolution_graph_mean_points_cache[0], 2.5, evolution_graph_mean_color)
		drawer.draw_circle(evolution_graph_best_points_cache[0], 2.5, evolution_graph_best_color)
	else:
		drawer.draw_polyline(evolution_graph_mean_points_cache, evolution_graph_mean_color, 2.0, true)
		drawer.draw_polyline(evolution_graph_best_points_cache, evolution_graph_best_color, 2.0, true)

	var default_font: Font = ThemeDB.fallback_font
	var font_size: int = 14
	drawer.draw_string(default_font, graph_origin + Vector2(6.0, 14.0), "Fitness evolution", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_string(default_font, graph_origin + Vector2(margin_left, evolution_graph_size.y - 6.0), "Generation", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_string(default_font, graph_origin + Vector2(6.0, margin_top + 8.0), "%.3f" % evolution_graph_maximum_value_cache, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_string(default_font, graph_origin + Vector2(6.0, margin_top + plot_size.y), "%.3f" % evolution_graph_minimum_value_cache, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)

	var legend_x: float = graph_origin.x + evolution_graph_size.x - 154.0
	var legend_y: float = graph_origin.y + 14.0
	drawer.draw_line(Vector2(legend_x, legend_y - 5.0), Vector2(legend_x + 20.0, legend_y - 5.0), evolution_graph_mean_color, 2.0)
	drawer.draw_string(default_font, Vector2(legend_x + 26.0, legend_y), "Mean", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_line(Vector2(legend_x + 77.0, legend_y - 5.0), Vector2(legend_x + 97.0, legend_y - 5.0), evolution_graph_best_color, 2.0)
	drawer.draw_string(default_font, Vector2(legend_x + 103.0, legend_y), "Best", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)


func _draw_genetic_diversity_graph(drawer: Node2D, simulation_origin: Vector2) -> void:
	var graph_origin = Vector2(
		simulation_origin.x + (simulation_size.x - evolution_graph_size.x) * 0.5,
		simulation_origin.y + simulation_size.y + evolution_graph_gap * 2.0 + evolution_graph_size.y
	)
	var graph_rect = Rect2(graph_origin, evolution_graph_size)
	drawer.draw_rect(graph_rect, evolution_graph_background_color, true)
	drawer.draw_rect(graph_rect, evolution_graph_border_color, false, 2.0)

	var margin_left: float = 62.0
	var margin_right: float = 14.0
	var margin_top: float = 18.0
	var margin_bottom: float = 28.0
	var plot_origin = graph_origin + Vector2(margin_left, margin_top)
	var plot_size = Vector2(
		evolution_graph_size.x - margin_left - margin_right,
		evolution_graph_size.y - margin_top - margin_bottom
	)

	drawer.draw_line(plot_origin, plot_origin + Vector2(0.0, plot_size.y), evolution_graph_border_color, 1.0)
	drawer.draw_line(plot_origin + Vector2(0.0, plot_size.y), plot_origin + plot_size, evolution_graph_border_color, 1.0)

	if genetic_diversity_graph_points_cache.is_empty() and elite_genetic_diversity_graph_points_cache.is_empty():
		return

	var log_range: float = -float(genetic_diversity_minimum_exponent_cache)
	var default_font: Font = ThemeDB.fallback_font
	var font_size: int = 14

	for exponent in range(0, genetic_diversity_minimum_exponent_cache - 1, -1):
		var decade_y: float = plot_origin.y + plot_size.y * (-float(exponent)) / log_range
		drawer.draw_line(Vector2(plot_origin.x, decade_y), Vector2(plot_origin.x + plot_size.x, decade_y), Color(1.0, 1.0, 1.0, 0.12), 1.0)
		var decade_label: String = "1" if exponent == 0 else "1e%d" % exponent
		var label_y: float = decade_y + (10.0 if exponent == 0 else 5.0)
		drawer.draw_string(default_font, Vector2(graph_origin.x + 6.0, label_y), decade_label, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)

	if genetic_diversity_graph_points_cache.size() == 1:
		drawer.draw_circle(genetic_diversity_graph_points_cache[0], 2.5, evolution_graph_mean_color)
	elif genetic_diversity_graph_points_cache.size() > 1:
		drawer.draw_polyline(genetic_diversity_graph_points_cache, evolution_graph_mean_color, 2.0, true)

	if elite_genetic_diversity_graph_points_cache.size() == 1:
		drawer.draw_circle(elite_genetic_diversity_graph_points_cache[0], 2.5, genetic_diversity_elite_color)
	elif elite_genetic_diversity_graph_points_cache.size() > 1:
		drawer.draw_polyline(elite_genetic_diversity_graph_points_cache, genetic_diversity_elite_color, 2.0, true)

	var turnover_suffix: String = "(%+d)" % current_elite_turnover_count if current_elite_turnover_count > 0 else "(0)"
	drawer.draw_string(default_font, graph_origin + Vector2(6.0, 14.0), "Genetic diversity (normalized Hamming distance, log scale) %s" % turnover_suffix, HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_string(default_font, graph_origin + Vector2(margin_left, evolution_graph_size.y - 6.0), "Generation", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	var legend_x: float = graph_origin.x + evolution_graph_size.x - 154.0
	var legend_y: float = graph_origin.y + 14.0
	drawer.draw_line(Vector2(legend_x, legend_y - 5.0), Vector2(legend_x + 20.0, legend_y - 5.0), evolution_graph_mean_color, 2.0)
	drawer.draw_string(default_font, Vector2(legend_x + 26.0, legend_y), "Mean", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_line(Vector2(legend_x + 77.0, legend_y - 5.0), Vector2(legend_x + 97.0, legend_y - 5.0), genetic_diversity_elite_color, 2.0)
	drawer.draw_string(default_font, Vector2(legend_x + 103.0, legend_y), "Best", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)


func _rebuild_history_graph_caches() -> void:
	_rebuild_evolution_graph_cache()
	_rebuild_genetic_diversity_graph_cache()


func _rebuild_evolution_graph_cache() -> void:
	evolution_graph_mean_points_cache = PackedVector2Array()
	evolution_graph_best_points_cache = PackedVector2Array()

	if evolution_mean_fitness_history.is_empty() or not is_instance_valid(simulation_space):
		return

	var mean_history: PackedFloat64Array = evolution_mean_fitness_history
	var best_history: PackedFloat64Array = evolution_best_fitness_history

	match evolution_graph_selector.selected:
		1:
			mean_history = _moving_average(evolution_mean_fitness_history, 10)
			best_history = _moving_average(evolution_best_fitness_history, 10)
		2:
			mean_history = _moving_average(evolution_mean_fitness_history, 50)
			best_history = _moving_average(evolution_best_fitness_history, 50)

	var minimum_value: float = INF
	var maximum_value: float = -INF
	for index in mean_history.size():
		minimum_value = minf(minimum_value, mean_history[index])
		minimum_value = minf(minimum_value, best_history[index])
		maximum_value = maxf(maximum_value, mean_history[index])
		maximum_value = maxf(maximum_value, best_history[index])

	if is_equal_approx(minimum_value, maximum_value):
		minimum_value -= 0.01
		maximum_value += 0.01

	evolution_graph_minimum_value_cache = minimum_value
	evolution_graph_maximum_value_cache = maximum_value

	var graph_origin = Vector2(
		simulation_space.position.x + (simulation_size.x - evolution_graph_size.x) * 0.5,
		simulation_space.position.y + simulation_size.y + evolution_graph_gap
	)
	var plot_origin = graph_origin + Vector2(46.0, 18.0)
	var plot_size = Vector2(
		evolution_graph_size.x - 46.0 - 14.0,
		evolution_graph_size.y - 18.0 - 28.0
	)
	var value_range: float = maximum_value - minimum_value
	var point_count: int = mean_history.size()
	evolution_graph_mean_points_cache.resize(point_count)
	evolution_graph_best_points_cache.resize(point_count)

	for index in point_count:
		var x: float = plot_origin.x if point_count == 1 else plot_origin.x + plot_size.x * float(index) / float(point_count - 1)
		var mean_y: float = plot_origin.y + plot_size.y * (1.0 - (mean_history[index] - minimum_value) / value_range)
		var best_y: float = plot_origin.y + plot_size.y * (1.0 - (best_history[index] - minimum_value) / value_range)
		evolution_graph_mean_points_cache[index] = Vector2(x, mean_y)
		evolution_graph_best_points_cache[index] = Vector2(x, best_y)


func _rebuild_genetic_diversity_graph_cache() -> void:
	genetic_diversity_graph_points_cache = PackedVector2Array()
	elite_genetic_diversity_graph_points_cache = PackedVector2Array()

	if genetic_diversity_history.is_empty() or not is_instance_valid(simulation_space):
		return

	var diversity_history: PackedFloat64Array = genetic_diversity_history
	var elite_history: PackedFloat64Array = elite_genetic_diversity_history
	match genetic_diversity_graph_selector.selected:
		1:
			diversity_history = _moving_average(genetic_diversity_history, 10)
			elite_history = _moving_average_available(elite_genetic_diversity_history, 10)
		2:
			diversity_history = _moving_average(genetic_diversity_history, 50)
			elite_history = _moving_average_available(elite_genetic_diversity_history, 50)

	# Both series share the same logarithmic axis. Missing elite history in
	# legacy saves is marked -1 (not a fabricated zero).
	var minimum_positive: float = 1.0
	for value in diversity_history:
		if value > 0.0:
			minimum_positive = minf(minimum_positive, value)
	for value in elite_history:
		if value > 0.0:
			minimum_positive = minf(minimum_positive, value)

	genetic_diversity_minimum_exponent_cache = int(floor(log(minimum_positive) / log(10.0)))
	genetic_diversity_minimum_exponent_cache = min(genetic_diversity_minimum_exponent_cache, -1)
	var minimum_log_value: float = pow(10.0, float(genetic_diversity_minimum_exponent_cache))
	var log_range: float = -float(genetic_diversity_minimum_exponent_cache)

	var graph_origin = Vector2(
		simulation_space.position.x + (simulation_size.x - evolution_graph_size.x) * 0.5,
		simulation_space.position.y + simulation_size.y + evolution_graph_gap * 2.0 + evolution_graph_size.y
	)
	var plot_origin = graph_origin + Vector2(62.0, 18.0)
	var plot_size = Vector2(
		evolution_graph_size.x - 62.0 - 14.0,
		evolution_graph_size.y - 18.0 - 28.0
	)
	var point_count: int = diversity_history.size()
	genetic_diversity_graph_points_cache.resize(point_count)

	for index in point_count:
		var x: float = plot_origin.x if point_count == 1 else plot_origin.x + plot_size.x * float(index) / float(point_count - 1)
		var value: float = maxf(diversity_history[index], minimum_log_value)
		var log_value: float = log(value) / log(10.0)
		var y: float = plot_origin.y + plot_size.y * (-log_value) / log_range
		genetic_diversity_graph_points_cache[index] = Vector2(x, y)

		if index < elite_history.size() and elite_history[index] >= 0.0:
			var elite_value: float = maxf(elite_history[index], minimum_log_value)
			var elite_log_value: float = log(elite_value) / log(10.0)
			var elite_y: float = plot_origin.y + plot_size.y * (-elite_log_value) / log_range
			elite_genetic_diversity_graph_points_cache.append(Vector2(x, elite_y))


func _flatten_genome(genome: Dictionary) -> PackedByteArray:
	var flattened: PackedByteArray = PackedByteArray()

	for gene_name in BRAIN_EXPLORER_GENE_ORDER:
		for digit in genome[gene_name]:
			flattened.append(int(digit))

	return flattened


func _genome_signature(genome: Dictionary) -> String:
	return Marshalls.raw_to_base64(_flatten_genome(genome))


func _elite_signatures_from_indices(indices: Array[int]) -> Array[String]:
	var signatures: Array[String] = []
	for elite_index in indices:
		if elite_index >= 0 and elite_index < population_genomes.size():
			signatures.append(_genome_signature(population_genomes[elite_index]))
	return signatures


func _elite_turnover_count(previous_signatures: Array[String], current_signatures: Array[String]) -> int:
	if previous_signatures.is_empty() or current_signatures.is_empty():
		return 0
	var previous_counts: Dictionary = {}
	for signature in previous_signatures:
		previous_counts[signature] = int(previous_counts.get(signature, 0)) + 1
	var shared_count: int = 0
	for signature in current_signatures:
		var remaining: int = int(previous_counts.get(signature, 0))
		if remaining > 0:
			shared_count += 1
			previous_counts[signature] = remaining - 1
	return maxi(0, current_signatures.size() - shared_count)


func _population_genetic_diversity() -> float:
	var included_agent_indices: Array[int] = []
	for agent_index in population_genomes.size():
		if agent_index not in genetic_diversity_excluded_agent_indices:
			included_agent_indices.append(agent_index)
	return _mean_genetic_diversity_for_indices(included_agent_indices)


func _elite_genetic_diversity() -> float:
	# Rank by the completed generation's fitness, not by the previous elites.
	return _mean_genetic_diversity_for_indices(_get_top_three_agent_indices())


func _mean_genetic_diversity_for_indices(agent_indices: Array[int]) -> float:
	if agent_indices.size() < 2:
		return 0.0

	var flattened_genomes: Array = []
	flattened_genomes.resize(agent_indices.size())
	for included_index in agent_indices.size():
		flattened_genomes[included_index] = _flatten_genome(population_genomes[agent_indices[included_index]])

	var genome_length: int = flattened_genomes[0].size()
	if genome_length == 0:
		return 0.0
	var total_normalized_distance: float = 0.0
	var pair_count: int = 0

	for first_index in flattened_genomes.size() - 1:
		var first_genome: PackedByteArray = flattened_genomes[first_index]
		for second_index in range(first_index + 1, flattened_genomes.size()):
			var second_genome: PackedByteArray = flattened_genomes[second_index]
			var hamming_distance: int = 0
			for gene_index in genome_length:
				if first_genome[gene_index] != second_genome[gene_index]:
					hamming_distance += 1
			total_normalized_distance += float(hamming_distance) / float(genome_length)
			pair_count += 1

	return total_normalized_distance / float(pair_count)


func _moving_average_available(values: PackedFloat64Array, window_size: int) -> PackedFloat64Array:
	# -1 means "not recorded" in saves predating the elite metric.
	var averaged_values: PackedFloat64Array = PackedFloat64Array()
	averaged_values.resize(values.size())
	var running_sum: float = 0.0
	var valid_count: int = 0
	for index in values.size():
		if values[index] >= 0.0:
			running_sum += values[index]
			valid_count += 1
		if index >= window_size and values[index - window_size] >= 0.0:
			running_sum -= values[index - window_size]
			valid_count -= 1
		averaged_values[index] = running_sum / float(valid_count) if valid_count > 0 else -1.0
	return averaged_values


func _moving_average(values: PackedFloat64Array, window_size: int) -> PackedFloat64Array:
	var averaged_values: PackedFloat64Array = PackedFloat64Array()
	averaged_values.resize(values.size())
	var running_sum: float = 0.0

	for index in values.size():
		running_sum += values[index]
		if index >= window_size:
			running_sum -= values[index - window_size]

		var sample_count: int = mini(index + 1, window_size)
		averaged_values[index] = running_sum / float(sample_count)

	return averaged_values



func _on_start_pressed() -> void:
	if _simulation_is_active():
		_request_reset()
		return

	# Commit the SpinBox's current value before starting a new population.
	_on_num_agents_value_changed(num_agents_spinbox.value)
	var parameters = _read_parameters()
	if parameters.is_empty():
		return

	current_brain_size = parameters["size"]
	experiment_duration_seconds = float(parameters["generation_time"])
	mutation_probability = float(parameters["mutation_probability"])
	connections_density = float(parameters["connections_density"])

	var loaded_population_io_compatible = (
		loaded_population_num_inputs == num_inputs
		and loaded_population_num_outputs == num_outputs
	)
	var can_use_loaded_population = (
		population_from_load
		and loaded_population_io_compatible
		and loaded_population_size == current_brain_size
		and is_equal_approx(loaded_population_density, connections_density)
		and population_genomes.size() == agent_count
	)

	if can_use_loaded_population:
		# Switching experiments frees the former neural backend and its GPU
		# buffers; a loaded population may therefore have genotypes but no
		# live phenotypes. Rebuild only when starting a compatible experiment.
		if population_brains.size() != agent_count:
			var rebuilt: Dictionary = _rebuild_population_from_genomes(
				population_genomes, current_brain_size, num_inputs, num_outputs
			)
			var rebuild_error: String = str(rebuilt.get("error", ""))
			if not rebuild_error.is_empty():
				summary_label.text = "Could not rebuild saved GPU population:\n" + rebuild_error
				_refresh_buttons()
				return
			population_brains = rebuilt["brains"]
		_start_experiment_from_population()
		return

	if population_from_load and not loaded_population_io_compatible:
		summary_label.text = (
			"The loaded population is incompatible with the selected simulation.\n"
			+ "Population: %d inputs / %d outputs\n" % [loaded_population_num_inputs, loaded_population_num_outputs]
			+ "Simulation: %d inputs / %d outputs" % [num_inputs, num_outputs]
		)
		_refresh_buttons()
		return

	_clear_runtime_state()
	_reset_experiment_entities()
	population_from_load = false
	_reset_evolution_statistics()
	population_brains.clear()
	population_genomes.clear()
	_begin_population_generation()


func _on_pause_pressed() -> void:
	if experiment_state == ExperimentState.RUNNING:
		if free_mode_active:
			free_mode_active = false
			target_dragging = false
			free_mode_button.text = "Free Mode"

		experiment_state = ExperimentState.PAUSED
		pause_button.text = "Continue"
		_update_summary()
		_refresh_buttons()
		return

	if experiment_state == ExperimentState.PAUSED:
		experiment_state = ExperimentState.RUNNING
		process_frames_to_skip = 0
		next_step_due_us = 0
		previous_raw_step_start_us = 0
		speed_stats_window_start_us = 0
		speed_stats_window_steps = 0
		effective_steps_per_second = 0.0
		pause_button.text = "Pause"
		_update_summary()
		_refresh_buttons()


func _on_free_mode_pressed() -> void:
	if experiment_state != ExperimentState.RUNNING:
		return

	free_mode_active = not free_mode_active
	target_dragging = false
	free_mode_button.text = "Exit Free Mode" if free_mode_active else "Free Mode"
	_update_summary()
	_refresh_buttons()
	queue_redraw()


func _input(event: InputEvent) -> void:
	if experiment_state != ExperimentState.RUNNING and experiment_state != ExperimentState.PAUSED:
		target_dragging = false
		return

	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed and _mouse_is_over_interactive_control():
			return

		var mouse_position: Vector2 = simulation_space.to_local(get_global_mouse_position())

		if event.pressed:
			if free_mode_active and simulation_selector.selected != 4:
				var half_target_size: float = target_square_size * 0.5
				var target_rect = Rect2(
					target.position - Vector2(half_target_size, half_target_size),
					Vector2(target_square_size, target_square_size)
				)

				if target_rect.has_point(mouse_position):
					target_dragging = true
					target_drag_offset = mouse_position - target.position
					get_viewport().set_input_as_handled()
					return

			_select_agent_at(mouse_position)
		else:
			target_dragging = false

	elif event is InputEventMouseMotion and free_mode_active and target_dragging and simulation_selector.selected != 4:
		var half_target_size: float = target_square_size * 0.5
		var mouse_position: Vector2 = simulation_space.to_local(get_global_mouse_position())
		var dragged_position: Vector2 = mouse_position - target_drag_offset
		target.position = Vector2(
			clampf(dragged_position.x, half_target_size, simulation_size.x - half_target_size),
			clampf(dragged_position.y, half_target_size, simulation_size.y - half_target_size)
		)
		if simulation_selector.selected == 3 or simulation_selector.selected == 5:
			_simulation_4_update_obstacle(false)
		queue_redraw()
		get_viewport().set_input_as_handled()


func _mouse_is_over_interactive_control() -> bool:
	var hovered_control: Control = get_viewport().gui_get_hovered_control()
	if hovered_control is BaseButton or hovered_control is LineEdit:
		return true
	# Clicking the preview is not a request to deselect the simulated agent.
	return (
		show_brain_button.button_pressed
		and agent_brain_preview != null
		and agent_brain_preview.get_global_rect().has_point(get_viewport().get_mouse_position())
	)


func _select_agent_at(mouse_position: Vector2) -> void:
	var half_agent_size: float = agent_square_size * 0.5
	var nearest_agent: int = -1
	var nearest_distance_squared: float = INF

	for agent_index in agents.size():
		if not _agent_is_visible(agent_index):
			continue
		var agent = agents[agent_index]
		if not agent.state.has("position"):
			continue
		var agent_position: Vector2 = agent.position
		var difference: Vector2 = mouse_position - agent_position

		if absf(difference.x) <= half_agent_size and absf(difference.y) <= half_agent_size:
			var distance_squared: float = difference.length_squared()
			if distance_squared < nearest_distance_squared:
				nearest_distance_squared = distance_squared
				nearest_agent = agent_index

	selected_agent_index = nearest_agent
	_update_selected_agent_info()
	queue_redraw()


func _update_selected_agent_info() -> void:
	if show_brain_button.button_pressed:
		_update_brain_preview(true)
		return
	if (
		selected_agent_index < 0
		or selected_agent_index >= agents.size()
		or selected_agent_index >= agent_dynamics.size()
	):
		agent_info_label.text = "Click an agent to inspect its neural signals."
		agent_info_label_2.text = ""
		agent_info_label_3.text = ""
		return

	var agent_info: Dictionary = agent_info_behavior.call(selected_agent_index)
	agent_info_label.text = str(agent_info["main"])
	agent_info_label_2.text = str(agent_info["state"])
	agent_info_label_3.text = str(agent_info["signal"])


# Only valid edits affect future offspring generations. A generation already
# being built keeps the mutation probability captured in its deferred call.
func _try_update_mutation_probability(value_text: String) -> bool:
	var normalized_text: String = value_text.strip_edges()
	if not normalized_text.is_valid_float():
		return false

	var new_probability: float = float(normalized_text)
	if is_nan(new_probability) or is_inf(new_probability) or new_probability < 0.0 or new_probability > 1.0:
		return false

	mutation_probability = new_probability
	return true


func _on_mutation_probability_text_changed(new_text: String) -> void:
	_try_update_mutation_probability(new_text)


func _on_mutation_probability_text_submitted(_new_text: String) -> void:
	_on_mutation_probability_focus_exited()


func _on_mutation_probability_focus_exited() -> void:
	if _try_update_mutation_probability(mutation_probability_field.text):
		return
	mutation_probability_field.text = str(mutation_probability)
	summary_label.text = "Mutation Probability must be a finite number between 0 and 1. Previous value restored."


func _read_parameters() -> Dictionary:
	var i_text = dimension_i_field.text.strip_edges()
	var j_text = dimension_j_field.text.strip_edges()
	var k_text = dimension_k_field.text.strip_edges()
	var generation_time_text = generation_time_field.text.strip_edges()
	var mutation_probability_text = mutation_probability_field.text.strip_edges()
	var connections_density_text = connections_density_field.text.strip_edges()

	if not i_text.is_valid_int() or not j_text.is_valid_int() or not k_text.is_valid_int():
		summary_label.text = "Dimensions I, J and K must be integers."
		return {}

	if not generation_time_text.is_valid_float():
		summary_label.text = "Generation Time must be a valid number in seconds."
		return {}

	if not mutation_probability_text.is_valid_float():
		summary_label.text = "Mutation Probability must be a valid number."
		return {}

	if not connections_density_text.is_valid_float():
		summary_label.text = "Connections Density must be a valid positive number."
		return {}

	var size = Vector3i(int(i_text), int(j_text), int(k_text))
	var generation_time = float(generation_time_text)
	var selected_mutation_probability = float(mutation_probability_text)
	var selected_density: float = float(connections_density_text)

	if size.x < 2 or size.y < 2 or size.z < 2:
		summary_label.text = "Dimensions I, J and K must all be at least 2."
		return {}

	if size.x * size.y * size.z < num_inputs + num_outputs:
		summary_label.text = "This simulation requires at least %d neurons (%d inputs and %d outputs)." % [
			num_inputs + num_outputs, num_inputs, num_outputs]
		return {}

	if generation_time <= 0.0:
		summary_label.text = "Generation Time must be greater than zero."
		return {}

	if is_nan(selected_mutation_probability) or is_inf(selected_mutation_probability) or selected_mutation_probability < 0.0 or selected_mutation_probability > 1.0:
		summary_label.text = "Mutation Probability must be a finite number between 0 and 1."
		return {}

	if is_nan(selected_density) or is_inf(selected_density) or selected_density <= 0.0:
		summary_label.text = "Connections Density (eta) must be a finite number greater than zero."
		return {}

	if simulation_size.x <= agent_square_size or simulation_size.y <= agent_square_size:
		summary_label.text = "Simulation size must be larger than the agent square size."
		return {}

	return {
		"size": size,
		"generation_time": generation_time,
		"mutation_probability": selected_mutation_probability,
		"connections_density": selected_density,
	}


func _genome_dictionary(genome: Genome) -> Dictionary:
	var genes := {}
	for gene_name in Offspring.GENE_NAMES:
		genes[String(gene_name)] = genome.get(gene_name).duplicate()
	return genes


func _genome_object(data: Dictionary) -> Genome:
	var genome := Genome.new()
	for gene_name in Offspring.GENE_NAMES:
		var key: String = String(gene_name)
		# Old saved genomes may contain the obsolete role gene; ignore it.
		if not data.has(key):
			return null
		var raw: Array = data[key]
		var values: Array[int] = []
		for digit in raw:
			values.append(int(digit))
		genome.set(String(gene_name), values)
	return genome


func _gpu_build(builder: BrainBuilderGPU, genome: Genome, size: Vector3i, input_count: int, output_count: int, progress: Callable, shared_rd: RenderingDevice) -> Dictionary:
	var brain: Dictionary = builder.build(_genome_dictionary(genome), size, input_count, output_count, progress, true, shared_rd, connections_density, BrainBuilderGPU.STATS_WEIGHT_SUM)
	if not bool(brain.get("valid", false)):
		builder.release_gpu_state(brain)
		return {}
	return brain


func _dispose_generation_population(result: Dictionary, shared_rd: RenderingDevice) -> Dictionary:
	var builder := BrainBuilderGPU.new()
	for brain in result.get("brains", []):
		if brain is Dictionary:
			builder.release_gpu_state(brain)
	result["brains"] = []
	if shared_rd != null:
		shared_rd.free()
	if _generation_rd == shared_rd:
		_generation_rd = null
		_generation_partial_result = {}
	return result


func _begin_population_generation() -> void:
	experiment_state = ExperimentState.GENERATING
	generation_progress_mutex.lock()
	generation_cancel_requested = false
	generation_agent_number = 1
	generation_brain_progress = 0
	generation_attempt_number = 1
	generation_progress_mutex.unlock()
	last_generation_ui_update_us = 0

	generation_active = true
	call_deferred("_run_initial_generation_async", current_brain_size,
		max_brain_generation_attempts, num_inputs, num_outputs)

	_refresh_buttons()
	queue_redraw()


func _generate_population_incremental(size: Vector3i, maximum_attempts: int, selected_num_inputs: int, selected_num_outputs: int) -> Dictionary:
	var result := {"brains": [], "genomes": [], "genetic_diversity_excluded_agent_indices": [], "error": ""}
	var rd: RenderingDevice = RenderingServer.create_local_rendering_device()
	if rd == null:
		result.error = "Failed to create a shared local RenderingDevice."
		return result
	_generation_rd = rd
	_generation_partial_result = result
	var builder := BrainBuilderGPU.new()
	for agent_index in agent_count:
		if _is_generation_cancel_requested():
			return _dispose_generation_population(result, rd)
		var accepted := false
		for attempt_index in maximum_attempts:
			if _is_generation_cancel_requested():
				return _dispose_generation_population(result, rd)
			_set_generation_progress(0, agent_index + 1, attempt_index + 1)
			var genome := Genome.new()
			var callback := _set_generation_progress.bind(agent_index + 1, attempt_index + 1)
			var brain: Dictionary = _gpu_build(builder, genome, size, selected_num_inputs, selected_num_outputs, callback, rd)
			if not brain.is_empty():
				result.brains.append(brain)
				result.genomes.append(_genome_dictionary(genome))
				accepted = true
			# Register any allocated GPU phenotype before suspending; cancellation
			# and scene teardown then have a complete resource ownership list.
			await get_tree().process_frame
			if accepted:
				break
		if not accepted:
			result.error = "Could not generate a valid GPU brain for agent %d after %d attempts." % [agent_index + 1, maximum_attempts]
			return _dispose_generation_population(result, rd)
	return result


func _begin_offspring_generation(top_indices: Array[int]) -> void:
	_release_gpu_population()
	agent_dynamics.clear()
	var current_elite_signatures: Array[String] = _elite_signatures_from_indices(top_indices)
	current_elite_turnover_count = _elite_turnover_count(previous_generation_elite_signatures, current_elite_signatures)
	previous_generation_elite_signatures = current_elite_signatures
	previous_generation_elite_indices = top_indices.duplicate()
	if elite_button.button_pressed and selected_agent_index not in previous_generation_elite_indices:
		selected_agent_index = -1
		_update_selected_agent_info()
	experiment_state = ExperimentState.OFFSPRING
	generation_progress_mutex.lock()
	generation_cancel_requested = false
	generation_agent_number = 1
	generation_brain_progress = 0
	generation_attempt_number = 1
	generation_progress_mutex.unlock()
	last_generation_ui_update_us = 0

	var parent_genomes: Array = population_genomes.duplicate(true)
	# Rank all non-elites by their complete-generation fitness before
	# beginning incremental GPU construction, so only the bottom five are replaced.
	var ranked_non_elite_indices: Array[int] = []
	for agent_index in agent_count:
		if agent_index not in top_indices:
			ranked_non_elite_indices.append(agent_index)
	ranked_non_elite_indices.sort_custom(_has_higher_generation_fitness)

	generation_active = true
	call_deferred("_run_offspring_generation_async", parent_genomes, top_indices,
		ranked_non_elite_indices, current_brain_size, max_brain_generation_attempts,
		mutation_probability, num_inputs, num_outputs, single_elite_button.button_pressed)

	_refresh_buttons()
	queue_redraw()


func _generate_offspring_population_incremental(
		parent_genomes: Array, top_indices: Array[int], ranked_non_elite_indices: Array[int],
		size: Vector3i, maximum_attempts: int, selected_mutation_probability: float,
		selected_num_inputs: int, selected_num_outputs: int, selected_single_elite: bool) -> Dictionary:
	var result := {"brains": [], "genomes": [], "genetic_diversity_excluded_agent_indices": [], "error": ""}
	if parent_genomes.size() != agent_count or top_indices.size() != 3 or ranked_non_elite_indices.size() != agent_count-3:
		result.error = "Incomplete parent population."
		return result
	var rd: RenderingDevice = RenderingServer.create_local_rendering_device()
	if rd == null:
		result.error = "Could not create GPU device for offspring."
		return result
	_generation_rd = rd
	_generation_partial_result = result
	result.brains.resize(agent_count)
	result.genomes.resize(agent_count)
	var builder := BrainBuilderGPU.new()
	var offspring := Offspring.new()
	var parents: Array = []
	for saved in parent_genomes:
		var genome: Genome = _genome_object(saved)
		if genome == null:
			result.error = "Invalid parent genotype."
			return _dispose_generation_population(result, rd)
		parents.append(genome)
	# Preserve the three elite genotypes; lifetime plasticity is not inherited.
	for elite_rank in 3:
		if _is_generation_cancel_requested():
			return _dispose_generation_population(result, rd)
		var index: int = top_indices[elite_rank]
		_set_generation_progress(0, elite_rank+1, 1)
		var brain := _gpu_build(builder, parents[index], size, selected_num_inputs, selected_num_outputs, _set_generation_progress.bind(elite_rank+1,1), rd)
		if brain.is_empty():
			result.error = "Elite genome %d generated an invalid brain." % (elite_rank+1)
			return _dispose_generation_population(result, rd)
		result.brains[index] = brain
		result.genomes[index] = _genome_dictionary(parents[index])
		await get_tree().process_frame
	var eligible: Array[int] = []
	var worst: Array[int] = []
	for rank in ranked_non_elite_indices.size():
		if rank < ranked_non_elite_indices.size()-5:
			eligible.append(ranked_non_elite_indices[rank])
		else:
			worst.append(ranked_non_elite_indices[rank])
	var rng := RandomNumberGenerator.new()
	rng.randomize()
	_shuffle_indices(eligible, rng)
	_shuffle_indices(worst, rng)
	result.genetic_diversity_excluded_agent_indices = worst.duplicate()
	# OFF: historical split 44/29/19 among the three elite parents.
	# ON: all eligible partners cross with the best elite. In both modes, the
	# three elites are preserved and the five renewal slots are unchanged.
	var first_elite_end: int = roundi(float(eligible.size()) * 44.0 / 92.0)
	var second_elite_end: int = roundi(float(eligible.size()) * 73.0 / 92.0)
	for partner_slot in eligible.size():
		if _is_generation_cancel_requested():
			return _dispose_generation_population(result, rd)
		var elite_rank: int = 0 if selected_single_elite or partner_slot < first_elite_end else (1 if partner_slot < second_elite_end else 2)
		var elite_index: int = top_indices[elite_rank]
		var agent_index: int = eligible[partner_slot]
		var done := false
		for attempt_index in maximum_attempts:
			if _is_generation_cancel_requested():
				return _dispose_generation_population(result, rd)
			var sequence: int = partner_slot + 4
			_set_generation_progress(0, sequence, attempt_index+1)
			var child := offspring.generate(parents[elite_index], parents[agent_index], selected_mutation_probability)
			var brain := _gpu_build(builder, child, size, selected_num_inputs, selected_num_outputs, _set_generation_progress.bind(sequence,attempt_index+1), rd)
			if not brain.is_empty():
				result.brains[agent_index] = brain
				result.genomes[agent_index] = _genome_dictionary(child)
				done = true
			await get_tree().process_frame
			if done:
				break
		if not done:
			result.error = "Could not generate offspring for agent %d." % (agent_index+1)
			return _dispose_generation_population(result, rd)
	var immigrant_genomes: Array = []
	var immigrant_order: Array[int] = []
	for index in worst.size():
		immigrant_genomes.append(Genome.new())
		immigrant_order.append(index)
	_shuffle_indices(immigrant_order, rng)
	for slot in worst.size():
		if _is_generation_cancel_requested():
			return _dispose_generation_population(result, rd)
		var index: int = worst[slot]
		var seed_genome: Genome = immigrant_genomes[immigrant_order[slot]]
		var done := false
		for attempt_index in maximum_attempts:
			if _is_generation_cancel_requested():
				return _dispose_generation_population(result, rd)
			var sequence: int = eligible.size() + slot + 4
			_set_generation_progress(0, sequence, attempt_index+1)
			var candidate: Genome
			if slot < 3:
				candidate = offspring.generate(parents[top_indices[slot]], seed_genome, selected_mutation_probability)
			else:
				if attempt_index > 0:
					seed_genome = Genome.new()
				candidate = seed_genome
			var brain := _gpu_build(builder, candidate, size, selected_num_inputs, selected_num_outputs, _set_generation_progress.bind(sequence,attempt_index+1), rd)
			if not brain.is_empty():
				result.brains[index] = brain
				result.genomes[index] = _genome_dictionary(candidate)
				done = true
			await get_tree().process_frame
			if done:
				break
		if not done:
			result.error = "Could not generate immigrant genome for agent %d." % (index+1)
			return _dispose_generation_population(result, rd)
	return result


func _shuffle_indices(indices: Array[int], rng: RandomNumberGenerator) -> void:
	for index in range(indices.size() - 1, 0, -1):
		var swap_index: int = rng.randi_range(0, index)
		var temporary: int = indices[index]
		indices[index] = indices[swap_index]
		indices[swap_index] = temporary


func _set_generation_progress(percent: int, agent_number: int, attempt_number: int) -> void:
	generation_progress_mutex.lock()
	generation_agent_number = agent_number
	generation_brain_progress = percent
	generation_attempt_number = attempt_number
	generation_progress_mutex.unlock()


func _is_generation_cancel_requested() -> bool:
	generation_progress_mutex.lock()
	var cancel_requested = generation_cancel_requested
	generation_progress_mutex.unlock()
	return cancel_requested


func _simulation_is_active() -> bool:
	return (
		generation_active
		or experiment_state == ExperimentState.GENERATING
		or experiment_state == ExperimentState.RUNNING
		or experiment_state == ExperimentState.PAUSED
		or experiment_state == ExperimentState.OFFSPRING
	)


func _request_reset() -> void:
	if generation_active:
		reset_pending = true
		generation_progress_mutex.lock()
		generation_cancel_requested = true
		generation_progress_mutex.unlock()
		summary_label.text = "Resetting simulation..."
		_refresh_buttons()
		return
	_complete_reset()


func _complete_reset() -> void:
	reset_pending = false
	generation_progress_mutex.lock()
	generation_cancel_requested = false
	generation_agent_number = 0
	generation_brain_progress = 0
	generation_attempt_number = 0
	generation_progress_mutex.unlock()

	_clear_runtime_state()
	_reset_evolution_statistics()
	population_brains.clear()
	population_genomes.clear()
	population_from_load = false
	loaded_population_size = Vector3i.ZERO
	loaded_population_density = 1.0
	loaded_population_num_inputs = -1
	loaded_population_num_outputs = -1
	current_brain_size = Vector3i.ZERO
	summary_label.text = "Ready."
	_refresh_buttons()
	queue_redraw()


func _start_experiment_from_population(new_generation: bool = true) -> void:
	if population_brains.size() != agent_count:
		experiment_state = ExperimentState.IDLE
		summary_label.text = "The population must contain %d brains." % agent_count
		_refresh_buttons()
		return
	if new_generation:
		generation_number += 1
		target_test_index = 0
		if simulation_selector.selected == 3 or simulation_selector.selected == 5:
			# Visit each difficulty sector once in a new randomized order.
			_simulation_4_shuffle_target_sectors()
		generation_real_start_us = Time.get_ticks_usec()
		generation_real_time_seconds = 0.0
		generation_fitness_sums = PackedFloat64Array()
		generation_fitness_sums.resize(agent_count)
		generation_mean_final_score_sum = 0.0
		generation_best_final_score = -1.0
		if simulation_selector.selected == 2:
			# Simulation 3 target speed: half the agent's movement speed.
			simulation_3_target_speed = agent_speed * 0.5
		if heatmap_tracking_enabled:
			_capture_generation_heatmap_phenotypes()
	fitness_reset_behavior.call()
	if new_generation or gpu_backend == null:
		if gpu_backend != null:
			gpu_backend.clear()
		gpu_backend = BrainPopulationGPU.new()
		if not gpu_backend.initialize(population_brains):
			var initialization_error: String = gpu_backend.last_error
			push_error("SimulationsGPU initialization: " + initialization_error)
			gpu_backend.clear()
			gpu_backend = null
			experiment_state = ExperimentState.FINISHED
			summary_label.text = "GPU initialization failed: " + initialization_error
			_refresh_buttons()
			return
		# The neural population API supports arbitrary channel counts. Simulations
		# 1-3 require 4 inputs, 4 needs 5, 5 needs 8 and 6 needs 13;
		# all have 4 outputs.
		for brain_index in agent_count:
			if gpu_backend.input_counts[brain_index] != num_inputs or gpu_backend.output_counts[brain_index] != num_outputs:
				var channel_error: String = "Brain %d has %d inputs / %d outputs; this simulation expects %d / %d." % [
					brain_index + 1, gpu_backend.input_counts[brain_index], gpu_backend.output_counts[brain_index], num_inputs, num_outputs]
				push_error(channel_error)
				gpu_backend.clear()
				gpu_backend = null
				experiment_state = ExperimentState.FINISHED
				summary_label.text = channel_error
				_refresh_buttons()
				return
		agent_dynamics = gpu_backend.dynamics
	else:
		gpu_backend.reset()
	free_mode_active = false
	target_dragging = false
	free_mode_button.text = "Free Mode"
	_reset_experiment_entities()
	experiment_elapsed_seconds = 0.0
	summary_elapsed_seconds = 0.0
	process_frames_to_skip = 0
	next_step_due_us = 0
	previous_raw_step_start_us = 0
	speed_stats_window_start_us = 0
	speed_stats_window_steps = 0
	effective_steps_per_second = 0.0
	last_runtime_ui_update_us = Time.get_ticks_usec()
	last_agent_info_update_us = last_runtime_ui_update_us
	best_score_ever = 0.0
	best_distance_ever = 1.0e30
	experiment_state = ExperimentState.RUNNING
	pause_button.text = "Pause"
	_recalculate_current_spatial_metrics()
	best_score_ever = current_best_score
	best_distance_ever = current_min_distance
	_update_selected_agent_info()
	_update_summary()
	_refresh_buttons()
	queue_redraw()


func _reset_experiment_entities() -> void:
	if target == null or agents.size() != agent_count:
		_setup_simulation(simulation_selector.selected)

	target.reset()
	for agent in agents:
		agent.reset()

	queue_redraw()


func _simulation_1_reset_agent(agent) -> void:
	agent.set_position(simulation_size * 0.5)


func _simulation_1_update_agent_inputs(agent) -> void:
	var agent_position: Vector2 = agent.position
	var normalized_delta = Vector2(
		(target.position.x - agent_position.x) / simulation_size.x,
		(target.position.y - agent_position.y) / simulation_size.y
	)

	agent.input_signals[0] = maxf(0.0, -normalized_delta.x)
	agent.input_signals[1] = maxf(0.0, normalized_delta.x)
	agent.input_signals[2] = maxf(0.0, -normalized_delta.y)
	agent.input_signals[3] = maxf(0.0, normalized_delta.y)


func _simulation_4_reset_agent(agent) -> void:
	_simulation_1_reset_agent(agent)
	agent.state["simulation_4_first_arrival"] = -1.0


func _simulation_4_update_agent_inputs(agent) -> void:
	_simulation_1_update_agent_inputs(agent)
	# Input 5 is touch only; it does not reveal obstacle direction or a route.
	agent.input_signals[4] = 1.0 if _simulation_4_touching_surface(agent.position) else 0.0


func _simulation_6_update_agent_inputs(agent) -> void:
	# Retain all five original Simulation 4 channels, unchanged.
	_simulation_4_update_agent_inputs(agent)
	_simulation_6_write_obstacle_proximity(agent.position, agent)


func _simulation_6_write_obstacle_proximity(position: Vector2, agent) -> void:
	# Match the inflation used in the Simulation 4 swept-collision solver.
	# The obstacle may move in Free Mode, so read its current transform.
	var half_agent: float = agent_square_size * 0.5
	var cos_angle: float = cos(simulation_4_obstacle_angle)
	var sin_angle: float = sin(simulation_4_obstacle_angle)
	var relative: Vector2 = position - simulation_4_obstacle_center
	var local_position: Vector2 = Vector2(
		relative.x * cos_angle + relative.y * sin_angle,
		-relative.x * sin_angle + relative.y * cos_angle
	)
	var projection: float = absf(cos_angle) + absf(sin_angle)
	var half_extents: Vector2 = simulation_4_obstacle_size * 0.5 + Vector2.ONE * (half_agent * projection)

	for sensor_index in 8:
		var direction: Vector2 = SIMULATION_6_SENSOR_DIRECTIONS[sensor_index]
		# Arena boundaries are also collision surfaces (as in the contact sensor).
		# Ray directions are unit vectors, so distances are in world pixels.
		var distance: float = INF
		if direction.x < -0.00000001:
			distance = minf(distance, (half_agent - position.x) / direction.x)
		elif direction.x > 0.00000001:
			distance = minf(distance, (simulation_size.x - half_agent - position.x) / direction.x)
		if direction.y < -0.00000001:
			distance = minf(distance, (half_agent - position.y) / direction.y)
		elif direction.y > 0.00000001:
			distance = minf(distance, (simulation_size.y - half_agent - position.y) / direction.y)

		var local_direction: Vector2 = Vector2(
			direction.x * cos_angle + direction.y * sin_angle,
			-direction.x * sin_angle + direction.y * cos_angle
		)
		distance = minf(distance, _simulation_6_ray_box_distance(local_position, local_direction, half_extents))
		agent.input_signals[5 + sensor_index] = 1.0 - clampf(maxf(distance, 0.0) / SIMULATION_6_SENSOR_RANGE, 0.0, 1.0)


func _simulation_6_ray_box_distance(origin: Vector2, direction: Vector2, half_extents: Vector2) -> float:
	# Slab intersection with the same (inflated) box used for agent collision.
	# Negative entry and nonnegative exit mean the ray starts inside/on the box.
	var entry: float = -INF
	var exit: float = INF
	if absf(direction.x) < 0.00000001:
		if absf(origin.x) > half_extents.x:
			return INF
	else:
		var tx1: float = (-half_extents.x - origin.x) / direction.x
		var tx2: float = (half_extents.x - origin.x) / direction.x
		entry = maxf(entry, minf(tx1, tx2))
		exit = minf(exit, maxf(tx1, tx2))
	if absf(direction.y) < 0.00000001:
		if absf(origin.y) > half_extents.y:
			return INF
	else:
		var ty1: float = (-half_extents.y - origin.y) / direction.y
		var ty2: float = (half_extents.y - origin.y) / direction.y
		entry = maxf(entry, minf(ty1, ty2))
		exit = minf(exit, maxf(ty1, ty2))
	if exit < 0.0 or entry > exit:
		return INF
	return maxf(entry, 0.0)


func _simulation_5_reset_agent(agent) -> void:
	agent.set_position(simulation_5_maze.start_position)
	agent.state["simulation_5_best_progress"] = 0.0
	agent.state["simulation_5_first_arrival"] = -1.0


func _simulation_5_update_agent_inputs(agent) -> void:
	# The first four channels retain exactly the original target displacement.
	_simulation_1_update_agent_inputs(agent)
	_simulation_5_write_wall_proximity(agent.position, agent)


func _simulation_5_write_wall_proximity(position: Vector2, agent) -> void:
	# The inflated wall rectangles are calculated once per maze generation.
	# Reuse the four existing signal slots instead of allocating a temporary
	# PackedFloat32Array for every agent and every simulation step.
	var half_agent: float = agent_square_size * 0.5
	var left_distance: float = maxf(position.x - half_agent, 0.0)
	var right_distance: float = maxf(simulation_size.x - half_agent - position.x, 0.0)
	var up_distance: float = maxf(position.y - half_agent, 0.0)
	var down_distance: float = maxf(simulation_size.y - half_agent - position.y, 0.0)

	for rect in simulation_5_maze.collision_walls:
		var low: Vector2 = rect.position
		var high: Vector2 = rect.end
		if position.y >= low.y and position.y <= high.y:
			if position.x < low.x:
				right_distance = minf(right_distance, low.x - position.x)
			elif position.x > high.x:
				left_distance = minf(left_distance, position.x - high.x)
			else:
				left_distance = 0.0
				right_distance = 0.0
		if position.x >= low.x and position.x <= high.x:
			if position.y < low.y:
				down_distance = minf(down_distance, low.y - position.y)
			elif position.y > high.y:
				up_distance = minf(up_distance, position.y - high.y)
			else:
				up_distance = 0.0
				down_distance = 0.0

	agent.input_signals[4] = 1.0 - clampf(left_distance / SIMULATION_5_SENSOR_RANGE, 0.0, 1.0)
	agent.input_signals[5] = 1.0 - clampf(right_distance / SIMULATION_5_SENSOR_RANGE, 0.0, 1.0)
	agent.input_signals[6] = 1.0 - clampf(up_distance / SIMULATION_5_SENSOR_RANGE, 0.0, 1.0)
	agent.input_signals[7] = 1.0 - clampf(down_distance / SIMULATION_5_SENSOR_RANGE, 0.0, 1.0)


func _simulation_4_touching_surface(position: Vector2) -> bool:
	var half_agent_size: float = agent_square_size * 0.5
	var tolerance: float = SIMULATION_4_TOUCH_TOLERANCE
	if (
		position.x <= half_agent_size + tolerance
		or position.y <= half_agent_size + tolerance
		or position.x >= simulation_size.x - half_agent_size - tolerance
		or position.y >= simulation_size.y - half_agent_size - tolerance
	):
		return true

	# Match the obstacle inflation used by the swept collision solver.
	var local_position: Vector2 = (position - simulation_4_obstacle_center).rotated(-simulation_4_obstacle_angle)
	var projection: float = absf(cos(simulation_4_obstacle_angle)) + absf(sin(simulation_4_obstacle_angle))
	var half_extents: Vector2 = simulation_4_obstacle_size * 0.5 + Vector2.ONE * (half_agent_size * projection)
	var dx: float = absf(local_position.x)
	var dy: float = absf(local_position.y)
	return (
		dx <= half_extents.x + tolerance
		and dy <= half_extents.y + tolerance
		and (dx >= half_extents.x - tolerance or dy >= half_extents.y - tolerance)
	)


func _simulation_1_apply_agent_outputs(agent, step_delta: float) -> void:
	var output_left: float = agent.output_signals[0]
	var output_right: float = agent.output_signals[1]
	var output_up: float = agent.output_signals[2]
	var output_down: float = agent.output_signals[3]
	var horizontal_drive = output_right - output_left
	var vertical_drive = output_down - output_up
	var half_agent_size = agent_square_size * 0.5
	var minimum_position = Vector2(half_agent_size, half_agent_size)
	var maximum_position = Vector2(
		simulation_size.x - half_agent_size,
		simulation_size.y - half_agent_size
	)
	var agent_position: Vector2 = agent.position

	agent_position += Vector2(horizontal_drive, vertical_drive) * agent_speed * step_delta
	agent_position.x = clampf(agent_position.x, minimum_position.x, maximum_position.x)
	agent_position.y = clampf(agent_position.y, minimum_position.y, maximum_position.y)
	agent.set_position(agent_position)


func _simulation_4_apply_agent_outputs(agent, step_delta: float) -> void:
	# Kinematic movement: no gravity, restitution, friction or agent-agent collisions.
	var displacement = Vector2(
		agent.output_signals[1] - agent.output_signals[0],
		agent.output_signals[3] - agent.output_signals[2]
	) * agent_speed * step_delta

	var previous_position: Vector2 = agent.position
	var agent_position: Vector2 = _simulation_4_move_with_collision(previous_position, displacement)
	var half_agent_size: float = agent_square_size * 0.5
	agent_position.x = clampf(agent_position.x, half_agent_size, simulation_size.x - half_agent_size)
	agent_position.y = clampf(agent_position.y, half_agent_size, simulation_size.y - half_agent_size)
	agent.set_position(agent_position)
	# Fitness collection normally runs before motor outputs. Also check after
	# movement to capture arrival on the very last tick of a target test.
	if not free_mode_active:
		_simulation_4_record_arrival(agent, experiment_elapsed_seconds + step_delta)


func _simulation_5_apply_agent_outputs(agent, step_delta: float) -> void:
	var movement: Vector2 = Vector2(
		agent.output_signals[1] - agent.output_signals[0],
		agent.output_signals[3] - agent.output_signals[2]
	) * agent_speed * step_delta
	var previous: Vector2 = agent.position
	var next_position: Vector2 = _simulation_5_move_with_collision(previous, movement)
	var half_agent: float = agent_square_size * 0.5
	next_position.x = clampf(next_position.x, half_agent, simulation_size.x - half_agent)
	next_position.y = clampf(next_position.y, half_agent, simulation_size.y - half_agent)
	agent.set_position(next_position)
	if not free_mode_active:
		# Preserve the post-movement arrival check. When there was no movement,
		# the pre-movement progress is still exact; avoid another distance query.
		if next_position == previous:
			_simulation_5_record_progress_value(agent, experiment_elapsed_seconds + step_delta, agent.pre_move_progress)
		else:
			_simulation_5_record_progress(agent, experiment_elapsed_seconds + step_delta)


func _simulation_5_move_with_collision(start: Vector2, movement: Vector2) -> Vector2:
	# Same continuous swept-AABB solver and four sliding iterations. The
	# broad-phase rejects only walls outside the swept center's AABB.
	var position: Vector2 = start
	var remaining: Vector2 = movement
	for _collision in 4:
		if remaining.length_squared() < 0.000000000001:
			break
		var earliest: float = INF
		var normal: Vector2 = Vector2.ZERO
		var destination: Vector2 = position + remaining
		var sweep_min_x: float = minf(position.x, destination.x)
		var sweep_max_x: float = maxf(position.x, destination.x)
		var sweep_min_y: float = minf(position.y, destination.y)
		var sweep_max_y: float = maxf(position.y, destination.y)
		for rect in simulation_5_maze.collision_walls:
			# Inclusive bounds preserve all contacts, including tangencies.
			# A small tolerance avoids culling floating-point boundary cases.
			var low: Vector2 = rect.position
			var high: Vector2 = rect.end
			if sweep_max_x < low.x - 0.000001 or sweep_min_x > high.x + 0.000001:
				continue
			if sweep_max_y < low.y - 0.000001 or sweep_min_y > high.y + 0.000001:
				continue
			var hit: Vector3 = _simulation_4_sweep_obstacle(
				position - rect.get_center(), remaining, rect.size * 0.5
			)
			if hit.x >= 0.0 and hit.x < earliest:
				earliest = hit.x
				normal = Vector2(hit.y, hit.z)
		if earliest == INF:
			position += remaining
			break
		position += remaining * earliest
		position += normal * SIMULATION_4_COLLISION_EPSILON
		remaining *= 1.0 - earliest
		var inward: float = remaining.dot(normal)
		if inward < 0.0:
			remaining -= normal * inward
	return position

func _simulation_4_move_with_collision(start: Vector2, movement: Vector2) -> Vector2:
	# Rotate into the obstacle's coordinates. Inflate by the world-aligned agent
	# square's projected half-width, so its corners cannot penetrate the wall.
	var local_position: Vector2 = (start - simulation_4_obstacle_center).rotated(-simulation_4_obstacle_angle)
	var remaining: Vector2 = movement.rotated(-simulation_4_obstacle_angle)
	var projection: float = absf(cos(simulation_4_obstacle_angle)) + absf(sin(simulation_4_obstacle_angle))
	var agent_margin: float = 0.5 * agent_square_size * projection
	var half_extents: Vector2 = simulation_4_obstacle_size * 0.5 + Vector2.ONE * agent_margin

	# Free Mode can move the target (and thus the obstacle) over an existing agent.
	# Project such an agent outside before performing the swept collision test.
	if absf(local_position.x) < half_extents.x and absf(local_position.y) < half_extents.y:
		var clearance_x: float = half_extents.x - absf(local_position.x)
		var clearance_y: float = half_extents.y - absf(local_position.y)
		if clearance_x < clearance_y:
			local_position.x = (1.0 if local_position.x >= 0.0 else -1.0) * (half_extents.x + SIMULATION_4_COLLISION_EPSILON)
		else:
			local_position.y = (1.0 if local_position.y >= 0.0 else -1.0) * (half_extents.y + SIMULATION_4_COLLISION_EPSILON)

	# Sweep the center against the inflated rectangle (prevents tunnelling).
	# At impact, remove only the inward normal component: the tangential
	# component is preserved, giving frictionless sliding and no bounce.
	for _collision_index in 2:
		if remaining.length_squared() <= 0.000000000001:
			break
		var hit: Vector3 = _simulation_4_sweep_obstacle(local_position, remaining, half_extents)
		if hit.x < 0.0:
			local_position += remaining
			break
		local_position += remaining * hit.x
		var normal: Vector2 = Vector2(hit.y, hit.z)
		local_position += normal * SIMULATION_4_COLLISION_EPSILON
		remaining *= 1.0 - hit.x
		var inward_speed: float = remaining.dot(normal)
		if inward_speed < 0.0:
			remaining -= normal * inward_speed

	return simulation_4_obstacle_center + local_position.rotated(simulation_4_obstacle_angle)


func _simulation_4_sweep_obstacle(position: Vector2, movement: Vector2, half_extents: Vector2) -> Vector3:
	# Slab raycast in the obstacle's local frame. x = -1 means no collision.
	var entry_time: float = -INF
	var exit_time: float = INF
	var hit_normal: Vector2 = Vector2.ZERO

	if absf(movement.x) < 0.00000001:
		if absf(position.x) >= half_extents.x:
			return Vector3(-1.0, 0.0, 0.0)
	else:
		var x_time_1: float = (-half_extents.x - position.x) / movement.x
		var x_time_2: float = (half_extents.x - position.x) / movement.x
		var x_entry: float = minf(x_time_1, x_time_2)
		entry_time = x_entry
		exit_time = maxf(x_time_1, x_time_2)
		hit_normal = Vector2(-1.0, 0.0) if movement.x > 0.0 else Vector2(1.0, 0.0)

	if absf(movement.y) < 0.00000001:
		if absf(position.y) >= half_extents.y:
			return Vector3(-1.0, 0.0, 0.0)
	else:
		var y_time_1: float = (-half_extents.y - position.y) / movement.y
		var y_time_2: float = (half_extents.y - position.y) / movement.y
		var y_entry: float = minf(y_time_1, y_time_2)
		if y_entry > entry_time:
			entry_time = y_entry
			hit_normal = Vector2(0.0, -1.0) if movement.y > 0.0 else Vector2(0.0, 1.0)
		exit_time = minf(exit_time, maxf(y_time_1, y_time_2))

	if entry_time < 0.0 or entry_time > 1.0 or entry_time > exit_time or exit_time < 0.0:
		return Vector3(-1.0, 0.0, 0.0)
	return Vector3(entry_time, hit_normal.x, hit_normal.y)


func _simulation_1_reset_target(selected_target) -> void:
	_set_random_target_position(selected_target)


func _simulation_1_update_target(_selected_target, _step_delta: float) -> void:
	pass


func _simulation_2_reset_target(selected_target) -> void:
	_set_random_target_position(selected_target)
	selected_target.state["position_change_elapsed"] = 0.0


func _simulation_2_update_target(selected_target, step_delta: float) -> void:
	selected_target.state["position_change_elapsed"] += step_delta

	while selected_target.state["position_change_elapsed"] >= target_position_change_interval_seconds:
		selected_target.state["position_change_elapsed"] -= target_position_change_interval_seconds
		_set_random_target_position(selected_target)


func _simulation_3_reset_target(selected_target) -> void:
	_set_random_target_position(selected_target)
	_simulation_3_choose_target_waypoint(selected_target)


func _simulation_3_update_target(selected_target, step_delta: float) -> void:
	var remaining_distance = simulation_3_target_speed * step_delta

	while remaining_distance > 0.0:
		var waypoint: Vector2 = selected_target.state["waypoint"]
		var displacement: Vector2 = waypoint - selected_target.position
		var distance_to_waypoint = displacement.length()

		if distance_to_waypoint <= remaining_distance:
			selected_target.position = waypoint
			remaining_distance -= distance_to_waypoint
			_simulation_3_choose_target_waypoint(selected_target)
		else:
			selected_target.position += displacement / distance_to_waypoint * remaining_distance
			remaining_distance = 0.0


func _simulation_3_choose_target_waypoint(selected_target) -> void:
	var half_target_size = target_square_size * 0.5
	var min_x = half_target_size
	var max_x = simulation_size.x - half_target_size
	var min_y = half_target_size
	var max_y = simulation_size.y - half_target_size
	var waypoint = selected_target.position

	while waypoint.distance_squared_to(selected_target.position) < 1.0:
		waypoint = Vector2(
			randf_range(min_x, max_x),
			randf_range(min_y, max_y)
		)

	selected_target.state["waypoint"] = waypoint


func _update_simulation_4_num_tests_label() -> void:
	if not is_instance_valid(simulation_4_num_tests_label):
		return
	simulation_4_num_tests_label.visible = simulation_selector.selected == 3 or simulation_selector.selected == 4 or simulation_selector.selected == 5
	if simulation_selector.selected == 4:
		simulation_4_num_tests_label.text = "Difficulty: %d / %d" % [simulation_5_difficulty, SIMULATION_5_MAX_DIFFICULTY]
	elif simulation_selector.selected == 5:
		simulation_4_num_tests_label.text = "Num Tests: %d" % simulation_6_difficulty
	else:
		simulation_4_num_tests_label.text = "Num Tests: %d" % simulation_4_difficulty


func _simulation_5_advance_difficulty_if_ready() -> void:
	# Mean over ALL agents and ALL tests, never a single elite or one layout.
	# Changes apply to the next generation only.
	if simulation_selector.selected != 4 or simulation_5_difficulty >= SIMULATION_5_MAX_DIFFICULTY:
		return
	if float(_generation_fitness_statistics()["mean"]) <= SIMULATION_5_ADVANCE_THRESHOLD:
		return
	simulation_5_difficulty += 1
	_update_simulation_4_num_tests_label()


func _simulation_5_reset_target(selected_target) -> void:
	# Generated once per target test, BEFORE resetting all agents.
	simulation_5_maze.generate(
		simulation_5_difficulty, simulation_size,
		SIMULATION_5_WALL_THICKNESS, agent_square_size
	)
	selected_target.position = simulation_5_maze.target_position

func _simulation_4_advance_difficulty_if_ready() -> void:
	# The generation just finished; its average includes every agent and
	# every target test of the CURRENT difficulty. The next difficulty is
	# applied only to the next generation, never in the middle of a test.
	if simulation_selector.selected != 3 and simulation_selector.selected != 5:
		return
	var current_difficulty: int = _simulation_4_active_difficulty()
	if current_difficulty >= SIMULATION_4_ANGULAR_SECTOR_COUNT:
		return
	var required_mean: float = SIMULATION_4_DIFFICULTY_BASE_THRESHOLD + SIMULATION_4_DIFFICULTY_THRESHOLD_STEP * float(current_difficulty - 3)
	if float(_generation_fitness_statistics()["mean"]) <= required_mean:
		return
	if simulation_selector.selected == 5:
		simulation_6_difficulty += 1
	else:
		simulation_4_difficulty += 1
	target_test_count = _simulation_4_active_difficulty()
	_update_simulation_4_num_tests_label()


func _simulation_4_active_difficulty() -> int:
	return simulation_6_difficulty if simulation_selector.selected == 5 else simulation_4_difficulty


func _simulation_4_shuffle_target_sectors() -> void:
	# At difficulty N, visit each of N angular sectors exactly once per
	# generation. N=3 is the initial curriculum stage; N=12 is the final stage.
	simulation_4_target_sector_order.clear()
	for sector_index in _simulation_4_active_difficulty():
		simulation_4_target_sector_order.append(sector_index)
	simulation_4_target_sector_order.shuffle()


func _simulation_4_reset_target(selected_target) -> void:
	# Called once before the initial population exists, too.
	if simulation_4_target_sector_order.size() != _simulation_4_active_difficulty():
		_simulation_4_shuffle_target_sectors()

	# One randomly placed target/obstacle in each sector for this generation.
	# Radial area-uniform sampling keeps the target inside the arena,
	# outside the configured minimum-distance constraint when feasible.
	var half_target_size: float = target_square_size * 0.5
	var minimum: Vector2 = Vector2.ONE * half_target_size
	var maximum: Vector2 = simulation_size - minimum
	var center: Vector2 = simulation_size * 0.5

	var sector_index: int = simulation_4_target_sector_order[target_test_index]
	var sector_width: float = TAU / float(_simulation_4_active_difficulty())
	var angle: float = (float(sector_index) + randf()) * sector_width
	var direction: Vector2 = Vector2(cos(angle), sin(angle))

	var maximum_radius: float = INF
	if direction.x > 0.0:
		maximum_radius = minf(maximum_radius, (maximum.x - center.x) / direction.x)
	elif direction.x < 0.0:
		maximum_radius = minf(maximum_radius, (minimum.x - center.x) / direction.x)
	if direction.y > 0.0:
		maximum_radius = minf(maximum_radius, (maximum.y - center.y) / direction.y)
	elif direction.y < 0.0:
		maximum_radius = minf(maximum_radius, (minimum.y - center.y) / direction.y)

	# The default dimensions always allow the configured minimum distance.
	# For degenerate dimensions, fall back to the farthest feasible point.
	var minimum_radius: float = minf(simulation_4_min_target_distance, maximum_radius)
	var squared_radius: float = lerpf(
		minimum_radius * minimum_radius,
		maximum_radius * maximum_radius,
		randf()
	)
	selected_target.position = center + direction * sqrt(squared_radius)
	_simulation_4_update_obstacle(true)


func _simulation_4_update_obstacle(randomize_dimensions: bool) -> void:
	var center: Vector2 = simulation_size * 0.5
	var radial: Vector2 = target.position - center
	var direction: Vector2 = radial.normalized() if radial.length_squared() > 0.000001 else Vector2.RIGHT
	var tangent: Vector2 = Vector2(-direction.y, direction.x)
	# The center-to-target line crosses the rectangle at its midpoint,
	# perpendicular to its long side.
	simulation_4_obstacle_center = center.lerp(target.position, 0.5)
	simulation_4_obstacle_angle = tangent.angle()

	if randomize_dimensions or simulation_4_obstacle_size == Vector2.ZERO:
		var variation: float = simulation_4_obstacle_size_variation
		simulation_4_obstacle_size = Vector2(
			simulation_4_obstacle_length * randf_range(1.0 - variation, 1.0 + variation),
			simulation_4_obstacle_thickness * randf_range(1.0 - variation, 1.0 + variation)
		)


func _draw_simulation_4_obstacle(origin: Vector2) -> void:
	var half_size: Vector2 = simulation_4_obstacle_size * 0.5
	var local_corners = PackedVector2Array([
		Vector2(-half_size.x, -half_size.y),
		Vector2(half_size.x, -half_size.y),
		Vector2(half_size.x, half_size.y),
		Vector2(-half_size.x, half_size.y),
	])
	var corners = PackedVector2Array()
	for corner in local_corners:
		corners.append(origin + simulation_4_obstacle_center + corner.rotated(simulation_4_obstacle_angle))

	draw_colored_polygon(corners, simulation_4_obstacle_color)
	corners.append(corners[0])
	draw_polyline(corners, simulation_4_obstacle_border_color, 2.0, true)


func _set_random_target_position(selected_target) -> void:
	var half_target_size = target_square_size * 0.5
	var min_x = half_target_size
	var max_x = simulation_size.x - half_target_size
	var min_y = half_target_size
	var max_y = simulation_size.y - half_target_size

	selected_target.position = Vector2(
		randf_range(min_x, max_x),
		randf_range(min_y, max_y)
	)


func _simulation_1_evaluate_agent(agent) -> Vector2:
	var agent_position: Vector2 = agent.position
	var normalized_delta = Vector2(
		(target.position.x - agent_position.x) / simulation_size.x,
		(target.position.y - agent_position.y) / simulation_size.y
	)
	var score_x = exp(-normalized_delta.x * normalized_delta.x)
	var score_y = exp(-normalized_delta.y * normalized_delta.y)
	var score = score_x * score_y
	var distance = normalized_delta.length()
	return Vector2(score, distance)


func _simulation_1_reset_fitness() -> void:
	fitness_integrals = PackedFloat64Array()
	fitness_integrals.resize(agent_count)


func _simulation_1_accumulate_fitness(step_delta: float) -> void:
	for agent_index in agent_count:
		var evaluation: Vector2 = _simulation_1_evaluate_agent(agents[agent_index])
		fitness_integrals[agent_index] += evaluation.x * step_delta


func _simulation_1_fitness(agent_index: int) -> float:
	if experiment_elapsed_seconds <= 0.0:
		return 0.0
	return fitness_integrals[agent_index] / experiment_elapsed_seconds


func _simulation_4_record_arrival(agent, elapsed_time: float) -> void:
	var position: Vector2 = agent.position
	var displacement: Vector2 = target.position - position
	# Both agents and target are rendered as axis-aligned squares. A goal is
	# reached when these squares first touch or overlap.
	var contact_half_size: float = (agent_square_size + target_square_size) * 0.5
	if float(agent.state["simulation_4_first_arrival"]) < 0.0 and absf(displacement.x) <= contact_half_size and absf(displacement.y) <= contact_half_size:
		agent.state["simulation_4_first_arrival"] = elapsed_time


# Simulation 4: score is 1 inside the target's 12-pixel radius (by default),
# then decays exponentially with excess center-to-center distance.
func _simulation_4_proximity_score(distance: float) -> float:
	var radius: float = maxf(target_square_size, 0.000001)
	return exp(-maxf(distance - radius, 0.0) / 100.0)


# Only the last second of each target test contributes to fitness.
# Clamp the integrated portion of each neural step to the final-time window,
# so the result does not depend on a tick landing exactly at its beginning.
func _simulation_4_accumulate_fitness(step_delta: float) -> void:
	# Keep the obstacle/arrival diagnostics correct throughout the test, even
	# though fitness deliberately uses only its final second.
	for agent in agents:
		_simulation_4_record_arrival(agent, experiment_elapsed_seconds)
	var window_start: float = maxf(experiment_duration_seconds - 1.0, 0.0)
	var window_delta: float = minf(experiment_elapsed_seconds + step_delta, experiment_duration_seconds) - maxf(experiment_elapsed_seconds, window_start)
	if window_delta <= 0.0:
		return

	for agent_index in agent_count:
		var position: Vector2 = agents[agent_index].position
		fitness_integrals[agent_index] += _simulation_4_proximity_score(position.distance_to(target.position)) * window_delta


func _simulation_4_fitness(agent_index: int) -> float:
	var window_length: float = minf(experiment_duration_seconds, 1.0)
	if window_length <= 0.0:
		return 0.0
	return clampf(fitness_integrals[agent_index] / window_length, 0.0, 1.0)


func _simulation_5_position_progress(position: Vector2) -> float:
	var delta: Vector2 = position - target.position
	var touching_distance: float = 0.5 * (agent_square_size + target_square_size)
	if absf(delta.x) <= touching_distance and absf(delta.y) <= touching_distance:
		return 1.0
	return simulation_5_maze.progress(position)


func _simulation_5_record_progress(agent, elapsed_time: float) -> void:
	# A post-movement distance query is cached for the next iteration, where
	# the agent begins at this same position and the maze target is unchanged.
	_simulation_5_record_progress_value(agent, elapsed_time, _simulation_5_evaluate_agent(agent).x)


func _simulation_5_record_progress_value(agent, elapsed_time: float, progress: float) -> void:
	# Reuse the evaluation already computed at the same position before movement.
	agent.state["simulation_5_best_progress"] = maxf(
		float(agent.state.get("simulation_5_best_progress", 0.0)), progress
	)
	if progress >= 1.0 and float(agent.state.get("simulation_5_first_arrival", -1.0)) < 0.0:
		agent.state["simulation_5_first_arrival"] = elapsed_time


func _simulation_5_accumulate_fitness(_step_delta: float) -> void:
	# Spatial progress only: no duration, speed, elapsed-time integration,
	# early-arrival bonus or time penalty anywhere in the fitness equation.
	for agent in agents:
		_simulation_5_record_progress(agent, experiment_elapsed_seconds)


func _simulation_5_fitness(agent_index: int) -> float:
	var agent = agents[agent_index]
	return maxf(
		float(agent.state.get("simulation_5_best_progress", 0.0)),
		_simulation_5_position_progress(agent.position)
	)


func _simulation_5_evaluate_agent(agent) -> Vector2:
	var position: Vector2 = agent.position
	# Both diagnostics use one query. Subsequent steps reuse the distance
	# calculated just after the previous movement at this exact position.
	var distance: float
	if agent.maze_distance_valid:
		distance = agent.maze_cached_distance
	else:
		distance = simulation_5_maze.distance_to_target(position)
		agent.maze_cached_distance = distance
		agent.maze_distance_valid = true
	var displacement: Vector2 = position - target.position
	var touching_distance: float = 0.5 * (agent_square_size + target_square_size)
	var progress: float
	if absf(displacement.x) <= touching_distance and absf(displacement.y) <= touching_distance:
		progress = 1.0
	elif is_inf(distance):
		progress = 0.0
	else:
		progress = clampf(1.0 - distance / simulation_5_maze.initial_distance, 0.0, 1.0)
	return Vector2(progress, distance / simulation_5_maze.initial_distance)


func _simulation_5_agent_info(agent_index: int) -> Dictionary:
	var info: Dictionary = _simulation_1_agent_info(agent_index)
	if info.is_empty():
		return info
	# Read the inputs actually consumed by the GPU, just as Simulation 1 does.
	# The external info column contains inputs and outputs ONLY.
	var stats: Dictionary = gpu_backend.selected_statistics(agent_index)
	if stats.is_empty():
		return info
	var inputs: Variant = stats["input_signals_used"]
	if inputs.size() < 8:
		return info
	var extra_inputs: String = (
		"\nInput 5 (wall left): %.2f"
		+ "\nInput 6 (wall right): %.2f"
		+ "\nInput 7 (wall up): %.2f"
		+ "\nInput 8 (wall down): %.2f"
	) % [float(inputs[4]), float(inputs[5]), float(inputs[6]), float(inputs[7])]
	info["main"] = str(info["main"]).replace(
		"\nOutput 1 (move left):",
		extra_inputs + "\nOutput 1 (move left):"
	)
	return info


func _simulation_4_evaluate_agent(agent) -> Vector2:
	var position: Vector2 = agent.position
	var distance: float = position.distance_to(target.position)
	var normalized_distance: float = distance / simulation_size.length()
	# The score component is instantaneous for diagnostics; accumulated fitness
	# still uses only the last second of the test.
	return Vector2(_simulation_4_proximity_score(distance), normalized_distance)


func _simulation_4_agent_info(agent_index: int) -> Dictionary:
	var info: Dictionary = _simulation_1_agent_info(agent_index)
	if info.is_empty():
		return info
	var stats: Dictionary = gpu_backend.selected_statistics(agent_index)
	if stats.is_empty():
		return info
	var inputs: Variant = stats["input_signals_used"]
	info["main"] = str(info["main"]).replace(
		"\nOutput 1 (move left):",
		"\nInput 5 (contact): %.2f\nOutput 1 (move left):" % float(inputs[4])
	)
	return info


func _simulation_6_agent_info(agent_index: int) -> Dictionary:
	var info: Dictionary = _simulation_4_agent_info(agent_index)
	if info.is_empty():
		return info
	var stats: Dictionary = gpu_backend.selected_statistics(agent_index)
	if stats.is_empty():
		return info
	var inputs: Variant = stats["input_signals_used"]
	if inputs.size() < 13:
		return info
	var directions: Array[String] = ["W", "E", "N", "S", "NW", "NE", "SW", "SE"]
	var extra: String = ""
	for sensor_index in 8:
		extra += "\nInput %d (wall %s): %.2f" % [6 + sensor_index, directions[sensor_index], float(inputs[5 + sensor_index])]
	info["main"] = str(info["main"]).replace("\nOutput 1 (move left):", extra + "\nOutput 1 (move left):")
	return info


func _distribution(values: Variant) -> Dictionary:
	var sorted_values: Array = []
	for value in values:
		sorted_values.append(float(value))
	if sorted_values.is_empty():
		return {"min":0.0,"max":0.0,"mean":0.0,"std":0.0,"25%":0.0,"50%":0.0,"75%":0.0,"count":0}
	sorted_values.sort()
	var n: int = sorted_values.size()
	var avg := 0.0
	for value in sorted_values:
		avg += value
	avg /= float(n)
	var variance := 0.0
	for value in sorted_values:
		variance += (value-avg)*(value-avg)
	return {"min":sorted_values[0],"max":sorted_values[n-1],"mean":avg,"std":sqrt(variance/float(n)),"25%":sorted_values[int(floor((n-1)*0.25))],"50%":sorted_values[int(floor((n-1)*0.5))],"75%":sorted_values[int(floor((n-1)*0.75))],"count":n}


func _simulation_1_agent_info(agent_index: int) -> Dictionary:
	if gpu_backend == null:
		return {}
	var state: Dictionary = gpu_backend.selected_statistics(agent_index)
	if state.is_empty():
		return {}
	var state_stats: Dictionary = _distribution(state["neuron_states_used"])
	var signal_stats: Dictionary = _distribution(state["normalized_signals"])
	var brain: Dictionary = population_brains[agent_index]
	var input_ids: PackedInt32Array = state["input_neuron_indices"]
	var input_values: Variant = state["input_signals_used"]
	var outputs: Variant = state["output_signals"]
	var input_display := []
	for i in 4:
		input_display.append(float(input_values[i]))
	var output_display := []
	for i in 4:
		output_display.append(float(outputs[i]))
	var main_text := ("AGENT %d\nInput 1 (left): %.2f\nInput 2 (right): %.2f\nInput 3 (up): %.2f\nInput 4 (down): %.2f\nOutput 1 (move left): %.2f\nOutput 2 (move right): %.2f\nOutput 3 (move up): %.2f\nOutput 4 (move down): %.2f") % ([agent_index+1] + input_display + output_display)
	var state_text := ("\nMin state: %.2f\nMax state: %.2f\nMean state: %.2f\nStd state: %.2f\n25%% state: %.2f\nMedian state: %.2f\n75%% state: %.2f\nNeuron count: %d\nInput influence: %.2f") % [state_stats["min"],state_stats["max"],state_stats["mean"],state_stats["std"],state_stats["25%"],state_stats["50%"],state_stats["75%"],state_stats["count"],float(brain.get("input_gain",0.0))]
	var signal_text := ("\nMin signal: %.2f\nMax signal: %.2f\nMean signal: %.2f\nStd signal: %.2f\n25%% signal: %.2f\nMedian signal: %.2f\n75%% signal: %.2f") % [signal_stats["min"],signal_stats["max"],signal_stats["mean"],signal_stats["std"],signal_stats["25%"],signal_stats["50%"],signal_stats["75%"]]
	return {"main":main_text,"state":state_text,"signal":signal_text}


func _release_gpu_population() -> void:
	_invalidate_brain_preview()
	if gpu_backend != null:
		gpu_backend.clear()
		gpu_backend = null
	else:
		# Unconsumed retained phenotypes (for example, a loaded population).
		var gpu_rd: RenderingDevice
		var builder := BrainBuilderGPU.new()
		for brain in population_brains:
			if brain is Dictionary and brain.has("gpu_state"):
				gpu_rd = brain["gpu_state"]["rd"]
				builder.release_gpu_state(brain)
		if gpu_rd != null:
			gpu_rd.free()


func _clear_runtime_state() -> void:
	_release_gpu_population()
	agent_dynamics.clear()
	fitness_reset_behavior.call()

	for agent in agents:
		agent.clear()
	if target != null:
		target.clear()
	selected_agent_index = -1
	agent_info_label.text = "Click an agent to inspect its neural signals."
	agent_info_label_2.text = ""
	agent_info_label_3.text = ""
	experiment_elapsed_seconds = 0.0
	summary_elapsed_seconds = 0.0
	best_score_ever = 0.0
	best_distance_ever = 1.0e30
	current_mean_score = 0.0
	current_best_score = 0.0
	current_best_score_agent = -1
	current_mean_distance = 0.0
	current_min_distance = 0.0
	current_closest_agent = -1
	experiment_state = ExperimentState.IDLE
	free_mode_active = false
	target_dragging = false
	free_mode_button.text = "Free Mode"
	pause_button.text = "Pause"
	queue_redraw()


# Fitness is evaluated at the same positions as the preceding sensor phase.
# Avoid re-evaluating the expensive distance field / proximity score in a
# second pass. Preserve the original accumulation and arrival timing.
func _accumulate_cached_step_fitness(step_delta: float) -> void:
	match active_simulation_index:
		0, 1, 2:
			for agent_index in agent_count:
				fitness_integrals[agent_index] += _cached_evaluation_scores[agent_index] * step_delta
		3, 5:
			for agent in agents:
				_simulation_4_record_arrival(agent, experiment_elapsed_seconds)
			var window_start: float = maxf(experiment_duration_seconds - 1.0, 0.0)
			var window_delta: float = minf(experiment_elapsed_seconds + step_delta, experiment_duration_seconds) - maxf(experiment_elapsed_seconds, window_start)
			if window_delta > 0.0:
				for agent_index in agent_count:
					fitness_integrals[agent_index] += _cached_evaluation_scores[agent_index] * window_delta
		4:
			for agent_index in agent_count:
				_simulation_5_record_progress_value(agents[agent_index], experiment_elapsed_seconds, _cached_evaluation_scores[agent_index])
		_:
			fitness_accumulation_behavior.call(step_delta)


func _advance_gpu_step(step_delta: float, accumulate_fitness: bool) -> bool:
	if gpu_backend == null:
		return false
	# Sensor measurements are defined by the scene, never by the neural core.
	# Preserve the C++ order: sense and evaluate the environment, advance one
	# neural step, read outputs, then apply the agent actions to the environment.
	if _packed_gpu_inputs.size() != gpu_backend.input_count:
		_packed_gpu_inputs.resize(gpu_backend.input_count)
	if accumulate_fitness and _cached_evaluation_scores.size() != agent_count:
		_cached_evaluation_scores.resize(agent_count)
	var is_maze: bool = active_simulation_index == 4
	for agent_index in agent_count:
		var sim_agent = agents[agent_index]
		sim_agent.update_inputs()
		var input_offset: int = gpu_backend.input_offsets[agent_index]
		for channel in sim_agent.input_signals.size():
			_packed_gpu_inputs[input_offset + channel] = float(sim_agent.input_signals[channel])
		if accumulate_fitness:
			var evaluation: Vector2 = evaluation_behavior.call(sim_agent)
			_cached_evaluation_scores[agent_index] = evaluation.x
			if is_maze:
				sim_agent.pre_move_progress = evaluation.x
			best_score_ever = maxf(best_score_ever, evaluation.x)
			best_distance_ever = minf(best_distance_ever, evaluation.y)
	if accumulate_fitness:
		_accumulate_cached_step_fitness(step_delta)
	if not gpu_backend.advance(step_delta, _packed_gpu_inputs):
		push_error("BrainPopulationGPU runtime: " + gpu_backend.last_error)
		experiment_state = ExperimentState.PAUSED
		pause_button.text = "Continue"
		summary_label.text = "GPU neural core stopped: " + gpu_backend.last_error
		_refresh_buttons()
		return false
	var packed_outputs: PackedFloat32Array = gpu_backend.output_signals
	for agent_index in agent_count:
		var sim_agent = agents[agent_index]
		var output_offset: int = gpu_backend.output_offsets[agent_index]
		for channel in sim_agent.output_signals.size():
			sim_agent.output_signals[channel] = float(packed_outputs[output_offset + channel])
		sim_agent.apply_outputs(step_delta)
	# Render every executed _process step; no intermediate neural step is skipped.
	queue_redraw()
	return true

func _recalculate_current_spatial_metrics() -> void:
	if agents.is_empty():
		current_mean_score = 0.0
		current_best_score = 0.0
		current_best_score_agent = -1
		current_mean_distance = 0.0
		current_min_distance = 0.0
		current_closest_agent = -1
		return

	var score_sum = 0.0
	var distance_sum = 0.0
	var best_score = -1.0
	var minimum_distance = 1.0e30
	var best_score_agent = -1
	var closest_agent = -1

	for agent_index in agents.size():
		var evaluation: Vector2 = evaluation_behavior.call(agents[agent_index])
		var score: float = evaluation.x
		var distance: float = evaluation.y

		score_sum += score
		distance_sum += distance

		if score > best_score:
			best_score = score
			best_score_agent = agent_index

		if distance < minimum_distance:
			minimum_distance = distance
			closest_agent = agent_index

	current_mean_score = score_sum / agents.size()
	current_best_score = best_score
	current_best_score_agent = best_score_agent
	current_mean_distance = distance_sum / agents.size()
	current_min_distance = minimum_distance
	current_closest_agent = closest_agent

	best_score_ever = maxf(best_score_ever, current_best_score)
	best_distance_ever = minf(best_distance_ever, current_min_distance)


func _has_higher_generation_fitness(agent_a: int, agent_b: int) -> bool:
	var fitness_a: float = generation_fitness_sums[agent_a]
	var fitness_b: float = generation_fitness_sums[agent_b]
	if fitness_a == fitness_b:
		return agent_a < agent_b
	return fitness_a > fitness_b


func _get_top_three_agent_indices() -> Array[int]:
	var top_indices: Array[int] = [-1, -1, -1]
	var top_fitnesses: Array[float] = [-1.0, -1.0, -1.0]

	for agent_index in agent_count:
		var fitness: float = generation_fitness_sums[agent_index] / target_test_count

		for rank in 3:
			if (
				fitness > top_fitnesses[rank]
				or (fitness == top_fitnesses[rank] and (top_indices[rank] < 0 or agent_index < top_indices[rank]))
			):
				for shift_rank in range(2, rank, -1):
					top_fitnesses[shift_rank] = top_fitnesses[shift_rank - 1]
					top_indices[shift_rank] = top_indices[shift_rank - 1]
				top_fitnesses[rank] = fitness
				top_indices[rank] = agent_index
				break

	return top_indices


func _current_fitness_statistics() -> Dictionary:
	var total_fitness: float = 0.0
	var best_fitness: float = -1.0
	var best_fitness_agent: int = 0

	for agent_index in agent_count:
		var fitness: float = float(fitness_behavior.call(agent_index))
		total_fitness += fitness

		if fitness > best_fitness:
			best_fitness = fitness
			best_fitness_agent = agent_index

	return {
		"mean": total_fitness / agent_count,
		"best": best_fitness,
		"best_agent": best_fitness_agent,
	}


func _generation_fitness_statistics() -> Dictionary:
	var total_fitness: float = 0.0
	var best_fitness: float = -1.0
	var best_fitness_agent: int = 0

	for agent_index in agent_count:
		var fitness: float = generation_fitness_sums[agent_index] / target_test_count
		total_fitness += fitness

		if fitness > best_fitness:
			best_fitness = fitness
			best_fitness_agent = agent_index

	return {
		"mean": total_fitness / agent_count,
		"best": best_fitness,
		"best_agent": best_fitness_agent,
	}


func _record_current_target_test() -> void:
	for agent_index in agent_count:
		var fitness: float = float(fitness_behavior.call(agent_index))
		generation_fitness_sums[agent_index] += fitness

	generation_mean_final_score_sum += current_mean_score
	generation_best_final_score = maxf(generation_best_final_score, current_best_score)


func _record_completed_generation() -> void:
	var fitness_statistics: Dictionary = _generation_fitness_statistics()
	var mean_fitness: float = float(fitness_statistics["mean"])
	var best_fitness: float = float(fitness_statistics["best"])
	var generation_mean_final_score: float = generation_mean_final_score_sum / target_test_count

	completed_generation_count += 1
	evolution_mean_score_sum += generation_mean_final_score
	evolution_mean_fitness_sum += mean_fitness
	evolution_mean_fitness_history.append(mean_fitness)
	evolution_best_fitness_history.append(best_fitness)
	genetic_diversity_history.append(_population_genetic_diversity())
	elite_genetic_diversity_history.append(_elite_genetic_diversity())
	if heatmap_tracking_enabled:
		_record_phenotypic_fitness_generation()
	_rebuild_history_graph_caches()
	_queue_history_graph_redraw()
	queue_redraw()

	if generation_mean_final_score > evolution_best_mean_score:
		evolution_best_mean_score = generation_mean_final_score
		evolution_best_mean_score_generation = generation_number

	if generation_best_final_score > evolution_best_agent_score:
		evolution_best_agent_score = generation_best_final_score
		evolution_best_agent_score_generation = generation_number

	if mean_fitness > evolution_best_mean_fitness:
		evolution_best_mean_fitness = mean_fitness
		evolution_best_mean_fitness_generation = generation_number

	if best_fitness > evolution_best_agent_fitness:
		evolution_best_agent_fitness = best_fitness
		evolution_best_agent_fitness_generation = generation_number


func _reset_evolution_statistics() -> void:
	previous_generation_elite_indices.clear()
	generation_number = 0
	completed_generation_count = 0
	evolution_mean_score_sum = 0.0
	evolution_mean_fitness_sum = 0.0
	evolution_best_mean_score = -1.0
	evolution_best_mean_score_generation = 0
	evolution_best_agent_score = -1.0
	evolution_best_agent_score_generation = 0
	evolution_best_mean_fitness = -1.0
	evolution_best_mean_fitness_generation = 0
	evolution_best_agent_fitness = -1.0
	evolution_best_agent_fitness_generation = 0
	evolution_mean_fitness_history = PackedFloat64Array()
	evolution_best_fitness_history = PackedFloat64Array()
	genetic_diversity_history = PackedFloat64Array()
	elite_genetic_diversity_history = PackedFloat64Array()
	evolution_graph_mean_points_cache = PackedVector2Array()
	evolution_graph_best_points_cache = PackedVector2Array()
	genetic_diversity_graph_points_cache = PackedVector2Array()
	elite_genetic_diversity_graph_points_cache = PackedVector2Array()
	genetic_diversity_excluded_agent_indices.clear()
	current_elite_turnover_count = 0
	previous_generation_elite_signatures.clear()
	evolution_graph_minimum_value_cache = 0.0
	evolution_graph_maximum_value_cache = 1.0
	genetic_diversity_minimum_exponent_cache = -1
	_queue_history_graph_redraw()
	target_test_index = 0
	simulation_4_difficulty = 3
	simulation_5_difficulty = 1
	simulation_6_difficulty = 3
	if simulation_selector.selected == 3:
		target_test_count = simulation_4_difficulty
	elif simulation_selector.selected == 4:
		target_test_count = SIMULATION_5_NUM_TESTS
	elif simulation_selector.selected == 5:
		target_test_count = simulation_6_difficulty
	simulation_4_target_sector_order.clear()
	_update_simulation_4_num_tests_label()
	generation_fitness_sums = PackedFloat64Array()
	generation_mean_final_score_sum = 0.0
	generation_best_final_score = -1.0
	generation_real_start_us = 0
	generation_real_time_seconds = 0.0
	last_generation_best_brain_data.clear()
	_reset_phenotypic_fitness_history()
	queue_redraw()


func _finish_experiment() -> void:
	experiment_elapsed_seconds = experiment_duration_seconds
	pause_button.text = "Pause"
	_recalculate_current_spatial_metrics()
	_record_current_target_test()

	if target_test_index < target_test_count - 1:
		target_test_index += 1
		_start_experiment_from_population(false)
		return

	if generation_real_start_us > 0:
		generation_real_time_seconds = float(Time.get_ticks_usec() - generation_real_start_us) / 1000000.0
		generation_real_start_us = 0

	_record_completed_generation()
	_update_summary()

	var top_indices: Array[int] = _get_top_three_agent_indices()
	if top_indices.size() != 3 or top_indices[0] < 0 or top_indices[1] < 0 or top_indices[2] < 0:
		experiment_state = ExperimentState.FINISHED
		summary_label.text += "\nCould not identify the top three agents for offspring generation."
		_refresh_buttons()
		queue_redraw()
		return

	_capture_last_generation_best_brain(int(top_indices[0]))
	_simulation_4_advance_difficulty_if_ready()
	_simulation_5_advance_difficulty_if_ready()
	_begin_offspring_generation(top_indices)


func _update_summary() -> void:
	if agent_dynamics.is_empty():
		return

	var fitness_statistics: Dictionary = _current_fitness_statistics()
	var mean_fitness: float = float(fitness_statistics["mean"])
	var best_fitness: float = float(fitness_statistics["best"])
	var best_fitness_agent: int = int(fitness_statistics["best_agent"])
	var displayed_generation_real_time: float = generation_real_time_seconds
	var simulation_4_success_text: String = ""
	if simulation_selector.selected == 3 or simulation_selector.selected == 4 or simulation_selector.selected == 5:
		var arrival_count: int = 0
		var arrival_key: String = "simulation_5_first_arrival" if simulation_selector.selected == 4 else "simulation_4_first_arrival"
		for agent in agents:
			if float(agent.state.get(arrival_key, -1.0)) >= 0.0:
				arrival_count += 1
		simulation_4_success_text = "Reached target: %d / %d\n" % [arrival_count, agent_count]
	if generation_real_start_us > 0:
		displayed_generation_real_time = float(Time.get_ticks_usec() - generation_real_start_us) / 1000000.0

	var status = _state_text()
	var evolution_summary: String = (
		"Evolution:\n"
		+ "Generation: %d\n" % generation_number
		+ "Completed generations: %d" % completed_generation_count
	)

	if completed_generation_count > 0:
		var mean_fitness_across_generations: float = evolution_mean_fitness_sum / completed_generation_count
		evolution_summary += (
			"\nMean fitness across generations: %.3f\n" % mean_fitness_across_generations
			+ "Best generation mean fitness: %.3f  (Generation %d)\n"
			% [evolution_best_mean_fitness, evolution_best_mean_fitness_generation]
			+ "Best agent fitness: %.3f  (Generation %d)"
			% [evolution_best_agent_fitness, evolution_best_agent_fitness_generation]
		)

	summary_label.text = (
		"Status: %s\n" % status
		+ "Time: %.3f / %.3f s\n" % [experiment_elapsed_seconds, experiment_duration_seconds]
		+ "Agents: %d\n" % agent_count
		+ "Speed: %s | Neural iterations/s: %.1f\n" % [
			speed_selector.get_item_text(speed_selector.selected), effective_steps_per_second]
		+ "Target test: %d / %d\n" % [target_test_index + 1, target_test_count]
		+ "Mean fitness: %.3f\n" % mean_fitness
		+ "Best fitness: %.3f  (Agent %d)\n" % [best_fitness, best_fitness_agent + 1]
		+ simulation_4_success_text
		+ "Generation real time: %.3f s\n\n" % displayed_generation_real_time
		+ evolution_summary
	)


func _state_text() -> String:
	match experiment_state:
		ExperimentState.GENERATING:
			return "Generating"
		ExperimentState.RUNNING:
			return "Free Mode" if free_mode_active else "Running"
		ExperimentState.PAUSED:
			return "Paused"
		ExperimentState.OFFSPRING:
			return "Offspring"
		ExperimentState.FINISHED:
			return "Finished"
		_:
			return "Idle"


func _refresh_buttons() -> void:
	var generating = generation_active or experiment_state == ExperimentState.GENERATING or experiment_state == ExperimentState.OFFSPRING
	var simulation_active = _simulation_is_active()
	start_button.text = "Reset" if simulation_active else "Start"
	start_button.disabled = reset_pending
	load_button.disabled = generating
	pause_button.disabled = (
		experiment_state != ExperimentState.RUNNING
		and experiment_state != ExperimentState.PAUSED
	)
	save_button.disabled = generating or not _has_population_to_save()
	save_brain_button.disabled = last_generation_best_brain_data.is_empty()
	free_mode_button.disabled = experiment_state != ExperimentState.RUNNING
	# Simulation changes remain available during evolution; the selection
	# callback enforces neural input/output compatibility.
	simulation_selector.disabled = reset_pending
	num_agents_spinbox.editable = not simulation_active and not reset_pending
	dimension_i_field.editable = not simulation_active
	dimension_j_field.editable = not simulation_active
	dimension_k_field.editable = not simulation_active
	connections_density_field.editable = not simulation_active
	generation_time_field.editable = not simulation_active
	mutation_probability_field.editable = true

	if experiment_state != ExperimentState.PAUSED:
		pause_button.text = "Pause"


func _has_population_to_save() -> bool:
	return agent_dynamics.size() == agent_count or population_brains.size() == agent_count


func _setup_file_dialogs() -> void:
	save_dialog = FileDialog.new()
	save_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	save_dialog.access = FileDialog.ACCESS_FILESYSTEM
	save_dialog.use_native_dialog = true
	save_dialog.filters = PackedStringArray(["*.neuronsim ; Neurons simulation"])
	save_dialog.current_file = "simulation.neuronsim"
	save_dialog.file_selected.connect(_save_to_file)
	add_child(save_dialog)

	load_dialog = FileDialog.new()
	load_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	load_dialog.access = FileDialog.ACCESS_FILESYSTEM
	load_dialog.use_native_dialog = true
	load_dialog.filters = PackedStringArray(["*.neuronsim ; Neurons simulation"])
	load_dialog.file_selected.connect(_load_from_file)
	add_child(load_dialog)

	save_brain_dialog = FileDialog.new()
	save_brain_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	save_brain_dialog.access = FileDialog.ACCESS_FILESYSTEM
	save_brain_dialog.use_native_dialog = true
	save_brain_dialog.filters = PackedStringArray(["*.json ; JSON files"])
	save_brain_dialog.current_file = "brain.json"
	save_brain_dialog.file_selected.connect(_save_best_brain_to_file)
	add_child(save_brain_dialog)


func _open_save_dialog() -> void:
	if not _has_population_to_save():
		summary_label.text = "There is no population to save."
		return

	save_dialog.popup_centered()


func _open_load_dialog() -> void:
	load_dialog.popup_centered()


func _open_save_brain_dialog() -> void:
	if save_brain_button.disabled:
		return

	save_brain_dialog.popup_centered()


func _capture_last_generation_best_brain(agent_index: int) -> void:
	var source_genome: Dictionary = population_genomes[agent_index]
	var genome_data: Dictionary = {}

	for gene_name in BRAIN_EXPLORER_GENE_ORDER:
		var gene_text = ""
		for digit in source_genome[gene_name]:
			gene_text += str(int(digit))
		genome_data[gene_name] = gene_text

	last_generation_best_brain_data = {
		"format_version": 4,
		"model_variant": "Vsimple",
		"external_parameters": {
			"size_x": str(current_brain_size.x),
			"size_y": str(current_brain_size.y),
			"size_z": str(current_brain_size.z),
			"connections_density": str(connections_density),
			"num_inputs": str(num_inputs),
			"num_outputs": str(num_outputs),
		},
		"genome": genome_data
	}

	_refresh_buttons()


func _save_best_brain_to_file(path: String) -> void:
	var save_path = path if path.get_extension().to_lower() == "json" else path + ".json"
	var file = FileAccess.open(save_path, FileAccess.WRITE)

	if file == null:
		summary_label.text = "Could not save brain configuration. Error: %s" % error_string(FileAccess.get_open_error())
		return

	file.store_string(JSON.stringify(last_generation_best_brain_data, "\t"))
	summary_label.text = "Brain configuration saved to:\n%s" % save_path


func _save_to_file(path: String) -> void:
	# Persist what the editable field currently shows, including changes made
	# during the running generation, rather than a startup-only snapshot.
	if not _try_update_mutation_probability(mutation_probability_field.text):
		summary_label.text = "Cannot save: Mutation Probability must be a finite number between 0 and 1."
		return

	var save_path = path
	if save_path.get_extension().to_lower() != SAVE_EXTENSION:
		save_path += "." + SAVE_EXTENSION

	var data = {
		"format_version": SAVE_FORMAT_VERSION,
		"model_variant": "Vsimple",
		"external_parameters": {
			"size": current_brain_size,
			"num_inputs": num_inputs,
			"num_outputs": num_outputs,
			"agent_count": agent_count,
			"generation_time_seconds": experiment_duration_seconds,
			"connections_density": connections_density,
		},
		"experiment_parameters": {
			"simulation_number": active_simulation_index + 1,
			"agent_speed": agent_speed,
			"simulation_size": simulation_size,
			"agent_square_size": agent_square_size,
			"target_square_size": target_square_size,
			"mutation_probability": mutation_probability,
			"single_elite": single_elite_button.button_pressed,
		},
		"genomes": population_genomes.duplicate(true),
		"summary": summary_label.text,
		"evolution": {
			"generation_number": generation_number,
			"completed_generation_count": completed_generation_count,
			"mean_score_sum": evolution_mean_score_sum,
			"mean_fitness_sum": evolution_mean_fitness_sum,
			"best_mean_score": evolution_best_mean_score,
			"best_mean_score_generation": evolution_best_mean_score_generation,
			"best_agent_score": evolution_best_agent_score,
			"best_agent_score_generation": evolution_best_agent_score_generation,
			"best_mean_fitness": evolution_best_mean_fitness,
			"best_mean_fitness_generation": evolution_best_mean_fitness_generation,
			"best_agent_fitness": evolution_best_agent_fitness,
			"best_agent_fitness_generation": evolution_best_agent_fitness_generation,
			"mean_fitness_history": evolution_mean_fitness_history.duplicate(),
			"best_fitness_history": evolution_best_fitness_history.duplicate(),
			"genetic_diversity_history": genetic_diversity_history.duplicate(),
			"elite_genetic_diversity_history": elite_genetic_diversity_history.duplicate(),
			"current_elite_turnover_count": current_elite_turnover_count,
			"genetic_diversity_excluded_agent_indices": genetic_diversity_excluded_agent_indices.duplicate(),
			"previous_generation_elite_indices": previous_generation_elite_indices.duplicate(),
			"previous_generation_elite_signatures": previous_generation_elite_signatures.duplicate(),
			"simulation_4_difficulty": simulation_4_difficulty,
			"simulation_5_difficulty": simulation_5_difficulty,
			"simulation_6_difficulty": simulation_6_difficulty,
		},
		"results": {
			"elapsed_time_seconds": experiment_elapsed_seconds,
			"agent_positions": _current_agent_positions(),
			"fitness_integrals": fitness_integrals.duplicate(),
			"best_score_ever": best_score_ever,
			"best_distance_ever": best_distance_ever,
		},
	}

	var file = FileAccess.open(save_path, FileAccess.WRITE)
	if file == null:
		summary_label.text = "Could not save simulation. Error: %s" % error_string(FileAccess.get_open_error())
		return

	file.store_var(data)
	file.close()
	save_button.tooltip_text = "Last saved file: %s" % save_path


func _current_agent_positions() -> Array:
	var positions: Array = []
	positions.resize(agents.size())

	for agent_index in agents.size():
		positions[agent_index] = agents[agent_index].state.get("position", Vector2.ZERO)

	return positions


func _rebuild_population_from_genomes(genomes: Array, size: Vector3i, selected_num_inputs: int, selected_num_outputs: int) -> Dictionary:
	var result := {"brains": [], "error": ""}
	var rd: RenderingDevice = RenderingServer.create_local_rendering_device()
	if rd == null:
		result.error = "Could not create a local GPU device."
		return result
	var builder := BrainBuilderGPU.new()
	for agent_index in genomes.size():
		if not (genomes[agent_index] is Dictionary):
			result.error = "Invalid saved genome %d." % (agent_index+1)
			return _dispose_generation_population(result, rd)
		var genome: Genome = _genome_object(genomes[agent_index])
		if genome == null:
			result.error = "Could not reconstruct genome %d." % (agent_index+1)
			return _dispose_generation_population(result, rd)
		var brain := _gpu_build(builder, genome, size, selected_num_inputs, selected_num_outputs, Callable(), rd)
		if brain.is_empty():
			result.error = "Saved genome %d generated an invalid GPU brain." % (agent_index+1)
			return _dispose_generation_population(result, rd)
		result.brains.append(brain)
	return result


func _load_from_file(path: String) -> void:
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		summary_label.text = "Could not load simulation. Error: %s" % error_string(FileAccess.get_open_error())
		return

	var data = file.get_var()
	file.close()

	var validation_error = _validate_save_data(data)
	if not validation_error.is_empty():
		summary_label.text = "Invalid simulation file:\n%s" % validation_error
		return

	var external: Dictionary = data["external_parameters"]
	var experiment_parameters: Dictionary = data.get("experiment_parameters", {})
	var evolution: Dictionary = data.get("evolution", {})
	var size: Vector3i = external["size"]
	var generation_time = float(external["generation_time_seconds"])
	# Older saved simulations were generated with the original density eta = 1.
	connections_density = float(external.get("connections_density", 1.0))
	# Version 2 saves without this optional field remain valid: infer from genomes.
	var saved_agent_count: int = data["genomes"].size()

	_clear_runtime_state()
	agent_count = saved_agent_count
	num_agents_spinbox.set_value_no_signal(float(agent_count))
	_setup_simulation(simulation_selector.selected)
	_reset_evolution_statistics()
	if simulation_selector.selected == 3:
		# Old Simulation 5 saves use the legacy difficulty key.
		simulation_4_difficulty = clampi(int(evolution.get("simulation_4_difficulty", 3)), 3, SIMULATION_4_ANGULAR_SECTOR_COUNT)
		target_test_count = simulation_4_difficulty
		_update_simulation_4_num_tests_label()
	elif simulation_selector.selected == 4:
		simulation_5_difficulty = clampi(int(evolution.get("simulation_5_difficulty", 1)), 1, SIMULATION_5_MAX_DIFFICULTY)
		target_test_count = SIMULATION_5_NUM_TESTS
		_update_simulation_4_num_tests_label()
	elif simulation_selector.selected == 5:
		simulation_6_difficulty = clampi(int(evolution.get("simulation_6_difficulty", 3)), 3, SIMULATION_4_ANGULAR_SECTOR_COUNT)
		target_test_count = simulation_6_difficulty
		_update_simulation_4_num_tests_label()

	if not evolution.is_empty():
		generation_number = int(evolution.get("generation_number", 0))
		completed_generation_count = int(evolution.get("completed_generation_count", 0))
		evolution_mean_score_sum = float(evolution.get("mean_score_sum", 0.0))
		evolution_mean_fitness_sum = float(evolution.get("mean_fitness_sum", 0.0))
		evolution_best_mean_score = float(evolution.get("best_mean_score", -1.0))
		evolution_best_mean_score_generation = int(evolution.get("best_mean_score_generation", 0))
		evolution_best_agent_score = float(evolution.get("best_agent_score", -1.0))
		evolution_best_agent_score_generation = int(evolution.get("best_agent_score_generation", 0))
		evolution_best_mean_fitness = float(evolution.get("best_mean_fitness", -1.0))
		evolution_best_mean_fitness_generation = int(evolution.get("best_mean_fitness_generation", 0))
		evolution_best_agent_fitness = float(evolution.get("best_agent_fitness", -1.0))
		evolution_best_agent_fitness_generation = int(evolution.get("best_agent_fitness_generation", 0))
		var saved_mean_fitness_history = evolution.get("mean_fitness_history", PackedFloat64Array())
		var saved_best_fitness_history = evolution.get("best_fitness_history", PackedFloat64Array())
		var saved_genetic_diversity_history = evolution.get("genetic_diversity_history", PackedFloat64Array())
		var saved_elite_genetic_diversity_history = evolution.get("elite_genetic_diversity_history", PackedFloat64Array())
		var saved_elite_turnover_count = int(evolution.get("current_elite_turnover_count", 0))
		var saved_genetic_diversity_excluded_indices = evolution.get("genetic_diversity_excluded_agent_indices", [])
		var saved_elite_indices = evolution.get("previous_generation_elite_indices", [])
		var saved_elite_signatures = evolution.get("previous_generation_elite_signatures", [])
		current_elite_turnover_count = maxi(0, mini(3, saved_elite_turnover_count))
		previous_generation_elite_indices.clear()
		if saved_elite_indices is Array:
			for elite_index_variant in saved_elite_indices:
				var elite_index: int = int(elite_index_variant)
				if elite_index >= 0 and elite_index < agent_count and elite_index not in previous_generation_elite_indices:
					previous_generation_elite_indices.append(elite_index)
		previous_generation_elite_signatures.clear()
		if saved_elite_signatures is Array:
			for signature_variant in saved_elite_signatures:
				var signature: String = String(signature_variant)
				if not signature.is_empty():
					previous_generation_elite_signatures.append(signature)
		if saved_mean_fitness_history is PackedFloat64Array:
			evolution_mean_fitness_history = saved_mean_fitness_history.duplicate()
		if saved_best_fitness_history is PackedFloat64Array:
			evolution_best_fitness_history = saved_best_fitness_history.duplicate()
		if saved_genetic_diversity_history is PackedFloat64Array:
			genetic_diversity_history = saved_genetic_diversity_history.duplicate()
		# Older format-version-2 files have no elite history. Preserve their
		# original generation axis without fabricating measurements.
		var historical_count: int = genetic_diversity_history.size()
		elite_genetic_diversity_history.resize(historical_count)
		elite_genetic_diversity_history.fill(-1.0)
		if saved_elite_genetic_diversity_history is PackedFloat64Array:
			var known_count: int = mini(historical_count, saved_elite_genetic_diversity_history.size())
			for index in known_count:
				elite_genetic_diversity_history[historical_count - known_count + index] = saved_elite_genetic_diversity_history[saved_elite_genetic_diversity_history.size() - known_count + index]
		genetic_diversity_excluded_agent_indices.clear()
		if saved_genetic_diversity_excluded_indices is Array:
			for index_variant in saved_genetic_diversity_excluded_indices:
				genetic_diversity_excluded_agent_indices.append(int(index_variant))

	population_genomes = data["genomes"].duplicate(true)
	# The obsolete neuron_role locus in historical populations has no effect
	# on placement or reproduction under the prime-gap rule.
	for saved_genome in population_genomes:
		if saved_genome is Dictionary:
			saved_genome.erase("neuron_role")
	var rebuilt_population = _rebuild_population_from_genomes(
		population_genomes,
		size,
		int(external["num_inputs"]),
		int(external["num_outputs"])
	)
	var rebuild_error = str(rebuilt_population.get("error", ""))
	if not rebuild_error.is_empty():
		population_genomes.clear()
		summary_label.text = "Could not rebuild saved population:\n%s" % rebuild_error
		_refresh_buttons()
		queue_redraw()
		return

	population_brains = rebuilt_population["brains"]
	population_from_load = true
	loaded_population_size = size
	loaded_population_density = connections_density
	loaded_population_num_inputs = int(external["num_inputs"])
	loaded_population_num_outputs = int(external["num_outputs"])
	current_brain_size = size
	experiment_duration_seconds = generation_time

	if experiment_parameters.has("agent_speed"):
		agent_speed = float(experiment_parameters["agent_speed"])
	if experiment_parameters.has("simulation_size"):
		simulation_size = experiment_parameters["simulation_size"]
	if experiment_parameters.has("agent_square_size"):
		agent_square_size = float(experiment_parameters["agent_square_size"])
	if experiment_parameters.has("target_square_size"):
		target_square_size = float(experiment_parameters["target_square_size"])
	# Older saves default to the historical three-elite reproduction mode.
	single_elite_button.set_pressed_no_signal(bool(experiment_parameters.get("single_elite", false)))
	_update_single_elite_button()
	if experiment_parameters.has("mutation_probability"):
		mutation_probability = float(experiment_parameters["mutation_probability"])
	else:
		mutation_probability = 0.001

	dimension_i_field.text = str(size.x)
	dimension_j_field.text = str(size.y)
	dimension_k_field.text = str(size.z)
	connections_density_field.text = str(connections_density)
	generation_time_field.text = str(generation_time)
	mutation_probability_field.text = str(mutation_probability)
	if loaded_population_num_inputs == num_inputs and loaded_population_num_outputs == num_outputs:
		summary_label.text = str(data.get("summary", "Simulation loaded."))
	else:
		summary_label.text = (
			"Simulation loaded. Population preserved, but incompatible with the selected simulation.\n"
			+ "Population: %d inputs / %d outputs\n" % [loaded_population_num_inputs, loaded_population_num_outputs]
			+ "Simulation: %d inputs / %d outputs" % [num_inputs, num_outputs]
		)
	load_button.tooltip_text = "Loaded file: %s" % path

	_rebuild_history_graph_caches()
	_queue_history_graph_redraw()
	_refresh_buttons()
	queue_redraw()


func _validate_save_data(data: Variant) -> String:
	if not (data is Dictionary):
		return "Root value is not a Dictionary."

	var format_version = int(data.get("format_version", -1))
	if format_version != SAVE_FORMAT_VERSION or str(data.get("model_variant", "")) != "Vsimple":
		return "Unsupported save format or neural model variant."

	if not data.has("external_parameters") or not (data["external_parameters"] is Dictionary):
		return "Missing external parameters."

	if not data.has("genomes") or not (data["genomes"] is Array):
		return "Missing genome population."

	var external: Dictionary = data["external_parameters"]
	if not external.has("size") or typeof(external["size"]) != TYPE_VECTOR3I:
		return "Invalid brain size."
	var saved_density = external.get("connections_density", 1.0)
	if not (saved_density is float or saved_density is int):
		return "Invalid saved connections density."
	var eta: float = float(saved_density)
	if is_nan(eta) or is_inf(eta) or eta <= 0.0:
		return "Saved connections density must be a positive finite number."

	if int(external.get("num_inputs", -1)) < 0 or int(external.get("num_outputs", -1)) < 0:
		return "Invalid population input/output configuration."

	var genomes: Array = data["genomes"]

	if genomes.size() < int(num_agents_spinbox.min_value) or genomes.size() > int(num_agents_spinbox.max_value):
		return "Saved populations must contain %d to %d genomes." % [int(num_agents_spinbox.min_value), int(num_agents_spinbox.max_value)]
	if external.has("agent_count") and int(external["agent_count"]) != genomes.size():
		return "Saved agent count does not match the number of genomes."

	for genome_variant in genomes:
		if not (genome_variant is Dictionary):
			return "A saved genome is not a Dictionary."

	return ""
