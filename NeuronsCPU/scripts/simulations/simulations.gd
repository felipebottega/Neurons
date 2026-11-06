extends Node2D


const AgentClass = preload("res://scripts/simulations/agent.gd")
const TargetClass = preload("res://scripts/simulations/target.gd")
const BRAIN_PREVIEW_SCENE: PackedScene = preload("res://scenes/agent_brain_preview.tscn")


enum ExperimentState {
	IDLE,
	GENERATING,
	RUNNING,
	PAUSED,
	OFFSPRING,
	FINISHED,
}


# Population range and initial value are defined by NumAgents SpinBox in the scene.
var agent_count: int = 0
const SAVE_FORMAT_VERSION: int = 2
const SAVE_EXTENSION: String = "neuronsim"
const BRAIN_WORKER_BATCHES: int = 8
const SIMULATION_STEP_SECONDS: float = 1.0 / 60.0
const PROCESS_FRAME_DIVISORS = [1, 2, 4, 8]
const SPEED_STATS_WINDOW_US: int = 2000000
const BRAIN_PREVIEW_UPDATE_INTERVAL_US: int = 100000
const RUNTIME_UI_UPDATE_INTERVAL_US: int = 16667  # 60 Hz maximum for the simulation visualization, unchanged.
const NON_SIMULATION_UI_UPDATE_INTERVAL_US: int = 250000  # 4 Hz maximum for labels and other UI.
const SIMULATION_4_COLLISION_EPSILON: float = 0.001
const SIMULATION_4_TOUCH_TOLERANCE: float = 0.05
const SIMULATION_5_ANGULAR_SECTOR_COUNT: int = 12
const SIMULATION_5_DIFFICULTY_BASE_THRESHOLD: float = 0.7
const SIMULATION_5_DIFFICULTY_THRESHOLD_STEP: float = 0.02
# Simulation 6 shares Simulation 4's environment and collision solver,
# while adopting the final-second fitness and sector curriculum of the GPU version.
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

const HEATMAP_PHENOTYPE_KEYS = [
	"max_connections",
	"beta",
	"exponential_factor",
	"refractory_strength",
	"structural_plasticity",
	"lambda",
	"nu",
	"mu",
	"input_gain",
	"mean_activation_threshold",
	"mean_decay_factor",
	"mean_retention_factor",
	"mean_polarity_factor",
	"mean_hebbian_plasticity_rate",
	"mean_modulatory_release_factor",
	"mean_modulatory_sensitivity",
	"mean_connection_energy",
	"mean_connections_per_neuron",
	"mean_connection_weight",
]
const HEATMAP_PHENOTYPE_LABELS = {
	"max_connections": "N_max",
	"beta": "Beta",
	"exponential_factor": "Exponential factor",
	"refractory_strength": "Maximum fatigue",
	"structural_plasticity": "Structural plasticity",
	"lambda": "Modulatory persistence",
	"nu": "Modulatory spread",
	"mu": "Eligibility persistence",
	"input_gain": "Input influence",
	"mean_activation_threshold": "Mean activation threshold",
	"mean_decay_factor": "Mean decay factor",
	"mean_retention_factor": "Mean retention factor",
	"mean_polarity_factor": "Mean polarity factor",
	"mean_hebbian_plasticity_rate": "Mean Hebbian plasticity rate",
	"mean_modulatory_release_factor": "Mean modulatory release factor",
	"mean_modulatory_sensitivity": "Mean modulatory sensitivity",
	"mean_connection_energy": "Mean connection energy",
	"mean_connections_per_neuron": "Mean connections per neuron",
	"mean_connection_weight": "Mean connection weight",
}
const HEATMAP_RESOLUTIONS = [10, 30, 100, 300]

const HEATMAP_PHENOTYPE_INDEX = {
	"max_connections": 0,
	"beta": 1,
	"exponential_factor": 2,
	"refractory_strength": 3,
	"structural_plasticity": 4,
	"lambda": 5,
	"nu": 6,
	"mu": 7,
	"input_gain": 8,
	"mean_activation_threshold": 9,
	"mean_decay_factor": 10,
	"mean_retention_factor": 11,
	"mean_polarity_factor": 12,
	"mean_hebbian_plasticity_rate": 13,
	"mean_modulatory_release_factor": 14,
	"mean_modulatory_sensitivity": 15,
	"mean_connection_energy": 16,
	"mean_connections_per_neuron": 17,
	"mean_connection_weight": 18,
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

@export_group("Simulation 4: Evolution")
@export_range(0.0, 0.3, 0.01) var simulation_4_progress_weight: float = 0.10

var brain_profiling: bool = false
var applied_brain_profiling: bool = false
var profiling_print_interval_steps: int = 300

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


@onready var dimension_i_field: LineEdit = $LineEdit
@onready var dimension_j_field: LineEdit = $LineEdit2
@onready var dimension_k_field: LineEdit = $LineEdit3
@onready var connections_density_field: LineEdit = $LineEdit6
@onready var generation_time_field: LineEdit = $LineEdit4
@onready var mutation_probability_field: LineEdit = $LineEdit5
@onready var start_button: Button = $Start
@onready var pause_button: Button = $Pause
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
@onready var simulation_5_num_tests_label: Label = $SimulationSpace/NumTestsLabel
@onready var speed_selector: OptionButton = $Speed
@onready var simulation_selector: OptionButton = $SimulationSelect
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
var last_generation_ui_update_us: int = 0
var current_brain_size: Vector3i = Vector3i.ZERO
var mutation_probability: float = 0.001
var connections_density: float = 1.0
# Read-only native brain preview, disabled until requested.
var agent_brain_preview: AgentBrainPreview
var preview_agent_index: int = -1
var last_brain_preview_update_us: int = 0
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
var simulation_4_obstacle_center: Vector2 = Vector2.ZERO
var simulation_4_obstacle_size: Vector2 = Vector2.ZERO
var simulation_4_obstacle_angle: float = 0.0
var simulation_5_target_sector_order: Array[int] = []
var simulation_5_difficulty: int = 3
var simulation_6_target_sector_order: Array[int] = []
var simulation_6_difficulty: int = 3

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
var evolution_graph_mean_points_cache: PackedVector2Array = PackedVector2Array()
var evolution_graph_best_points_cache: PackedVector2Array = PackedVector2Array()
var evolution_graph_minimum_value_cache: float = 0.0
var evolution_graph_maximum_value_cache: float = 1.0
var genetic_diversity_graph_points_cache: PackedVector2Array = PackedVector2Array()
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

var generation_thread: Thread
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

var profiling_steps_accumulated: int = 0
var profiling_total_us_accumulated: int = 0
var profiling_brain_cpu_us_accumulated: int = 0
var profiling_prepare_and_firing_us_accumulated: int = 0
var profiling_modulatory_field_us_accumulated: int = 0
var profiling_connections_us_accumulated: int = 0
var profiling_homeostatic_us_accumulated: int = 0
var profiling_propagation_statistics_us_accumulated: int = 0
var profiling_state_update_us_accumulated: int = 0
var profiling_structural_plasticity_us_accumulated: int = 0


func _ready() -> void:
	previous_fps_limit = Engine.max_fps
	previous_vsync_mode = DisplayServer.window_get_vsync_mode()
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	agent_count = int(num_agents_spinbox.value)
	num_agents_spinbox.value_changed.connect(_on_num_agents_value_changed)
	history_graphs_layer = Node2D.new()
	history_graphs_layer.draw.connect(_draw_history_graphs.bind(history_graphs_layer))
	add_child(history_graphs_layer)

	elite_button.toggle_mode = true
	elite_button.button_pressed = false
	elite_button.text = "OFF"
	elite_button.toggled.connect(_on_elite_toggled)
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


func _on_num_agents_value_changed(value: float) -> void:
	if _simulation_is_active() or reset_pending:
		num_agents_spinbox.set_value_no_signal(float(agent_count))
		return
	var selected_count := int(value)
	if selected_count == agent_count:
		return
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

	# Loaded brain snapshots may already contain lifetime changes in weights and
	# connectivity, so the loaded generation is intentionally omitted.
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
	values[int(HEATMAP_PHENOTYPE_INDEX["refractory_strength"])] = float(brain.get("refractory_strength", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["structural_plasticity"])] = float(brain.get("structural_plasticity", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["lambda"])] = float(brain.get("lambda", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["nu"])] = float(brain.get("nu", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["mu"])] = float(brain.get("mu", 0.0))
	values[int(HEATMAP_PHENOTYPE_INDEX["input_gain"])] = float(brain.get("input_gain", 0.0))

	var neurons: Array = brain.get("neurons", [])
	if neurons.is_empty():
		return values

	var activation_threshold_sum: float = 0.0
	var decay_factor_sum: float = 0.0
	var retention_factor_sum: float = 0.0
	var polarity_factor_sum: float = 0.0
	var hebbian_plasticity_rate_sum: float = 0.0
	var modulatory_release_factor_sum: float = 0.0
	var modulatory_sensitivity_sum: float = 0.0
	var connection_energy_sum: float = 0.0
	var connection_weight_sum: float = 0.0
	var connection_count: int = 0

	for neuron_variant in neurons:
		var neuron: Dictionary = neuron_variant
		activation_threshold_sum += float(neuron.get("activation_threshold", 0.0))
		decay_factor_sum += float(neuron.get("decay_factor", 0.0))
		retention_factor_sum += float(neuron.get("retention_factor", 0.0))
		polarity_factor_sum += float(neuron.get("polarity_factor", 0.0))
		hebbian_plasticity_rate_sum += float(neuron.get("hebbian_plasticity_rate", 0.0))
		modulatory_release_factor_sum += float(neuron.get("modulatory_release_factor", 0.0))
		modulatory_sensitivity_sum += float(neuron.get("modulatory_sensitivity", 0.0))
		connection_energy_sum += float(neuron.get("connections_value", 0.0))

		var connections: Array = neuron.get("connections", [])
		connection_count += connections.size()
		for connection_variant in connections:
			var connection: Dictionary = connection_variant
			connection_weight_sum += float(connection.get("weight", 0.0))

	var neuron_count: float = float(neurons.size())
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_activation_threshold"])] = activation_threshold_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_decay_factor"])] = decay_factor_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_retention_factor"])] = retention_factor_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_polarity_factor"])] = polarity_factor_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_hebbian_plasticity_rate"])] = hebbian_plasticity_rate_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_modulatory_release_factor"])] = modulatory_release_factor_sum / neuron_count
	values[int(HEATMAP_PHENOTYPE_INDEX["mean_modulatory_sensitivity"])] = modulatory_sensitivity_sum / neuron_count
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
	if generation_thread != null:
		return

	if (
		experiment_state == ExperimentState.GENERATING
		or experiment_state == ExperimentState.RUNNING
		or experiment_state == ExperimentState.PAUSED
		or experiment_state == ExperimentState.OFFSPRING
	):
		return

	_setup_simulation(index)
	_clear_runtime_state()
	_reset_evolution_statistics()

	if population_from_load:
		if loaded_population_num_inputs == num_inputs and loaded_population_num_outputs == num_outputs:
			summary_label.text = "Loaded population ready for the selected simulation. Evolution history reset."
		else:
			summary_label.text = (
				"Loaded population preserved, but incompatible with the selected simulation.\n"
				+ "Population: %d inputs / %d outputs\n" % [loaded_population_num_inputs, loaded_population_num_outputs]
				+ "Simulation: %d inputs / %d outputs" % [num_inputs, num_outputs]
			)
	else:
		summary_label.text = "Ready."

	_refresh_buttons()
	queue_redraw()


func _setup_simulation(index: int) -> void:
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
	# Fit thirteen sensor values in the existing agent information panel.
	agent_info_label.add_theme_font_size_override("font_size", 12 if index == 5 else 14)
	_update_simulation_5_num_tests_label()


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
	# Four goal-direction inputs, one binary touch input and four motor outputs.
	# Fitness measures progress and actual arrival, not proximity over time.
	num_inputs = 5
	num_outputs = 4
	target_test_count = 3
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
	# Same agents, inputs, outputs and obstacles as Simulation 4.
	# Curriculum: three tests initially, progressing up to 12 tests.
	# Fitness continues to use only proximity during the final second.
	num_inputs = 5
	num_outputs = 4
	target_test_count = simulation_5_difficulty
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_5_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_5_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_5_fitness")
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
		Callable(self, "_simulation_5_reset_target"),
		Callable(self, "_simulation_1_update_target")
	)


func _setup_simulation_6() -> void:
	# Simulation 4 obstacle/physics with 13 channels, using the final-second
	# proximity fitness and sector curriculum of the current GPU Simulation 6.
	# Inputs 1-4 = target displacement, 5 = binary contact;
	# 6-13 = W, E, N, S, NW, NE, SW, SE obstacle proximity.
	num_inputs = 13
	num_outputs = 4
	target_test_count = simulation_6_difficulty
	target_position_change_interval_seconds = 0.0
	evaluation_behavior = Callable(self, "_simulation_5_evaluate_agent")
	fitness_reset_behavior = Callable(self, "_simulation_1_reset_fitness")
	fitness_accumulation_behavior = Callable(self, "_simulation_5_accumulate_fitness")
	fitness_behavior = Callable(self, "_simulation_5_fitness")
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
		Callable(self, "_simulation_6_reset_target"),
		Callable(self, "_simulation_1_update_target")
	)


func _exit_tree() -> void:
	Engine.max_fps = previous_fps_limit
	DisplayServer.window_set_vsync_mode(previous_vsync_mode)
	if generation_thread != null and generation_thread.is_started():
		generation_thread.wait_to_finish()


func _process(_delta: float) -> void:
	if experiment_state == ExperimentState.RUNNING:
		_advance_simulation_on_process_frame()
		if experiment_state == ExperimentState.RUNNING:
			_update_runtime_ui_if_due()
			if show_brain_button.button_pressed:
				_update_brain_preview()

	if generation_thread == null:
		return

	if reset_pending:
		if generation_thread.is_alive():
			return

		generation_thread.wait_to_finish()
		generation_thread = null
		_complete_reset()
		return

	if generation_thread.is_alive():
		var current_time_us: int = Time.get_ticks_usec()
		if current_time_us - last_generation_ui_update_us >= NON_SIMULATION_UI_UPDATE_INTERVAL_US:
			last_generation_ui_update_us = current_time_us
			generation_progress_mutex.lock()
			var agent_number = generation_agent_number
			var brain_progress = generation_brain_progress
			var attempt_number = generation_attempt_number
			generation_progress_mutex.unlock()

			var generation_title: String = "Generating offspring brains..." if experiment_state == ExperimentState.OFFSPRING else "Generating brains..."
			summary_label.text = (
				generation_title + "\n"
				+ "Agent: %d / %d\n" % [agent_number, agent_count]
				+ "Attempt: %d\n" % attempt_number
				+ "Brain generation: %d%%" % brain_progress
			)
		return

	var result = generation_thread.wait_to_finish()
	generation_thread = null

	if not (result is Dictionary):
		population_brains.clear()
		population_genomes.clear()
		experiment_state = ExperimentState.IDLE
		summary_label.text = "Brain generation failed: invalid worker result."
		_refresh_buttons()
		queue_redraw()
		return

	var error_message = str(result.get("error", ""))
	if not error_message.is_empty():
		population_brains.clear()
		population_genomes.clear()
		experiment_state = ExperimentState.IDLE
		summary_label.text = error_message
		_refresh_buttons()
		queue_redraw()
		return

	population_brains = result.get("brains", [])
	population_genomes = result.get("genomes", [])
	var excluded_indices_variant = result.get("genetic_diversity_excluded_agent_indices", [])
	genetic_diversity_excluded_agent_indices.clear()
	if excluded_indices_variant is Array:
		for index_variant in excluded_indices_variant:
			genetic_diversity_excluded_agent_indices.append(int(index_variant))
	population_from_load = false

	if population_brains.size() != agent_count or population_genomes.size() != agent_count:
		population_brains.clear()
		population_genomes.clear()
		experiment_state = ExperimentState.IDLE
		summary_label.text = "Brain generation failed: incomplete population."
		_refresh_buttons()
		queue_redraw()
		return

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
		_queue_inputs_and_accumulate_fitness(SIMULATION_STEP_SECONDS, false)
		_advance_all_brains(SIMULATION_STEP_SECONDS)
		_move_agents(SIMULATION_STEP_SECONDS)
		summary_elapsed_seconds += SIMULATION_STEP_SECONDS
		return true
	var remaining_time: float = experiment_duration_seconds - experiment_elapsed_seconds
	if remaining_time <= 0.0:
		_finish_experiment()
		return false
	var step_delta: float = minf(SIMULATION_STEP_SECONDS, remaining_time)
	target.update(step_delta)
	_queue_inputs_and_accumulate_fitness(step_delta, true)
	_advance_all_brains(step_delta)
	_move_agents(step_delta)
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

	# Keep the simulation visualization and metric sampling at the original 60 Hz.
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

func _on_elite_toggled(enabled: bool) -> void:
	elite_button.text = "ON" if enabled else "OFF"
	if enabled and selected_agent_index >= 0 and selected_agent_index not in previous_generation_elite_indices:
		selected_agent_index = -1
		_update_selected_agent_info()
	queue_redraw()


func _on_show_brain_toggled(enabled: bool) -> void:
	show_brain_button.text = "ON" if enabled else "OFF"
	agent_info_label.visible = not enabled
	agent_info_label_2.visible = not enabled
	agent_info_label_3.visible = not enabled
	if enabled:
		# No SubViewport (or its 3D resources) exists while OFF.
		agent_brain_preview = BRAIN_PREVIEW_SCENE.instantiate() as AgentBrainPreview
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
	last_brain_preview_update_us = 0
	if agent_brain_preview != null and is_instance_valid(agent_brain_preview):
		agent_brain_preview.clear_brain()


func _update_brain_preview(force: bool = false) -> void:
	if not show_brain_button.button_pressed or agent_brain_preview == null:
		return
	var valid_agent: bool = (selected_agent_index >= 0 and selected_agent_index < agents.size()
		and selected_agent_index < population_brains.size() and selected_agent_index < agent_dynamics.size())
	if not valid_agent:
		if preview_agent_index != -1:
			_invalidate_brain_preview()
		return
	if preview_agent_index != selected_agent_index:
		agent_brain_preview.set_brain(population_brains[selected_agent_index])
		preview_agent_index = selected_agent_index
		force = true
	var now_us: int = Time.get_ticks_usec()
	if not force and now_us - last_brain_preview_update_us < BRAIN_PREVIEW_UPDATE_INTERVAL_US:
		return
	last_brain_preview_update_us = now_us
	var dynamics: BrainDynamicsNative = agent_dynamics[selected_agent_index]
	var signals: PackedFloat32Array = dynamics.get_visual_signals()
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

	if simulation_selector.selected == 3 or simulation_selector.selected == 4 or simulation_selector.selected == 5:
		_draw_simulation_4_obstacle(origin)

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
		var agent_position: Vector2 = agent.state["position"]
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

	var legend_x: float = graph_origin.x + evolution_graph_size.x - 245.0
	var legend_y: float = graph_origin.y + 14.0
	drawer.draw_line(Vector2(legend_x, legend_y - 5.0), Vector2(legend_x + 22.0, legend_y - 5.0), evolution_graph_mean_color, 2.0)
	drawer.draw_string(default_font, Vector2(legend_x + 28.0, legend_y), "Mean fitness", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_line(Vector2(legend_x + 124.0, legend_y - 5.0), Vector2(legend_x + 146.0, legend_y - 5.0), evolution_graph_best_color, 2.0)
	drawer.draw_string(default_font, Vector2(legend_x + 150.0, legend_y), "Best fitness", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)


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

	if genetic_diversity_graph_points_cache.is_empty():
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
	else:
		drawer.draw_polyline(genetic_diversity_graph_points_cache, evolution_graph_mean_color, 2.0, true)

	drawer.draw_string(default_font, graph_origin + Vector2(6.0, 14.0), "Genetic diversity (normalized Hamming distance, log scale)", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)
	drawer.draw_string(default_font, graph_origin + Vector2(margin_left, evolution_graph_size.y - 6.0), "Generation", HORIZONTAL_ALIGNMENT_LEFT, -1.0, font_size, Color.WHITE)


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

	if genetic_diversity_history.is_empty() or not is_instance_valid(simulation_space):
		return

	var diversity_history: PackedFloat64Array = genetic_diversity_history
	match genetic_diversity_graph_selector.selected:
		1:
			diversity_history = _moving_average(genetic_diversity_history, 10)
		2:
			diversity_history = _moving_average(genetic_diversity_history, 50)

	var minimum_positive: float = 1.0
	for value in diversity_history:
		if value > 0.0:
			minimum_positive = min(minimum_positive, value)

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
		var value: float = max(diversity_history[index], minimum_log_value)
		var log_value: float = log(value) / log(10.0)
		var y: float = plot_origin.y + plot_size.y * (-log_value) / log_range
		genetic_diversity_graph_points_cache[index] = Vector2(x, y)

func _flatten_genome(genome: Dictionary) -> PackedByteArray:
	var flattened: PackedByteArray = PackedByteArray()

	for gene_name in BRAIN_EXPLORER_GENE_ORDER:
		for digit in genome[gene_name]:
			flattened.append(int(digit))

	return flattened


func _population_genetic_diversity() -> float:
	var included_agent_indices: Array[int] = []
	for agent_index in population_genomes.size():
		if agent_index not in genetic_diversity_excluded_agent_indices:
			included_agent_indices.append(agent_index)

	if included_agent_indices.size() < 2:
		return 0.0

	var flattened_genomes: Array = []
	flattened_genomes.resize(included_agent_indices.size())

	for included_index in included_agent_indices.size():
		flattened_genomes[included_index] = _flatten_genome(population_genomes[included_agent_indices[included_index]])

	var genome_length: int = flattened_genomes[0].size()
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

	var parameters = _read_parameters()
	if parameters.is_empty():
		return

	current_brain_size = parameters["size"]
	experiment_duration_seconds = float(parameters["generation_time"])
	mutation_probability = float(parameters["mutation_probability"])
	connections_density = float(parameters["connections_density"])
	_clear_runtime_state()
	_reset_experiment_entities()

	var loaded_population_io_compatible = (
		loaded_population_num_inputs == num_inputs
		and loaded_population_num_outputs == num_outputs
	)
	var can_use_loaded_population = (
		population_from_load
		and loaded_population_io_compatible
		and loaded_population_size == current_brain_size
		and is_equal_approx(loaded_population_density, connections_density)
		and population_brains.size() == agent_count
		and population_genomes.size() == agent_count
	)

	if can_use_loaded_population:
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
			if free_mode_active:
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

	elif event is InputEventMouseMotion and free_mode_active and target_dragging:
		var half_target_size: float = target_square_size * 0.5
		var mouse_position: Vector2 = simulation_space.to_local(get_global_mouse_position())
		var dragged_position: Vector2 = mouse_position - target_drag_offset
		target.position = Vector2(
			clampf(dragged_position.x, half_target_size, simulation_size.x - half_target_size),
			clampf(dragged_position.y, half_target_size, simulation_size.y - half_target_size)
		)
		if simulation_selector.selected == 3 or simulation_selector.selected == 4 or simulation_selector.selected == 5:
			_simulation_4_update_obstacle(false)
		queue_redraw()
		get_viewport().set_input_as_handled()


func _mouse_is_over_interactive_control() -> bool:
	var hovered_control: Control = get_viewport().gui_get_hovered_control()
	if hovered_control is BaseButton or hovered_control is LineEdit or hovered_control is SpinBox:
		return true
	return (show_brain_button.button_pressed and agent_brain_preview != null
		and agent_brain_preview.get_global_rect().has_point(get_viewport().get_mouse_position()))


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
		var agent_position: Vector2 = agent.state["position"]
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
	var density_text = connections_density_field.text.strip_edges()

	if not i_text.is_valid_int() or not j_text.is_valid_int() or not k_text.is_valid_int():
		summary_label.text = "Dimensions I, J and K must be integers."
		return {}

	if not generation_time_text.is_valid_float():
		summary_label.text = "Generation Time must be a valid number in seconds."
		return {}

	if not density_text.is_valid_float() or is_nan(float(density_text)) or is_inf(float(density_text)) or float(density_text) <= 0.0:
		summary_label.text = "Connections Density must be a positive finite number."
		return {}

	if not mutation_probability_text.is_valid_float():
		summary_label.text = "Mutation Probability must be a valid number."
		return {}

	var size = Vector3i(int(i_text), int(j_text), int(k_text))
	var generation_time = float(generation_time_text)
	var selected_mutation_probability = float(mutation_probability_text)

	if size.x < 2 or size.y < 2 or size.z < 2:
		summary_label.text = "Dimensions I, J and K must all be at least 2."
		return {}

	if generation_time <= 0.0:
		summary_label.text = "Generation Time must be greater than zero."
		return {}

	if is_nan(selected_mutation_probability) or is_inf(selected_mutation_probability) or selected_mutation_probability < 0.0 or selected_mutation_probability > 1.0:
		summary_label.text = "Mutation Probability must be between 0 and 1."
		return {}

	if simulation_size.x <= agent_square_size or simulation_size.y <= agent_square_size:
		summary_label.text = "Simulation size must be larger than the agent square size."
		return {}

	return {
		"size": size,
		"generation_time": generation_time,
		"mutation_probability": selected_mutation_probability,
		"connections_density": float(density_text),
	}


func _begin_population_generation() -> void:
	experiment_state = ExperimentState.GENERATING
	generation_progress_mutex.lock()
	generation_cancel_requested = false
	generation_agent_number = 1
	generation_brain_progress = 0
	generation_attempt_number = 1
	generation_progress_mutex.unlock()
	last_generation_ui_update_us = 0

	generation_thread = Thread.new()
	var error = generation_thread.start(
		_generate_population_thread.bind(
			current_brain_size,
			max_brain_generation_attempts,
			num_inputs,
			num_outputs
		)
	)

	if error != OK:
		generation_thread = null
		experiment_state = ExperimentState.IDLE
		summary_label.text = "Could not start population generation. Error: %s" % error_string(error)

	_refresh_buttons()
	queue_redraw()


func _generate_population_thread(size: Vector3i, maximum_attempts: int, selected_num_inputs: int, selected_num_outputs: int) -> Dictionary:
	var result = {
		"brains": [],
		"genomes": [],
		"genetic_diversity_excluded_agent_indices": [],
		"error": "",
	}
	var builder: BrainBuilderNative = BrainBuilderNative.new()

	for agent_index in agent_count:
		if _is_generation_cancel_requested():
			return result

		var accepted = false

		for attempt_index in maximum_attempts:
			if _is_generation_cancel_requested():
				return result

			_set_generation_progress_from_thread(0, agent_index + 1, attempt_index + 1)
			var genome: GenomeNative = GenomeNative.new()
			var progress_callback = _set_generation_progress_from_thread.bind(agent_index + 1, attempt_index + 1)
			var brain = builder.build(genome, size, selected_num_inputs, selected_num_outputs, progress_callback, connections_density)

			if _is_generation_cancel_requested():
				return result

			if not brain.is_empty() and bool(brain.get("valid", false)):
				result["brains"].append(brain)
				result["genomes"].append(genome.to_dictionary())
				accepted = true
				break


		if not accepted:
			result["error"] = (
				"Could not generate a valid brain for agent %d after %d attempts."
				% [agent_index + 1, maximum_attempts]
			)
			return result

	return result


func _begin_offspring_generation(top_indices: Array[int]) -> void:
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
	# starting the worker thread, so only the bottom five are replaced.
	var ranked_non_elite_indices: Array[int] = []
	for agent_index in agent_count:
		if agent_index not in top_indices:
			ranked_non_elite_indices.append(agent_index)
	ranked_non_elite_indices.sort_custom(_has_higher_generation_fitness)

	generation_thread = Thread.new()
	var error = generation_thread.start(
		_generate_offspring_population_thread.bind(
			parent_genomes,
			top_indices,
			ranked_non_elite_indices,
			current_brain_size,
			max_brain_generation_attempts,
			mutation_probability,
			num_inputs,
			num_outputs
		)
	)

	if error != OK:
		generation_thread = null
		experiment_state = ExperimentState.IDLE
		summary_label.text = "Could not start offspring generation. Error: %s" % error_string(error)

	_refresh_buttons()
	queue_redraw()


func _generate_offspring_population_thread(
		parent_genomes: Array,
		top_indices: Array[int],
		ranked_non_elite_indices: Array[int],
		size: Vector3i,
		maximum_attempts: int,
		selected_mutation_probability: float,
		selected_num_inputs: int,
		selected_num_outputs: int
	) -> Dictionary:
	var result = {
		"brains": [],
		"genomes": [],
		"genetic_diversity_excluded_agent_indices": [],
		"error": "",
	}

	if parent_genomes.size() != agent_count or top_indices.size() != 3 or ranked_non_elite_indices.size() != agent_count - 3:
		result["error"] = "Offspring generation received an incomplete parent population."
		return result

	result["brains"].resize(agent_count)
	result["genomes"].resize(agent_count)

	var builder: BrainBuilderNative = BrainBuilderNative.new()
	var offspring: OffspringNative = OffspringNative.new()
	var parent_native_genomes: Array = []
	parent_native_genomes.resize(agent_count)

	for agent_index in agent_count:
		if _is_generation_cancel_requested():
			return result

		var parent_genome: GenomeNative = offspring.genome_from_dictionary(parent_genomes[agent_index])
		if parent_genome == null:
			result["error"] = "Could not reconstruct parent genome %d." % (agent_index + 1)
			return result
		parent_native_genomes[agent_index] = parent_genome

	# The three selected genomes survive unchanged. Their brains are rebuilt from
	# the genotype so lifetime plasticity is not inherited by the next generation.
	for elite_rank in 3:
		if _is_generation_cancel_requested():
			return result

		var elite_index: int = int(top_indices[elite_rank])
		var elite_genome: GenomeNative = parent_native_genomes[elite_index]
		var generated_agent_number: int = elite_rank + 1
		_set_generation_progress_from_thread(0, generated_agent_number, 1)
		var progress_callback = _set_generation_progress_from_thread.bind(generated_agent_number, 1)
		var elite_brain = builder.build(elite_genome, size, selected_num_inputs, selected_num_outputs, progress_callback, connections_density)

		if _is_generation_cancel_requested():
			return result

		if elite_brain.is_empty() or not bool(elite_brain.get("valid", false)):
			result["error"] = "Elite genome %d generated an invalid brain." % (elite_rank + 1)
			return result

		result["brains"][elite_index] = elite_brain
		result["genomes"][elite_index] = elite_genome.to_dictionary()

	# All but the five worst non-elites can participate as existing parents.
	# The bottom five are excluded regardless of their previous genotype.
	var eligible_partner_indices: Array[int] = []
	var worst_agent_indices: Array[int] = []
	for rank in ranked_non_elite_indices.size():
		if rank < ranked_non_elite_indices.size() - 5:
			eligible_partner_indices.append(ranked_non_elite_indices[rank])
		else:
			worst_agent_indices.append(ranked_non_elite_indices[rank])

	var pairing_rng: RandomNumberGenerator = RandomNumberGenerator.new()
	pairing_rng.randomize()
	_shuffle_indices(eligible_partner_indices, pairing_rng)
	_shuffle_indices(worst_agent_indices, pairing_rng)
	result["genetic_diversity_excluded_agent_indices"] = worst_agent_indices.duplicate()

	# At 100 agents historical proportions are 44/29/19 among the 92 partners.
	var first_elite_end: int = roundi(float(eligible_partner_indices.size()) * 44.0 / 92.0)
	var second_elite_end: int = roundi(float(eligible_partner_indices.size()) * 73.0 / 92.0)
	for partner_slot in eligible_partner_indices.size():
		if _is_generation_cancel_requested():
			return result

		var elite_rank: int
		if partner_slot < first_elite_end:
			elite_rank = 0
		elif partner_slot < second_elite_end:
			elite_rank = 1
		else:
			elite_rank = 2

		var elite_index: int = int(top_indices[elite_rank])
		var partner_index: int = eligible_partner_indices[partner_slot]
		var elite_genome: GenomeNative = parent_native_genomes[elite_index]
		var partner_genome: GenomeNative = parent_native_genomes[partner_index]
		var generated_agent_number: int = partner_slot + 4
		var accepted: bool = false

		for attempt_index in maximum_attempts:
			if _is_generation_cancel_requested():
				return result

			_set_generation_progress_from_thread(0, generated_agent_number, attempt_index + 1)
			var child_genome: GenomeNative = offspring.generate(
				elite_genome,
				partner_genome,
				selected_mutation_probability
			)

			if child_genome == null:
				result["error"] = "Could not generate offspring genome for agent %d." % (partner_index + 1)
				return result

			var progress_callback = _set_generation_progress_from_thread.bind(generated_agent_number, attempt_index + 1)
			var child_brain = builder.build(child_genome, size, selected_num_inputs, selected_num_outputs, progress_callback, connections_density)

			if _is_generation_cancel_requested():
				return result

			if not child_brain.is_empty() and bool(child_brain.get("valid", false)):
				result["brains"][partner_index] = child_brain
				result["genomes"][partner_index] = child_genome.to_dictionary()
				accepted = true
				break

		if not accepted:
			result["error"] = (
				"Could not generate a valid offspring for agent %d after %d attempts."
				% [partner_index + 1, maximum_attempts]
			)
			return result

	# Generate five independent immigrant genotypes and draw three without
	# replacement for one crossover with each elite. The two unselected
	# genomes enter the population unchanged, subject to brain validation.
	var immigrant_genomes: Array = []
	var immigrant_genome_order: Array[int] = []
	for immigrant_index in worst_agent_indices.size():
		immigrant_genomes.append(GenomeNative.new())
		immigrant_genome_order.append(immigrant_index)
	_shuffle_indices(immigrant_genome_order, pairing_rng)

	# Discard the five worst *old* genotypes. Three new genotypes are crossed
	# with the elites (one per elite), two remain entirely random.
	for random_slot in worst_agent_indices.size():
		if _is_generation_cancel_requested():
			return result

		var agent_index: int = worst_agent_indices[random_slot]
		var generated_agent_number: int = eligible_partner_indices.size() + random_slot + 4
		var accepted: bool = false
		var random_parent_genome: GenomeNative = immigrant_genomes[immigrant_genome_order[random_slot]]

		for attempt_index in maximum_attempts:
			if _is_generation_cancel_requested():
				return result

			_set_generation_progress_from_thread(0, generated_agent_number, attempt_index + 1)
			var genome: GenomeNative
			if random_slot < 3:
				genome = offspring.generate(
					parent_native_genomes[int(top_indices[random_slot])],
					random_parent_genome,
					selected_mutation_probability
				)
				if genome == null:
					result["error"] = "Could not generate offspring genome for agent %d." % (agent_index + 1)
					return result
			else:
				# If a pure-random genome produces an invalid brain, resample.
				# Each attempt must sample a fresh genome, as at population startup.
				if attempt_index > 0:
					random_parent_genome = GenomeNative.new()
				genome = random_parent_genome

			var progress_callback = _set_generation_progress_from_thread.bind(generated_agent_number, attempt_index + 1)
			var brain = builder.build(genome, size, selected_num_inputs, selected_num_outputs, progress_callback, connections_density)

			if _is_generation_cancel_requested():
				return result

			if not brain.is_empty() and bool(brain.get("valid", false)):
				result["brains"][agent_index] = brain
				result["genomes"][agent_index] = genome.to_dictionary()
				accepted = true
				break

		if not accepted:
			result["error"] = (
				"Could not generate a valid renewed brain for agent %d after %d attempts."
				% [agent_index + 1, maximum_attempts]
			)
			return result

	return result


func _shuffle_indices(indices: Array[int], rng: RandomNumberGenerator) -> void:
	for index in range(indices.size() - 1, 0, -1):
		var swap_index: int = rng.randi_range(0, index)
		var temporary: int = indices[index]
		indices[index] = indices[swap_index]
		indices[swap_index] = temporary


func _set_generation_progress_from_thread(percent: int, agent_number: int, attempt_number: int) -> void:
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
		generation_thread != null
		or experiment_state == ExperimentState.GENERATING
		or experiment_state == ExperimentState.RUNNING
		or experiment_state == ExperimentState.PAUSED
		or experiment_state == ExperimentState.OFFSPRING
	)


func _request_reset() -> void:
	if generation_thread != null and generation_thread.is_alive():
		reset_pending = true
		generation_progress_mutex.lock()
		generation_cancel_requested = true
		generation_progress_mutex.unlock()
		summary_label.text = "Resetting simulation..."
		_refresh_buttons()
		return

	if generation_thread != null:
		generation_thread.wait_to_finish()
		generation_thread = null

	_complete_reset()


func _complete_reset() -> void:
	# Reset means recreating the entire scene, not clearing its state piecemeal.
	# Keep the generation cancellation/waiting logic in _request_reset()
	# so the scene is only destroyed after in-flight work has finished.
	reset_pending = true
	_refresh_buttons()
	get_tree().call_deferred("reload_current_scene")


func _start_experiment_from_population(new_generation: bool = true) -> void:
	if population_brains.size() != agent_count:
		experiment_state = ExperimentState.IDLE
		summary_label.text = "The population must contain exactly %d brains." % agent_count
		_refresh_buttons()
		return

	if new_generation:
		generation_number += 1
		target_test_index = 0
		if simulation_selector.selected == 4:
			_simulation_5_shuffle_target_sectors()
		elif simulation_selector.selected == 5:
			_simulation_6_shuffle_target_sectors()
		generation_real_start_us = Time.get_ticks_usec()
		generation_real_time_seconds = 0.0
		generation_fitness_sums = PackedFloat64Array()
		generation_fitness_sums.resize(agent_count)
		generation_mean_final_score_sum = 0.0
		generation_best_final_score = -1.0

		if simulation_selector.selected == 2:
			simulation_3_target_speed = agent_speed * 0.5

		if heatmap_tracking_enabled:
			_capture_generation_heatmap_phenotypes()

	fitness_reset_behavior.call()

	if new_generation or agent_dynamics.size() != agent_count:
		agent_dynamics.clear()
		agent_dynamics.resize(agent_count)
		for agent_index in agent_count:
			var dynamics = BrainDynamicsNative.new()
			dynamics.set_profiling_enabled(brain_profiling)
			dynamics.set_brain(population_brains[agent_index])
			agent_dynamics[agent_index] = dynamics
	else:
		for dynamics_variant in agent_dynamics:
			var dynamics: BrainDynamicsNative = dynamics_variant
			dynamics.set_profiling_enabled(brain_profiling)
			dynamics.reset()

	_invalidate_brain_preview()
	applied_brain_profiling = brain_profiling
	free_mode_active = false
	target_dragging = false
	free_mode_button.text = "Free Mode"
	_reset_brain_profiling_accumulators()
	_reset_experiment_entities()
	experiment_elapsed_seconds = 0.0
	summary_elapsed_seconds = 0.0
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
	agent.state["position"] = simulation_size * 0.5


func _simulation_1_update_agent_inputs(agent) -> void:
	var agent_position: Vector2 = agent.state["position"]
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
	# Target.reset() precedes agent.reset(), so the distance is well defined.
	var start_position: Vector2 = agent.state["position"]
	var initial_distance: float = start_position.distance_to(target.position)
	agent.state["simulation_4_initial_distance"] = initial_distance
	agent.state["simulation_4_nearest_distance"] = initial_distance
	agent.state["simulation_4_first_arrival"] = -1.0
	agent.state["simulation_4_proximity_integral"] = 0.0


func _simulation_4_update_agent_inputs(agent) -> void:
	_simulation_1_update_agent_inputs(agent)
	# Input 5 is touch only; it does not reveal obstacle direction or a route.
	agent.input_signals[4] = 1.0 if _simulation_4_touching_surface(agent.state["position"]) else 0.0


func _simulation_6_update_agent_inputs(agent) -> void:
	# Retain all five original Simulation 4 channels, unchanged.
	_simulation_4_update_agent_inputs(agent)
	_simulation_6_write_obstacle_proximity(agent.state["position"], agent)


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
	var agent_position: Vector2 = agent.state["position"]

	agent_position += Vector2(horizontal_drive, vertical_drive) * agent_speed * step_delta
	agent_position.x = clampf(agent_position.x, minimum_position.x, maximum_position.x)
	agent_position.y = clampf(agent_position.y, minimum_position.y, maximum_position.y)
	agent.state["position"] = agent_position


func _simulation_4_apply_agent_outputs(agent, step_delta: float) -> void:
	# Kinematic movement: no gravity, restitution, friction or agent-agent collisions.
	var displacement = Vector2(
		agent.output_signals[1] - agent.output_signals[0],
		agent.output_signals[3] - agent.output_signals[2]
	) * agent_speed * step_delta

	var previous_position: Vector2 = agent.state["position"]
	var agent_position: Vector2 = _simulation_4_move_with_collision(previous_position, displacement)
	var half_agent_size: float = agent_square_size * 0.5
	agent_position.x = clampf(agent_position.x, half_agent_size, simulation_size.x - half_agent_size)
	agent_position.y = clampf(agent_position.y, half_agent_size, simulation_size.y - half_agent_size)
	agent.state["position"] = agent_position
	# Fitness collection normally runs before motor outputs. Also check after
	# movement to capture arrival on the very last tick of a target test.
	if not free_mode_active:
		_simulation_4_record_progress(agent, experiment_elapsed_seconds + step_delta)


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


func _simulation_4_reset_target(selected_target) -> void:
	# The target is stationary during each test, like in Simulation 1.
	var half_target_size: float = target_square_size * 0.5
	var minimum: Vector2 = Vector2.ONE * half_target_size
	var maximum: Vector2 = simulation_size - minimum
	var center: Vector2 = simulation_size * 0.5
	var furthest_radius: float = center.distance_to(minimum)
	var minimum_radius: float = minf(simulation_4_min_target_distance, furthest_radius)
	var minimum_radius_squared: float = minimum_radius * minimum_radius

	var candidate: Vector2 = center
	var found: bool = false
	for _attempt in 256:
		candidate = Vector2(
			randf_range(minimum.x, maximum.x),
			randf_range(minimum.y, maximum.y)
		)
		if candidate.distance_squared_to(center) >= minimum_radius_squared:
			found = true
			break

	if not found:
		# Degenerate configurations with a radius nearly as large as the arena.
		candidate = Vector2(
			minimum.x if randf() < 0.5 else maximum.x,
			minimum.y if randf() < 0.5 else maximum.y
		)

	selected_target.position = candidate
	_simulation_4_update_obstacle(true)


func _update_simulation_5_num_tests_label() -> void:
	if not is_instance_valid(simulation_5_num_tests_label):
		return
	simulation_5_num_tests_label.visible = simulation_selector.selected == 4 or simulation_selector.selected == 5
	var difficulty: int = simulation_6_difficulty if simulation_selector.selected == 5 else simulation_5_difficulty
	simulation_5_num_tests_label.text = "Num Tests: %d" % difficulty


func _simulation_5_advance_difficulty_if_ready() -> void:
	# The generation just finished; its average includes all agents and
	# every target test of the CURRENT difficulty. The next difficulty is
	# applied only to the next generation, never in the middle of a test.
	if simulation_selector.selected != 4 or simulation_5_difficulty >= SIMULATION_5_ANGULAR_SECTOR_COUNT:
		return
	var required_mean: float = SIMULATION_5_DIFFICULTY_BASE_THRESHOLD + SIMULATION_5_DIFFICULTY_THRESHOLD_STEP * float(simulation_5_difficulty - 3)
	if float(_generation_fitness_statistics()["mean"]) <= required_mean:
		return
	simulation_5_difficulty += 1
	target_test_count = simulation_5_difficulty
	_update_simulation_5_num_tests_label()


func _simulation_6_advance_difficulty_if_ready() -> void:
	# Decide only after completing all tests of the current generation.
	if simulation_selector.selected != 5 or simulation_6_difficulty >= SIMULATION_5_ANGULAR_SECTOR_COUNT:
		return
	var required_mean: float = SIMULATION_5_DIFFICULTY_BASE_THRESHOLD + SIMULATION_5_DIFFICULTY_THRESHOLD_STEP * float(simulation_6_difficulty - 3)
	if float(_generation_fitness_statistics()["mean"]) <= required_mean:
		return
	simulation_6_difficulty += 1
	target_test_count = simulation_6_difficulty
	_update_simulation_5_num_tests_label()


func _simulation_6_shuffle_target_sectors() -> void:
	simulation_6_target_sector_order.clear()
	for sector_index in simulation_6_difficulty:
		simulation_6_target_sector_order.append(sector_index)
	simulation_6_target_sector_order.shuffle()


func _simulation_6_reset_target(selected_target) -> void:
	if simulation_6_target_sector_order.size() != simulation_6_difficulty:
		_simulation_6_shuffle_target_sectors()
	_place_target_in_curriculum_sector(
		selected_target,
		simulation_6_target_sector_order[target_test_index],
		simulation_6_difficulty
	)


func _simulation_5_shuffle_target_sectors() -> void:
	# At difficulty N, visit each of N angular sectors exactly once per
	# generation. N=3 is the initial curriculum stage; N=12 is the final stage.
	simulation_5_target_sector_order.clear()
	for sector_index in simulation_5_difficulty:
		simulation_5_target_sector_order.append(sector_index)
	simulation_5_target_sector_order.shuffle()


func _simulation_5_reset_target(selected_target) -> void:
	# Called once before the initial population exists, too.
	if simulation_5_target_sector_order.size() != simulation_5_difficulty:
		_simulation_5_shuffle_target_sectors()
	_place_target_in_curriculum_sector(
		selected_target,
		simulation_5_target_sector_order[target_test_index],
		simulation_5_difficulty
	)


func _place_target_in_curriculum_sector(selected_target, sector_index: int, difficulty: int) -> void:
	# One randomly placed target/obstacle in each sector for this generation.
	# Radial area-uniform sampling keeps the target inside the arena,
	# outside the same minimum-distance constraint as Simulation 4 when feasible.
	var half_target_size: float = target_square_size * 0.5
	var minimum: Vector2 = Vector2.ONE * half_target_size
	var maximum: Vector2 = simulation_size - minimum
	var center: Vector2 = simulation_size * 0.5

	var sector_width: float = TAU / float(difficulty)
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
	var agent_position: Vector2 = agent.state["position"]
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


func _simulation_4_record_progress(agent, elapsed_time: float) -> void:
	# Keep updating nearest distance even after the first contact.
	var position: Vector2 = agent.state["position"]
	var displacement: Vector2 = target.position - position
	var distance: float = displacement.length()
	agent.state["simulation_4_nearest_distance"] = minf(
		float(agent.state["simulation_4_nearest_distance"]), distance
	)

	# Both agents and target are rendered as axis-aligned squares. A goal is
	# reached when these squares first touch or overlap.
	var contact_half_size: float = (agent_square_size + target_square_size) * 0.5
	if float(agent.state["simulation_4_first_arrival"]) < 0.0 and absf(displacement.x) <= contact_half_size and absf(displacement.y) <= contact_half_size:
		agent.state["simulation_4_first_arrival"] = elapsed_time


# Fitness is calculated for evolutionary selection only, not fed to the brain.
# At each physics tick, integrate proximity throughout the target test.
# R is the existing target square size, measured center-to-center.
func _simulation_4_proximity_score(distance: float) -> float:
	var radius: float = maxf(target_square_size, 0.000001)
	if distance >= radius:
		return 0.0
	var relative_distance: float = distance / radius
	var shape: float = 1.0 - relative_distance * relative_distance
	return shape * shape


func _simulation_4_accumulate_fitness(step_delta: float) -> void:
	for agent in agents:
		_simulation_4_record_progress(agent, experiment_elapsed_seconds)
		var position: Vector2 = agent.state["position"]
		var distance: float = position.distance_to(target.position)
		agent.state["simulation_4_proximity_integral"] = float(
			agent.state["simulation_4_proximity_integral"]
		) + _simulation_4_proximity_score(distance) * step_delta


# F = w * best_normalized_progress + (1-w) * time_average(proximity).
# A momentary contact earns only its contribution to the time integral;
# sustained presence within target_square_size is what raises fitness.
func _simulation_4_agent_fitness(agent) -> float:
	var initial_distance: float = maxf(float(agent.state.get("simulation_4_initial_distance", 0.0)), 0.000001)
	var nearest_distance: float = float(agent.state.get("simulation_4_nearest_distance", initial_distance))
	var progress: float = clampf(1.0 - nearest_distance / initial_distance, 0.0, 1.0)
	var elapsed: float = maxf(experiment_elapsed_seconds, 0.000001)
	var proximity_integral: float = float(agent.state.get("simulation_4_proximity_integral", 0.0))
	var mean_proximity: float = clampf(proximity_integral / elapsed, 0.0, 1.0)
	return simulation_4_progress_weight * progress + (1.0 - simulation_4_progress_weight) * mean_proximity


func _simulation_4_fitness(agent_index: int) -> float:
	return _simulation_4_agent_fitness(agents[agent_index])


# Simulation 5: score is 1 inside the target's 12-pixel radius (by default),
# then decays exponentially with excess center-to-center distance.
func _simulation_5_proximity_score(distance: float) -> float:
	var radius: float = maxf(target_square_size, 0.000001)
	return exp(-maxf(distance - radius, 0.0) / 100.0)


# Only the last second of each target test contributes to fitness.
# Clamp the integrated portion of each physics tick to the final-time window,
# so the result does not depend on a tick landing exactly at its beginning.
func _simulation_5_accumulate_fitness(step_delta: float) -> void:
	var window_start: float = maxf(experiment_duration_seconds - 1.0, 0.0)
	var window_delta: float = minf(experiment_elapsed_seconds + step_delta, experiment_duration_seconds) - maxf(experiment_elapsed_seconds, window_start)
	if window_delta <= 0.0:
		return

	for agent_index in agent_count:
		var position: Vector2 = agents[agent_index].state["position"]
		fitness_integrals[agent_index] += _simulation_5_proximity_score(position.distance_to(target.position)) * window_delta


func _simulation_5_fitness(agent_index: int) -> float:
	var window_length: float = minf(experiment_duration_seconds, 1.0)
	if window_length <= 0.0:
		return 0.0
	return clampf(fitness_integrals[agent_index] / window_length, 0.0, 1.0)


func _simulation_5_evaluate_agent(agent) -> Vector2:
	var position: Vector2 = agent.state["position"]
	var normalized_distance: float = position.distance_to(target.position) / simulation_size.length()
	# The score component is instantaneous for diagnostics; accumulated fitness
	# still uses only the last second of the test.
	return Vector2(_simulation_5_proximity_score(position.distance_to(target.position)), normalized_distance)


func _simulation_4_evaluate_agent(agent) -> Vector2:
	var position: Vector2 = agent.state["position"]
	var normalized_distance: float = position.distance_to(target.position) / simulation_size.length()
	return Vector2(_simulation_4_agent_fitness(agent), normalized_distance)


func _simulation_4_agent_info(agent_index: int) -> Dictionary:
	var agent = agents[agent_index]
	var info: Dictionary = _simulation_1_agent_info(agent_index)
	var main_text: String = str(info["main"])
	main_text = main_text.replace(
		"\nOutput 1 (move left):",
		"\nInput 5 (contact): %.2f\nOutput 1 (move left):" % agent.input_signals[4]
	)
	var first_arrival: float = float(agent.state.get("simulation_4_first_arrival", -1.0))
	main_text += "\nReached target: %s" % ("Yes" if first_arrival >= 0.0 else "No")
	if first_arrival >= 0.0:
		main_text += "\nFirst arrival: %.2f s" % first_arrival
	main_text += "\nFitness: %.3f" % float(fitness_behavior.call(agent_index))
	info["main"] = main_text
	return info


func _simulation_6_agent_info(agent_index: int) -> Dictionary:
	var info: Dictionary = _simulation_4_agent_info(agent_index)
	if info.is_empty():
		return info
	var inputs: PackedFloat64Array = agents[agent_index].input_signals
	if inputs.size() < 13:
		return info
	var directions: Array[String] = ["W", "E", "N", "S", "NW", "NE", "SW", "SE"]
	var extra: String = ""
	for sensor_index in 8:
		extra += "\nInput %d (wall %s): %.2f" % [6 + sensor_index, directions[sensor_index], float(inputs[5 + sensor_index])]
	info["main"] = str(info["main"]).replace("\nOutput 1 (move left):", extra + "\nOutput 1 (move left):")
	return info


func _simulation_1_agent_info(agent_index: int) -> Dictionary:
	var agent = agents[agent_index]
	var dynamics: BrainDynamicsNative = agent_dynamics[agent_index]
	var state_statistics: Dictionary = dynamics.get_neuron_statistics("state")
	var signal_statistics: Dictionary = dynamics.get_neuron_statistics("signal")
	var input_influence: float = float(population_brains[agent_index].get("input_gain", 0.0))

	var main_text: String = (
		"AGENT %d\n"
		+ "Input 1 (left): %.2f\n"
		+ "Input 2 (right): %.2f\n"
		+ "Input 3 (up): %.2f\n"
		+ "Input 4 (down): %.2f\n"
		+ "Output 1 (move left): %.2f\n"
		+ "Output 2 (move right): %.2f\n"
		+ "Output 3 (move up): %.2f\n"
		+ "Output 4 (move down): %.2f"
	) % [
		agent_index + 1,
		agent.input_signals[0],
		agent.input_signals[1],
		agent.input_signals[2],
		agent.input_signals[3],
		dynamics.get_output_signal(0),
		dynamics.get_output_signal(1),
		dynamics.get_output_signal(2),
		dynamics.get_output_signal(3),
	]

	var state_text: String = (
		"\nMin state: %.2f\n"
		+ "Max state: %.2f\n"
		+ "Mean state: %.2f\n"
		+ "Std state: %.2f\n"
		+ "25%% state: %.2f\n"
		+ "Median state: %.2f\n"
		+ "75%% state: %.2f\n"
		+ "Neuron count: %d\n"
		+ "Input influence: %.2f"
	) % [
		state_statistics["min"],
		state_statistics["max"],
		state_statistics["mean"],
		state_statistics["std"],
		state_statistics["25%"],
		state_statistics["50%"],
		state_statistics["75%"],
		state_statistics["count"],
		input_influence,
	]

	var signal_text: String = (
		"\nMin signal: %.2f\n"
		+ "Max signal: %.2f\n"
		+ "Mean signal: %.2f\n"
		+ "Std signal: %.2f\n"
		+ "25%% signal: %.2f\n"
		+ "Median signal: %.2f\n"
		+ "75%% signal: %.2f"
	) % [
		signal_statistics["min"],
		signal_statistics["max"],
		signal_statistics["mean"],
		signal_statistics["std"],
		signal_statistics["25%"],
		signal_statistics["50%"],
		signal_statistics["75%"],
	]

	return {
		"main": main_text,
		"state": state_text,
		"signal": signal_text,
	}


func _clear_runtime_state() -> void:
	_invalidate_brain_preview()
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
	_reset_brain_profiling_accumulators()
	queue_redraw()


func _queue_inputs_and_accumulate_fitness(step_delta: float, accumulate_fitness: bool = true) -> void:
	for agent_index in agent_count:
		var agent = agents[agent_index]
		var dynamics: BrainDynamicsNative = agent_dynamics[agent_index]

		agent.update_inputs()
		for input_index in num_inputs:
			dynamics.add_input_signal(input_index, agent.input_signals[input_index])

		if accumulate_fitness:
			var evaluation: Vector2 = evaluation_behavior.call(agent)
			best_score_ever = maxf(best_score_ever, evaluation.x)
			best_distance_ever = minf(best_distance_ever, evaluation.y)

	if accumulate_fitness:
		fitness_accumulation_behavior.call(step_delta)


func _advance_all_brains(step_delta: float) -> void:
	if brain_profiling != applied_brain_profiling:
		for dynamics_variant in agent_dynamics:
			var dynamics: BrainDynamicsNative = dynamics_variant
			dynamics.set_profiling_enabled(brain_profiling)
		applied_brain_profiling = brain_profiling

	var batch_count: int = _get_brain_batch_count()
	var total_start_us: int = Time.get_ticks_usec() if brain_profiling else 0
	var group_task_id: int = WorkerThreadPool.add_group_task(
		_advance_brain_batch.bind(step_delta, batch_count),
		batch_count,
		batch_count,
		true,
		"Advance brain batches"
	)
	WorkerThreadPool.wait_for_group_task_completion(group_task_id)

	if not brain_profiling:
		return

	var batch_total_us: int = Time.get_ticks_usec() - total_start_us
	profiling_steps_accumulated += 1
	profiling_total_us_accumulated += batch_total_us

	for dynamics_variant in agent_dynamics:
		var dynamics: BrainDynamicsNative = dynamics_variant
		profiling_brain_cpu_us_accumulated += dynamics.get_profile_total_us()
		profiling_prepare_and_firing_us_accumulated += dynamics.get_profile_prepare_and_firing_us()
		profiling_modulatory_field_us_accumulated += dynamics.get_profile_modulatory_field_us()
		profiling_connections_us_accumulated += dynamics.get_profile_connections_us()
		profiling_homeostatic_us_accumulated += dynamics.get_profile_homeostatic_us()
		profiling_propagation_statistics_us_accumulated += dynamics.get_profile_propagation_statistics_us()
		profiling_state_update_us_accumulated += dynamics.get_profile_state_update_us()
		profiling_structural_plasticity_us_accumulated += dynamics.get_profile_structural_plasticity_us()

	if profiling_steps_accumulated >= profiling_print_interval_steps:
		_print_brain_profiling(batch_count)
		_reset_brain_profiling_accumulators()


## Returns a small number of coarse worker batches instead of one task per brain.
func _get_brain_batch_count() -> int:
	var processor_count: int = maxi(1, OS.get_processor_count())
	return mini(agent_dynamics.size(), mini(BRAIN_WORKER_BATCHES, processor_count))


## Advances one contiguous range of independent brains on a WorkerThreadPool worker.
func _advance_brain_batch(batch_index: int, step_delta: float, batch_count: int) -> void:
	var population_size: int = agent_dynamics.size()
	var start_index: int = floori(float(batch_index * population_size) / float(batch_count))
	var end_index: int = floori(float((batch_index + 1) * population_size) / float(batch_count))

	for agent_index in range(start_index, end_index):
		var dynamics: BrainDynamicsNative = agent_dynamics[agent_index]
		dynamics.advance(step_delta)


func _reset_brain_profiling_accumulators() -> void:
	profiling_steps_accumulated = 0
	profiling_total_us_accumulated = 0
	profiling_brain_cpu_us_accumulated = 0
	profiling_prepare_and_firing_us_accumulated = 0
	profiling_modulatory_field_us_accumulated = 0
	profiling_connections_us_accumulated = 0
	profiling_homeostatic_us_accumulated = 0
	profiling_propagation_statistics_us_accumulated = 0
	profiling_state_update_us_accumulated = 0
	profiling_structural_plasticity_us_accumulated = 0


func _print_brain_profiling(batch_count: int) -> void:
	if profiling_steps_accumulated <= 0:
		return

	var divisor: float = 1000.0 * float(profiling_steps_accumulated)
	var total_ms: float = float(profiling_total_us_accumulated) / divisor
	var brain_cpu_ms: float = float(profiling_brain_cpu_us_accumulated) / divisor
	var prepare_and_firing_ms: float = float(profiling_prepare_and_firing_us_accumulated) / divisor
	var modulatory_field_ms: float = float(profiling_modulatory_field_us_accumulated) / divisor
	var connections_ms: float = float(profiling_connections_us_accumulated) / divisor
	var homeostatic_ms: float = float(profiling_homeostatic_us_accumulated) / divisor
	var propagation_statistics_ms: float = float(profiling_propagation_statistics_us_accumulated) / divisor
	var state_update_ms: float = float(profiling_state_update_us_accumulated) / divisor
	var structural_plasticity_ms: float = float(profiling_structural_plasticity_us_accumulated) / divisor
	var measured_stages_ms: float = (
		prepare_and_firing_ms
		+ modulatory_field_ms
		+ connections_ms
		+ homeostatic_ms
		+ propagation_statistics_ms
		+ state_update_ms
		+ structural_plasticity_ms
	)
	var brain_overhead_ms: float = maxf(0.0, brain_cpu_ms - measured_stages_ms)
	var parallel_speedup: float = 0.0 if total_ms <= 0.0 else brain_cpu_ms / total_ms

	var profile_text = (
		"Brain profile — %d brains in %d worker batches, average over %d simulation steps:\n"
		+ "  Total batch wall time:    %.3f ms\n"
		+ "  Summed brain CPU time:    %.3f ms\n"
		+ "  Parallel speedup:         %.3fx\n"
		+ "  Prepare + firing:         %.3f ms\n"
		+ "  Modulatory field:         %.3f ms\n"
		+ "  Connections + plasticity: %.3f ms\n"
		+ "  Homeostatic reconnection: %.3f ms\n"
		+ "  Propagation statistics:   %.3f ms\n"
		+ "  Neuron state update:      %.3f ms\n"
		+ "  Structural plasticity:    %.3f ms\n"
		+ "  Brain profiling overhead: %.3f ms"
	) % [
		agent_dynamics.size(),
		batch_count,
		profiling_steps_accumulated,
		total_ms,
		brain_cpu_ms,
		parallel_speedup,
		prepare_and_firing_ms,
		modulatory_field_ms,
		connections_ms,
		homeostatic_ms,
		propagation_statistics_ms,
		state_update_ms,
		structural_plasticity_ms,
		brain_overhead_ms,
	]
	print(profile_text)


func _move_agents(step_delta: float) -> void:
	for agent_index in agent_count:
		var agent = agents[agent_index]
		var dynamics: BrainDynamicsNative = agent_dynamics[agent_index]

		for output_index in num_outputs:
			agent.output_signals[output_index] = dynamics.get_output_signal(output_index)

		agent.apply_outputs(step_delta)


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
	evolution_graph_mean_points_cache = PackedVector2Array()
	evolution_graph_best_points_cache = PackedVector2Array()
	genetic_diversity_graph_points_cache = PackedVector2Array()
	genetic_diversity_excluded_agent_indices.clear()
	evolution_graph_minimum_value_cache = 0.0
	evolution_graph_maximum_value_cache = 1.0
	genetic_diversity_minimum_exponent_cache = -1
	_queue_history_graph_redraw()
	target_test_index = 0
	simulation_5_difficulty = 3
	simulation_6_difficulty = 3
	if simulation_selector.selected == 4:
		target_test_count = simulation_5_difficulty
	elif simulation_selector.selected == 5:
		target_test_count = simulation_6_difficulty
	simulation_5_target_sector_order.clear()
	simulation_6_target_sector_order.clear()
	_update_simulation_5_num_tests_label()
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
		print("Generation %d real time: %.3f s" % [generation_number, generation_real_time_seconds])

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
	_simulation_5_advance_difficulty_if_ready()
	_simulation_6_advance_difficulty_if_ready()
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
		for agent in agents:
			if float(agent.state.get("simulation_4_first_arrival", -1.0)) >= 0.0:
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
		+ "Neural iterations/s: %.1f\n" % effective_steps_per_second
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
	var generating = experiment_state == ExperimentState.GENERATING or generation_thread != null
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
	simulation_selector.disabled = simulation_active
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
		"format_version": 3,
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
	# Mutation can be edited while running; always persist the current valid field.
	if not _try_update_mutation_probability(mutation_probability_field.text):
		summary_label.text = "Cannot save: Mutation Probability must be a finite number between 0 and 1."
		return
	var save_path = path
	if save_path.get_extension().to_lower() != SAVE_EXTENSION:
		save_path += "." + SAVE_EXTENSION

	var data = {
		"format_version": SAVE_FORMAT_VERSION,
		"external_parameters": {
			"size": current_brain_size,
			"num_inputs": num_inputs,
			"num_outputs": num_outputs,
			"agent_count": agent_count,
			"connections_density": connections_density,
			"generation_time_seconds": experiment_duration_seconds,
		},
		"experiment_parameters": {
			"agent_speed": agent_speed,
			"simulation_size": simulation_size,
			"agent_square_size": agent_square_size,
			"target_square_size": target_square_size,
			"mutation_probability": mutation_probability,
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
			"genetic_diversity_excluded_agent_indices": genetic_diversity_excluded_agent_indices.duplicate(),
			"previous_generation_elite_indices": previous_generation_elite_indices.duplicate(),
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
	print("Simulation saved to: ", save_path)


func _current_agent_positions() -> Array:
	var positions: Array = []
	positions.resize(agents.size())

	for agent_index in agents.size():
		positions[agent_index] = agents[agent_index].state.get("position", Vector2.ZERO)

	return positions


func _rebuild_population_from_genomes(
		genomes: Array,
		size: Vector3i,
		selected_num_inputs: int,
		selected_num_outputs: int
	) -> Dictionary:
	var result = {
		"brains": [],
		"error": "",
	}
	var builder: BrainBuilderNative = BrainBuilderNative.new()
	var offspring: OffspringNative = OffspringNative.new()

	for agent_index in genomes.size():
		var genome_variant = genomes[agent_index]
		if not (genome_variant is Dictionary):
			result["error"] = "Saved genome %d is not a Dictionary." % (agent_index + 1)
			return result

		var genome: GenomeNative = offspring.genome_from_dictionary(genome_variant)
		if genome == null:
			result["error"] = "Could not reconstruct saved genome %d." % (agent_index + 1)
			return result

		var brain = builder.build(genome, size, selected_num_inputs, selected_num_outputs, Callable(), connections_density)
		if brain.is_empty() or not bool(brain.get("valid", false)):
			result["error"] = "Saved genome %d generated an invalid brain." % (agent_index + 1)
			return result

		result["brains"].append(brain)

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
	# Legacy binary saves imply eta=1.0, and their genome count is authoritative.
	connections_density = float(external.get("connections_density", 1.0))
	var saved_agent_count: int = data["genomes"].size()

	_clear_runtime_state()
	agent_count = saved_agent_count
	num_agents_spinbox.set_value_no_signal(float(agent_count))
	_setup_simulation(simulation_selector.selected)
	_reset_evolution_statistics()
	if simulation_selector.selected == 4:
		# Optional for compatibility with saves created before this curriculum.
		simulation_5_difficulty = clampi(int(evolution.get("simulation_5_difficulty", 3)), 3, SIMULATION_5_ANGULAR_SECTOR_COUNT)
		target_test_count = simulation_5_difficulty
		_update_simulation_5_num_tests_label()
	elif simulation_selector.selected == 5:
		simulation_6_difficulty = clampi(int(evolution.get("simulation_6_difficulty", 3)), 3, SIMULATION_5_ANGULAR_SECTOR_COUNT)
		target_test_count = simulation_6_difficulty
		_update_simulation_5_num_tests_label()

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
		var saved_genetic_diversity_excluded_indices = evolution.get("genetic_diversity_excluded_agent_indices", [])
		var saved_elite_indices = evolution.get("previous_generation_elite_indices", [])
		if saved_elite_indices is Array:
			for elite_index_variant in saved_elite_indices:
				var elite_index: int = int(elite_index_variant)
				if elite_index >= 0 and elite_index < agent_count and elite_index not in previous_generation_elite_indices:
					previous_generation_elite_indices.append(elite_index)
		if saved_mean_fitness_history is PackedFloat64Array:
			evolution_mean_fitness_history = saved_mean_fitness_history.duplicate()
		if saved_best_fitness_history is PackedFloat64Array:
			evolution_best_fitness_history = saved_best_fitness_history.duplicate()
		if saved_genetic_diversity_history is PackedFloat64Array:
			genetic_diversity_history = saved_genetic_diversity_history.duplicate()
		genetic_diversity_excluded_agent_indices.clear()
		if saved_genetic_diversity_excluded_indices is Array:
			for index_variant in saved_genetic_diversity_excluded_indices:
				genetic_diversity_excluded_agent_indices.append(int(index_variant))

	population_genomes = data["genomes"].duplicate(true)
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
	if format_version != SAVE_FORMAT_VERSION:
		return "Unsupported save format version."

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
