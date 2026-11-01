extends Node

# Runs automatically when validation.tscn starts. Both projects use the same
# res://validation_genome.json and parameters. No visualisation or FPS coupling.
const SIZE := Vector3i(10, 10, 6)
const INPUT_COUNT := 4
const OUTPUT_COUNT := 4
const ITERATIONS := 1000
const DT := 1.0 / 60.0
const GENOME_PATH := "res://validation_genome.json"
const OUTPUT_NAME := "validation_cpp.jsonl"
const STATE_FIELDS := [
	"normalized_signals", "neuron_states_used", "incoming_signals",
	"fatigues", "fatigues_used", "activity_traces",
	"effective_thresholds_used", "modulatory_releases",
	"modulatory_fields", "effective_modulations", "firing_states",
	"incoming_connection_counts"
]
const SUMMARY_FIELDS := [
	"iteration_count", "elapsed_time_seconds", "total_connections",
	"total_connections_destroyed", "total_structural_plasticity_connections_formed",
	"total_homeostatic_connections_formed", "added_connections_last_iteration",
	"mean_signal", "mean_weight", "mean_modulatory_release",
	"mean_modulatory_field", "mean_effective_modulation",
	"mean_eligibility_trace"
]

var _worker: Thread
var _guard := Mutex.new()
var _status := "Preparing validation..."
@onready var status_label: Label = $Status

func _ready() -> void:
	_worker = Thread.new()
	var err := _worker.start(_run)
	if err != OK:
		status_label.text = "Could not start validation worker: " + error_string(err)

func _process(_delta: float) -> void:
	_guard.lock()
	var message: String = _status
	_guard.unlock()
	if status_label.text != message:
		status_label.text = message

func _exit_tree() -> void:
	if _worker != null and _worker.is_started():
		_worker.wait_to_finish()

func _report(message: String) -> void:
	_guard.lock()
	_status = message
	_guard.unlock()
	print("[CPP Validation] ", message)

func _load_genes() -> Dictionary:
	if not FileAccess.file_exists(GENOME_PATH):
		return {}
	var data: Variant = JSON.parse_string(FileAccess.get_file_as_string(GENOME_PATH))
	if not (data is Dictionary) or not data.has("genome"):
		return {}
	var result: Dictionary = {}
	for name in data["genome"]:
		var digits: Array[int] = []
		for ch in String(data["genome"][name]):
			if not "0123".contains(ch):
				return {}
			digits.append(int(ch))
		result[name] = digits
	return result

func _open_output() -> FileAccess:
	var output := FileAccess.open("res://" + OUTPUT_NAME, FileAccess.WRITE)
	if output == null:
		output = FileAccess.open("user://" + OUTPUT_NAME, FileAccess.WRITE)
	return output

func _spatial(neuron: Dictionary) -> int:
	var pos: Vector3i = neuron["position"]
	return pos.x + SIZE.x * (pos.y + SIZE.y * pos.z)

func _metadata(brain: Dictionary, genome: Dictionary) -> Dictionary:
	var neuron_info: Array = []
	for neuron in brain["neurons"]:
		neuron_info.append([
			_spatial(neuron), int(bool(neuron["input"])), int(bool(neuron["output"])),
			float(neuron["activation_threshold"]), float(neuron["decay_factor"]),
			float(neuron["retention_factor"]), int(neuron["polarity_factor"]),
			float(neuron["hebbian_plasticity_rate"]),
			float(neuron["modulatory_release_factor"]),
			float(neuron["modulatory_sensitivity"]),
			float(neuron["connections_value"])
		])
	neuron_info.sort_custom(func(a, b): return int(a[0]) < int(b[0]))
	var genome_strings := {}
	for name in genome:
		var digits := ""
		for digit in genome[name]:
			digits += str(digit)
		genome_strings[name] = digits
	return {
		"record": "meta", "schema": 1, "backend": "CPP", "size": [SIZE.x, SIZE.y, SIZE.z],
		"num_inputs": INPUT_COUNT, "num_outputs": OUTPUT_COUNT,
		"steps": ITERATIONS, "dt": DT, "input_step": 1,
		"input_value": 1.0, "genome": genome_strings,
		"neuron_static_columns": ["spatial", "input", "output", "threshold", "decay", "retention", "polarity", "hebbian", "release_factor", "sensitivity", "energy"],
		"neuron_static": neuron_info,
		"neuron_dynamic_columns": ["spatial"] + STATE_FIELDS,
		"edge_columns": ["source_spatial", "target_spatial", "weight", "eligibility_trace"],
		"brain_parameters": {
			"beta": brain["beta"], "max_connections": brain["max_connections"],
			"lambda": brain["lambda"], "nu": brain["nu"], "mu": brain["mu"],
			"structural_plasticity": brain["structural_plasticity"],
			"input_gain": brain["input_gain"]
		}
	}

func _neurons_from_state(state: Dictionary, brain: Dictionary) -> Array:
	var result: Array = []
	var original: Array = brain["neurons"]
	for rank in original.size():
		var row: Array = [_spatial(original[rank])]
		for key in STATE_FIELDS:
			var values: Variant = state.get(key, [])
			row.append(values[rank] if rank < values.size() else null)
		result.append(row)
	result.sort_custom(func(a, b): return int(a[0]) < int(b[0]))
	return result

func _edges_from_cpp(snapshot: Dictionary) -> Array:
	var output: Array = []
	for neuron in snapshot["neurons"]:
		var source: int = _spatial(neuron)
		for edge in neuron["connections"]:
			var target_pos: Vector3i = edge["target"]
			var target: int = target_pos.x + SIZE.x * (target_pos.y + SIZE.y * target_pos.z)
			output.append([source, target, float(edge["weight"]), float(edge.get("eligibility_trace", 0.0))])
	output.sort_custom(func(a, b): return int(a[0]) < int(b[0]) if a[0] != b[0] else int(a[1]) < int(b[1]))
	return output

func _frame(iteration: int, state: Dictionary, brain: Dictionary, edges: Array) -> Dictionary:
	var summary := {}
	for key in SUMMARY_FIELDS:
		summary[key] = state.get(key)
	return {
		"record": "frame", "iteration": iteration,
		"stimulus": 1.0 if iteration == 1 else 0.0,
		"summary": summary,
		"inputs_used": Array(state.get("input_signals_used", [])),
		"outputs": Array(state.get("output_signals", [])),
		"neurons": _neurons_from_state(state, brain),
		"edges": edges
	}

func _run() -> void:
	var genes: Dictionary = _load_genes()
	if genes.is_empty():
		_report("ERROR: cannot load fixed validation genome: " + GENOME_PATH)
		return
	# GenomeNative generates random genes by default. Replace ALL of them with
	# the same explicit digits used by the GPU build, via the existing importer.
	var genome: GenomeNative = OffspringNative.new().genome_from_dictionary(genes)
	if genome == null:
		_report("ERROR: cannot import genome into GenomeNative")
		return
	_report("CPP: constructing brain...")
	var builder := BrainBuilderNative.new()
	var brain: Dictionary = builder.build(genome, SIZE, INPUT_COUNT, OUTPUT_COUNT)
	if not bool(brain.get("valid", false)):
		_report("ERROR: invalid C++ brain; see Godot Output")
		return
	var dyn := BrainDynamicsNative.new()
	dyn.set_brain(brain)
	var file: FileAccess = _open_output()
	if file == null:
		_report("ERROR: cannot open output file")
		return
	file.store_line(JSON.stringify(_metadata(brain, genes)))
	file.store_line(JSON.stringify(_frame(0, dyn.get_viewer_state(), brain, _edges_from_cpp(dyn.get_brain_snapshot()))))
	for t in range(1, ITERATIONS + 1):
		if t == 1:
			for input_id in INPUT_COUNT:
				dyn.add_input_signal(input_id, 1.0)
		dyn.advance(DT)
		file.store_line(JSON.stringify(_frame(t, dyn.get_viewer_state(), brain, _edges_from_cpp(dyn.get_brain_snapshot()))))
		if t % 10 == 0:
			_report("CPP: %d / %d iterations" % [t, ITERATIONS])
	file.flush()
	var path: String = ProjectSettings.globalize_path(file.get_path_absolute())
	file.close()
	_report("COMPLETE: %d frames saved to %s" % [ITERATIONS + 1, path])
