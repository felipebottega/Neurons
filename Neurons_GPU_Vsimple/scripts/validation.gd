extends Node

# Runs automatically when validation.tscn starts. Both projects use the same
# res://validation_genome.json and parameters. No visualisation or FPS coupling.
const SIZE := Vector3i(10, 10, 6)
const INPUT_COUNT := 4
const OUTPUT_COUNT := 4
const ITERATIONS := 1000
const DT := 1.0 / 60.0
const GENOME_PATH := "res://validation_genome.json"
const OUTPUT_NAME := "validation_gpu.jsonl"
const STATE_FIELDS := [
	"normalized_signals", "neuron_states_used", "incoming_signals",
	"incoming_connection_counts"
]
const SUMMARY_FIELDS := [
	"iteration_count", "elapsed_time_seconds", "total_connections",
	"mean_signal", "mean_weight"
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
	print("[GPU Validation] ", message)

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
			int(neuron["polarity_factor"]), float(neuron["connections_value"])
		])
	neuron_info.sort_custom(func(a, b): return int(a[0]) < int(b[0]))
	var genome_strings := {}
	for name in Offspring.GENE_NAMES:
		var digits := ""
		for digit in genome[String(name)]:
			digits += str(digit)
		genome_strings[String(name)] = digits
	return {
		"record": "meta", "schema": 2, "backend": "GPU_Vsimple",
		"size": [SIZE.x, SIZE.y, SIZE.z],
		"num_inputs": INPUT_COUNT, "num_outputs": OUTPUT_COUNT,
		"steps": ITERATIONS, "dt": DT, "input_step": 1,
		"input_value": 1.0, "genome": genome_strings,
		"neuron_static_columns": ["spatial", "input", "output", "polarity", "connection_energy"],
		"neuron_static": neuron_info,
		"neuron_dynamic_columns": ["spatial"] + STATE_FIELDS,
		"edge_columns": ["source_spatial", "target_spatial", "weight", "reserved"],
		"brain_parameters": {
			"beta": brain["beta"], "max_connections": brain["max_connections"],
			"input_gain": brain["input_gain"], "exponential_factor": brain["exponential_factor"]
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

func _edges_from_gpu(dyn: BrainViewerGPU) -> Array:
	# Validation deliberately reads all per-slice connections for numerical
	# comparison with C++. Normal Viewer dynamics never performs this export.
	return dyn.read_synapses()

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
	var genome: Dictionary = _load_genes()
	if genome.is_empty():
		_report("ERROR: cannot load fixed validation genome: " + GENOME_PATH)
		return
	_report("GPU: constructing brain...")
	# RenderingDevice and all its API calls stay on THIS thread throughout.
	var rd: RenderingDevice = RenderingServer.create_local_rendering_device()
	if rd == null:
		_report("ERROR: cannot create a local RenderingDevice")
		return
	var builder := BrainBuilderGPU.new()
	var brain: Dictionary = builder.build(genome, SIZE, INPUT_COUNT, OUTPUT_COUNT, Callable(), true, rd, 1.0, BrainBuilderGPU.STATS_WEIGHT_SUM)
	if not bool(brain.get("valid", false)) or not brain.has("gpu_state"):
		_report("ERROR: invalid brain; see Godot Output")
		return
	var dyn := BrainViewerGPU.new()
	if not dyn.initialize(brain):
		_report("ERROR: GPU dynamics initialization failed")
		if dyn.rd != null:
			dyn.clear()
		return
	var file: FileAccess = _open_output()
	if file == null:
		_report("ERROR: cannot open output file")
		dyn.clear()
		return
	file.store_line(JSON.stringify(_metadata(brain, genome)))
	file.store_line(JSON.stringify(_frame(0, dyn.get_viewer_state(), brain, _edges_from_gpu(dyn))))
	for t in range(1, ITERATIONS + 1):
		if t == 1:
			for input_id in INPUT_COUNT:
				dyn.add_input_signal(input_id, 1.0)
		if dyn.advance(DT, true) < 0:
			file.close()
			_report("ERROR: GPU dynamics failed at iteration %d" % t)
			dyn.clear()
			return
		file.store_line(JSON.stringify(_frame(t, dyn.get_viewer_state(), brain, _edges_from_gpu(dyn))))
		if t % 10 == 0:
			_report("GPU: %d / %d iterations" % [t, ITERATIONS])
	file.flush()
	var path: String = ProjectSettings.globalize_path(file.get_path_absolute())
	file.close()
	dyn.clear()
	_report("COMPLETE: %d frames saved to %s" % [ITERATIONS + 1, path])
