class_name BrainViewerGPU
extends RefCounted

# Presentation-only adapter. Neural dynamics are executed exclusively by the
# same BrainPopulationGPU shaders as the multi-agent simulation. In particular,
# none of the legacy brain_viewer_* dynamics shaders are dispatched here.
const MAX_LINE_SEGMENTS: int = 180000

var rd: RenderingDevice
var _core: BrainPopulationGPU
var _brain: Dictionary = {}
var _meta: BrainDynamicsGPU
var _total: int = 0
var _pending: PackedFloat32Array = PackedFloat32Array()
var _statistics: Dictionary = {}
var _iteration: int = 0
var _run_time: float = 0.0
var _lost_edges: int = 0
var _reconnected: int = 0
var _structural_created: int = 0
var _new_edges_since_snapshot: int = 0

func initialize(brain: Dictionary) -> bool:
	if rd != null or _core != null:
		push_error("BrainViewerGPU: clear the current brain before initializing")
		return false
	if not brain.has("gpu_state") or not bool(brain.get("valid", false)):
		push_error("BrainViewerGPU: GPU phenotype required")
		return false
	_brain = brain
	_core = BrainPopulationGPU.new()
	if not _core.initialize([brain], true):
		rd = _core.rd
		push_error("BrainViewerGPU: " + _core.last_error)
		return false
	rd = _core.rd
	_meta = _core.dynamics[0]
	_total = brain["size"].x * brain["size"].y * brain["size"].z
	_pending.resize(_core.input_count)
	_pending.fill(0.0)
	return capture_visual_state()

func add_input_signal(input_slot: int, value: float) -> void:
	if input_slot >= 0 and input_slot < _pending.size():
		_pending[input_slot] = value

func advance(delta_seconds: float, capture_snapshot: bool = true) -> int:
	if _core == null or rd == null:
		return -1
	if not _core.advance(delta_seconds, _pending):
		push_error("BrainViewerGPU: " + _core.last_error)
		return -1
	_pending.fill(0.0)
	_iteration += 1
	_run_time += delta_seconds
	var changes: PackedInt32Array = _core.inspection_changes
	_lost_edges += changes[0]
	_reconnected += changes[1]
	_structural_created += changes[2]
	_new_edges_since_snapshot += changes[1] + changes[2]
	if capture_snapshot and not capture_visual_state():
		return -1
	return changes[0]

func capture_visual_state() -> bool:
	if _core == null or rd == null:
		return false
	var snap: Dictionary = _core.inspection_snapshot(0)
	if snap.is_empty():
		return false
	var raw: PackedFloat32Array = snap["states"]
	var incoming: PackedInt32Array = snap["incoming_degrees"]
	var living: PackedInt32Array = snap["live_degrees"]
	var weights: PackedFloat32Array = snap["synapse_stats"]
	if raw.size() != 12 * _total or incoming.size() != _total or living.size() != _total or weights.size() != 2 * _total:
		push_error("BrainViewerGPU: incomplete Vsimple neural inspection")
		return false
	var normalized := PackedFloat32Array()
	var states := PackedFloat32Array()
	var next_incoming := PackedFloat32Array()
	var incoming_counts := PackedFloat32Array()
	var outgoing_counts := PackedFloat32Array()
	normalized.resize(_total)
	states.resize(_total)
	next_incoming.resize(_total)
	incoming_counts.resize(_total)
	outgoing_counts.resize(_total)
	var signal_sum: float = 0.0
	var weight_sum: float = 0.0
	var live_edges: int = 0
	var internal_out_sum: int = 0
	var outgoing_n: int = 0
	var min_degree: int = 2147483647
	var max_degree: int = 0
	for rank in _total:
		var spatial: int = _meta._rank_to_spatial[rank]
		var at: int = 12 * spatial
		var degree: int = living[spatial]
		normalized[rank] = raw[at + 3]
		states[rank] = raw[at + 4]
		next_incoming[rank] = raw[at + 7]
		incoming_counts[rank] = float(incoming[spatial])
		outgoing_counts[rank] = float(degree)
		signal_sum += normalized[rank]
		live_edges += degree
		weight_sum += weights[2 * spatial]
		if _meta._rank_is_output[rank] == 0:
			outgoing_n += 1
			internal_out_sum += degree
			min_degree = mini(min_degree, degree)
			max_degree = maxi(max_degree, degree)
	var outputs := PackedFloat64Array()
	outputs.resize(_meta._output_indices.size())
	var input_signals_used := PackedFloat32Array()
	input_signals_used.resize(_meta._input_indices.size())
	for i in _meta._output_indices.size():
		var rank: int = _meta._output_indices[i]
		outputs[i] = normalized[rank]
	for i in _meta._input_indices.size():
		input_signals_used[i] = states[_meta._input_indices[i]]
	_statistics = {
		"normalized_signals": normalized,
		"neuron_states_used": states,
		"incoming_signals": next_incoming,
		"incoming_connection_counts": incoming_counts,
		"outgoing_connection_counts": outgoing_counts,
		"output_signals": outputs,
		"input_signals_used": input_signals_used,
		"input_neuron_indices": _meta._input_indices,
		"output_slot_by_neuron_index": _meta._output_slots,
		"iteration_count": _iteration,
		"elapsed_time_seconds": _run_time,
		"total_connections": live_edges,
		"total_connections_destroyed": 0,
		"total_homeostatic_connections_formed": 0,
		"total_structural_plasticity_connections_formed": 0,
		"added_connections_last_iteration": 0,
		"input_gain": float(_brain["input_gain"]),
		"input_min_weight": 1.0 / (1.0 + float(_brain["input_gain"])),
		"mean_signal": signal_sum / float(maxi(1, _total)),
		"mean_weight": weight_sum / float(maxi(1, live_edges)),
		"mean_outgoing_connections": int(float(internal_out_sum) / float(maxi(1, outgoing_n))),
		"min_out_degree": 0 if outgoing_n == 0 else min_degree,
		"max_out_degree": max_degree
	}
	_new_edges_since_snapshot = 0
	return true

func get_viewer_state() -> Dictionary:
	return _statistics

func export_lines(selected_spatial: int = -1, capacity: int = MAX_LINE_SEGMENTS) -> Dictionary:
	if _core == null:
		return {}
	return _core.inspection_lines(0, selected_spatial, capacity)

func read_synapses() -> Array:
	return _core.inspection_synapses(0) if _core != null else []

func count_reciprocal_edges() -> int:
	var present: Dictionary = {}
	for entry in read_synapses():
		present[Vector2i(int(entry[0]), int(entry[1]))] = true
	var total: int = 0
	for pair in present:
		if present.has(Vector2i(pair.y, pair.x)):
			total += 1
	return total

func get_initial_degrees() -> PackedInt32Array:
	return _core.inspection_incoming_degrees(0) if _core != null else PackedInt32Array()

func clear() -> void:
	if _core != null:
		_core.clear()
	_core = null
	rd = null
	_meta = null
	_brain = {}
	_statistics.clear()
	_pending = PackedFloat32Array()
	_total = 0
	_iteration = 0
	_run_time = 0.0
	_lost_edges = 0
	_reconnected = 0
	_structural_created = 0
	_new_edges_since_snapshot = 0
