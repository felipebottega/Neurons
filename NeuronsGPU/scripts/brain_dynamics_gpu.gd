class_name BrainDynamicsGPU
extends RefCounted

# The persistent simulation is spatially indexed and entirely GPU-resident.
# Only neuron states/statistics and a bounded, on-demand line sample are read
# for the Godot 3D UI. Runtime edges are never unpacked into GDScript dictionaries.
# Existing validated kernel paths remain unchanged during this extraction.
# Shader file names can be cleaned up after execution equivalence is restored.
const SHADER_PREFIX := "res://shaders/brain_viewer_"
const MAX_LINE_SEGMENTS := 180000
# Diagnostic timings only. Each value is CPU wall-clock time, not an independent
# GPU hardware timer. In particular submit+sync includes waiting for the GPU.
const PROFILE_DYNAMICS := true

var rd: RenderingDevice
var _shader_cache: Dictionary = {}
var _buffers: Dictionary = {}
var _brain: Dictionary = {}
var _size := Vector3i.ZERO
var _total := 0
var _stride := 0
var _input_indices := PackedInt32Array() # ordered public neuron indices
var _output_indices := PackedInt32Array()
var _rank_to_spatial := PackedInt32Array()
var _rank_is_output := PackedByteArray()
var _decay_rates := PackedFloat32Array()
var _output_slots := PackedInt32Array()
var _new_edges_since_snapshot := 0
var _statistics_iteration := -1
var _last_advance_profile_us: Dictionary = {}
var _pending := PackedFloat32Array()
var _run_time := 0.0
var _iteration := 0
var _lost_edges := 0
var _reconnected := 0
var _structural_created := 0
var _warnings_capacity := 0
var _statistics: Dictionary = {}
var _last_batch_profile_us: Dictionary = {}
var _last_pull_profile_us: Dictionary = {}
var _owns_shader_cache := true
var _owns_device := true
var _retain_initial := false
var _initial_stride := 0


# Per-brain values belong to one dynamics session, not to the RefCounted
# object's lifetime. Call only before allocating a brain or after freeing it.
func _reset_session_state() -> void:
	_brain = {}
	_size = Vector3i.ZERO
	_total = 0
	_stride = 0
	_input_indices = PackedInt32Array()
	_output_indices = PackedInt32Array()
	_rank_to_spatial = PackedInt32Array()
	_rank_is_output = PackedByteArray()
	_decay_rates = PackedFloat32Array()
	_output_slots = PackedInt32Array()
	_pending = PackedFloat32Array()
	_iteration = 0
	_run_time = 0.0
	_lost_edges = 0
	_reconnected = 0
	_structural_created = 0
	_warnings_capacity = 0
	_new_edges_since_snapshot = 0
	_statistics_iteration = -1
	_statistics = {}
	_last_advance_profile_us.clear()
	_last_batch_profile_us.clear()
	_last_pull_profile_us.clear()


func initialize(brain: Dictionary, shared_shaders: Dictionary = {}, retain_initial: bool = false, defer_initial_copy: bool = false) -> bool:
	if rd != null:
		push_error("BrainDynamicsGPU: clear the previous brain before initializing another")
		return false
	# Batched simulation setup requires immutable initial synapses to remain
	# resident until SimulationsGPUBackend dispatches all copies together.
	if defer_initial_copy and not retain_initial:
		push_error("BrainDynamicsGPU: deferred initial copy requires retained GPU synapses")
		return false
	if not brain.has("gpu_state") or not bool(brain.get("valid", false)):
		push_error("BrainDynamicsGPU: expected a retained GPU phenotype")
		return false
	_reset_session_state()
	_brain = brain
	var gpu: Dictionary = brain["gpu_state"]
	rd = gpu["rd"]
	_owns_device = bool(gpu.get("owns_rd", true))
	_owns_shader_cache = shared_shaders.is_empty()
	_retain_initial = retain_initial
	_size = brain["size"]
	_total = _size.x * _size.y * _size.z
	var original_stride: int = gpu["slot_stride"]
	_initial_stride = original_stride
	_stride = original_stride + maxi(64, ceili(float(original_stride) * 0.5))
	_build_rank_maps()
	_pending.resize(_total)
	_pending.fill(0.0)
	var plane: int = _size.x * _size.y
	var zeros_heads: Array[RID] = []
	var edges: Array[RID] = []
	var links: Array[RID] = []
	var transmissions: Array[RID] = []
	for k in _size.z:
		var head_values := PackedInt32Array()
		head_values.resize(_total)
		head_values.fill(-1)
		zeros_heads.append(rd.storage_buffer_create(head_values.size() * 4, head_values.to_byte_array()))
		edges.append(rd.storage_buffer_create(plane * _stride * 16))
		links.append(rd.storage_buffer_create(plane * _stride * 4))
		transmissions.append(rd.storage_buffer_create(plane * _stride * 8))
	_buffers = {
		"phenotype": gpu["neuron_buffer"],
		"roles": gpu["roles_buffer"],
		"edges": edges,
		"counts": rd.storage_buffer_create(_total * 4),
		"live_degrees": rd.storage_buffer_create(_total * 4),
		"incoming_degrees": rd.storage_buffer_create(_total * 4, rd.buffer_get_data(gpu["incoming_buffer"])),
		"heads": zeros_heads,
		"links": links,
		"transmissions": transmissions,
		"states": _zero_buffer(_total * 48),
		"modulation": _zero_buffer(_total * 8),
		"incoming": _zero_buffer(_total * 4),
		"pending": _zero_buffer(_total * 4),
		"changes": _zero_buffer(20),
		"rescue_flags": _zero_buffer(_total * 4),
		"rescue_sources": _zero_buffer(_total * 4),
		"statistics": rd.storage_buffer_create(_total * 16),
		"initial_slots": gpu["slot_buffers"] if retain_initial else [],
		"initial_counts": gpu["outgoing_buffer"] if retain_initial else RID()
	}
	for key in _buffers:
		if key == "initial_counts" or key == "initial_slots":
			continue
		var resource: Variant = _buffers[key]
		if resource is Array:
			for rid in resource:
				if not rid.is_valid():
					push_error("BrainDynamicsGPU: allocation failed for " + key)
					return false
		elif not resource.is_valid():
			push_error("BrainDynamicsGPU: allocation failed for " + key)
			return false
	if _owns_shader_cache:
		if not _load_shaders():
			return false
	else:
		_shader_cache = shared_shaders.duplicate()
	if not defer_initial_copy:
		for k in _size.z:
			if not _dispatch("copy", [gpu["slot_buffers"][k], gpu["outgoing_buffer"], _buffers.edges[k], _buffers.counts, _buffers.live_degrees, _buffers.heads[k], _buffers.links[k], _buffers.changes], plane, original_stride, k):
				return false
		var errors: PackedInt32Array = rd.buffer_get_data(_buffers.changes).to_int32_array()
		if errors.size() < 4 or errors[3] != 0:
			push_error("BrainDynamicsGPU: invalid initial synapse target index")
			return false
	# Only the retained construction slots are released; the phenotype and roles
	# remain GPU-resident and are owned by the dynamics instance until clear().
	if not retain_initial:
		for slot in gpu["slot_buffers"]:
			rd.free_rid(slot)
		rd.free_rid(gpu["outgoing_buffer"])
	rd.free_rid(gpu["incoming_buffer"])
	gpu["outgoing_buffer"] = RID()
	gpu["incoming_buffer"] = RID()
	# The retained immutable source buffers are now owned by _buffers.
	gpu["slot_buffers"] = []
	if retain_initial:
		gpu["outgoing_buffer"] = RID()
	brain["gpu_state"] = gpu
	# The simulation has no use for a full CPU state snapshot here. It reads
	# agent positions per frame; neuron details are captured only when selected.
	if not defer_initial_copy and not _pull_state():
		return false
	return true

func _zero_buffer(byte_count: int) -> RID:
	var zeros := PackedByteArray()
	zeros.resize(byte_count)
	return rd.storage_buffer_create(byte_count, zeros)

func _build_rank_maps() -> void:
	_rank_to_spatial.resize(_total)
	_rank_is_output.resize(_total)
	_decay_rates.resize(_total)
	_output_slots.resize(_total)
	_output_slots.fill(-1)
	_input_indices = PackedInt32Array()
	_output_indices = PackedInt32Array()
	for idx in _total:
		var neuron: Dictionary = _brain["neurons"][idx]
		var pos: Vector3i = neuron["position"]
		_rank_to_spatial[idx] = pos.x + _size.x * (pos.y + _size.y * pos.z)
		_rank_is_output[idx] = 1 if neuron["output"] else 0
		_decay_rates[idx] = 0.0 if neuron["input"] else exp(-12.0 / float(neuron["decay_factor"]))
		if neuron["input"]:
			_input_indices.append(idx)
		elif neuron["output"]:
			_output_slots[idx] = _output_indices.size()
			_output_indices.append(idx)

func _load_shaders() -> bool:
	for label in ["copy", "prepare", "modulation", "edges", "gather", "state", "homeostasis_select", "homeostasis", "structural", "statistics", "export", "clear_incoming"]:
		var path: String = SHADER_PREFIX + label + ".glsl"
		var resource: RDShaderFile = load(path)
		if resource == null:
			push_error("BrainDynamicsGPU: missing shader: " + path)
			return false
		var spirv: RDShaderSPIRV = resource.get_spirv()
		if spirv == null:
			return false
		var error: String = spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
		if not error.is_empty():
			push_error("BrainDynamicsGPU: shader error in " + path + "\n" + error)
			return false
		var shader: RID = rd.shader_create_from_spirv(spirv)
		var pipeline: RID = rd.compute_pipeline_create(shader)
		if not shader.is_valid() or not pipeline.is_valid():
			push_error("BrainDynamicsGPU: cannot create pipeline: " + path)
			return false
		_shader_cache[label] = {"shader": shader, "pipeline": pipeline}
	return true

func _params(max_items: int = 0, slice_k: int = 0, selected_id: int = -1) -> PackedByteArray:
	var b := PackedByteArray()
	b.resize(80)
	for index in [0,1,2,3,4,5,17,18,19]:
		var number := 0
		match index:
			0: number = _size.x
			1: number = _size.y
			2: number = _size.z
			3: number = _total
			4: number = _stride
			5: number = _iteration
			17: number = slice_k
			18: number = max_items
			19: number = selected_id if selected_id >= 0 else _total
		b.encode_u32(index * 4, number)
	b.encode_float(24, 0.0)
	b.encode_float(28, float(_brain["exponential_factor"]))
	b.encode_float(32, float(_brain["refractory_strength"]))
	b.encode_float(36, float(_brain["mu"]))
	b.encode_float(40, float(_brain["lambda"]))
	b.encode_float(44, float(_brain["nu"]))
	var factor: float = float(_brain["exponential_factor"])
	b.encode_float(48, -factor * log(0.95))
	b.encode_float(52, -factor * log(0.05))
	b.encode_float(56, float(_brain["structural_plasticity"]))
	b.encode_float(60, float(_brain["input_gain"]))
	b.encode_float(64, 1.0 / (1.0 + float(_brain["input_gain"])))
	return b

# The homeostatic selection and creation shaders use the same V_init box
# as the first pass at construction. Append its three per-brain radii only
# for those shaders; all other push-constant layouts remain unchanged.
func _homeostasis_params(slice_k: int = 0) -> PackedByteArray:
	var params: PackedByteArray = _params(0, slice_k)
	params.resize(92)
	params.encode_u32(80, int(_brain["initial_search_radius_i"]))
	params.encode_u32(84, int(_brain["initial_search_radius_j"]))
	params.encode_u32(88, int(_brain["initial_search_radius_k"]))
	return params

func _dispatch(label: String, bindings: Array, count: int, max_items: int = 0, slice_k: int = 0, selected_id: int = -1) -> bool:
	if not _shader_cache.has(label):
		return false
	var resources: Dictionary = _shader_cache[label]
	var uniforms: Array[RDUniform] = []
	for i in bindings.size():
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(bindings[i])
		uniforms.append(u)
	var uniform_set: RID = rd.uniform_set_create(uniforms, resources["shader"], 0)
	if not uniform_set.is_valid():
		push_error("BrainDynamicsGPU: invalid bindings in " + label)
		return false
	var params: PackedByteArray = _homeostasis_params(slice_k) if label in ["homeostasis_select", "homeostasis"] else _params(max_items, slice_k, selected_id)
	var commands: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, resources["pipeline"])
	rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
	rd.compute_list_set_push_constant(commands, params, params.size())
	rd.compute_list_dispatch(commands, ceili(float(count)/64.0), 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	rd.free_rid(uniform_set)
	return true

func _dispatch_batch(steps: Array) -> bool:
	# Each step specifies its own work size and source slice. All steps share
	# one device submission, but are separated by explicit memory barriers.
	var started_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	var bound_sets: Array[RID] = []
	var commands: int = rd.compute_list_begin()
	for n in steps.size():
		var step: Array = steps[n]
		var label: String = step[0]
		var bindings: Array = step[1]
		var count: int = step[2]
		var slice_k: int = step[3] if step.size() > 3 else 0
		var resources: Dictionary = _shader_cache[label]
		var uniforms: Array[RDUniform] = []
		for i in bindings.size():
			var item := RDUniform.new()
			item.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
			item.binding = i
			item.add_id(bindings[i])
			uniforms.append(item)
		var uniform_set: RID = rd.uniform_set_create(uniforms, resources.shader, 0)
		if not uniform_set.is_valid():
			push_error("BrainDynamicsGPU: invalid bindings for " + label)
			rd.compute_list_end()
			for existing in bound_sets:
				rd.free_rid(existing)
			return false
		bound_sets.append(uniform_set)
		rd.compute_list_bind_compute_pipeline(commands, resources.pipeline)
		rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
		var params: PackedByteArray = _homeostasis_params(slice_k) if label in ["homeostasis_select", "homeostasis"] else _params(0, slice_k)
		rd.compute_list_set_push_constant(commands, params, params.size())
		rd.compute_list_dispatch(commands, ceili(float(count) / 64.0), 1, 1)
		if n + 1 < steps.size():
			rd.compute_list_add_barrier(commands)
	rd.compute_list_end()
	var submitted_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	rd.submit()
	rd.sync()
	var synchronized_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	for uniform_set in bound_sets:
		rd.free_rid(uniform_set)
	if PROFILE_DYNAMICS:
		_last_batch_profile_us = {
			"gpu_record": submitted_us - started_us,
			"gpu_submit_sync": synchronized_us - submitted_us,
			"gpu_cleanup": Time.get_ticks_usec() - synchronized_us
		}
	return true


func _reset_heads() -> void:
	for k in _size.z:
		var head_values := PackedInt32Array()
		head_values.resize(_total)
		head_values.fill(-1)
		var bytes: PackedByteArray = head_values.to_byte_array()
		rd.buffer_update(_buffers.heads[k], 0, bytes.size(), bytes)


func _compact_dead_connections() -> bool:
	# The copy kernel's max_items=0 mode compacts in place, rebuilding local
	# linked lists. One invocation owns each source, avoiding write conflicts.
	_reset_heads()
	var plane: int = _size.x * _size.y
	for k in _size.z:
		if not _dispatch("copy", [_buffers.edges[k], _buffers.counts, _buffers.edges[k], _buffers.counts, _buffers.live_degrees, _buffers.heads[k], _buffers.links[k], _buffers.changes], plane, 0, k):
			return false
	return true


func _grow_to(required: int) -> bool:
	if required <= _stride:
		return true
	var previous_stride: int = _stride
	var new_stride: int = maxi(required, previous_stride * 2)
	var plane: int = _size.x * _size.y
	var new_edges: Array[RID] = []
	var new_links: Array[RID] = []
	var new_transmissions: Array[RID] = []
	for k in _size.z:
		new_edges.append(rd.storage_buffer_create(plane * new_stride * 16))
		new_links.append(rd.storage_buffer_create(plane * new_stride * 4))
		new_transmissions.append(rd.storage_buffer_create(plane * new_stride * 8))
	for collection in [new_edges, new_links, new_transmissions]:
		for rid in collection:
			if not rid.is_valid():
				push_error("BrainDynamicsGPU: GPU rejected connection buffer expansion")
				for owned_collection in [new_edges, new_links, new_transmissions]:
					for allocated in owned_collection:
						if allocated.is_valid():
							rd.free_rid(allocated)
				return false
	# All live slots were compacted before this call. The existing copy
	# kernel moves them directly on the GPU and rebuilds each local head list,
	# preserving weights and traces without any CPU synapse readback.
	_reset_heads()
	_stride = new_stride
	for k in _size.z:
		if not _dispatch("copy", [_buffers.edges[k], _buffers.counts, new_edges[k], _buffers.counts, _buffers.live_degrees, _buffers.heads[k], new_links[k], _buffers.changes], plane, previous_stride, k):
			_stride = previous_stride
			for owned_collection in [new_edges, new_links, new_transmissions]:
				for rid in owned_collection:
					if rid.is_valid():
						rd.free_rid(rid)
			return false
	for key in ["edges", "links", "transmissions"]:
		for rid in _buffers[key]:
			rd.free_rid(rid)
	_buffers.edges = new_edges
	_buffers.links = new_links
	_buffers.transmissions = new_transmissions
	print("[BrainDynamicsGPU] Expanded synaptic stride ", previous_stride, " -> ", new_stride)
	return true


func _ensure_capacity_for_next_step() -> bool:
	# Each target can request at most one rescue; each internal source can form
	# at most one structural synapse in this iteration. Reserve for both before
	# running either kernel, never dropping a new connection on a full slot.
	var count_values: PackedInt32Array = rd.buffer_get_data(_buffers.counts).to_int32_array()
	var selected: PackedInt32Array = rd.buffer_get_data(_buffers.rescue_sources).to_int32_array()
	if count_values.size() != _total or selected.size() != _total:
		push_error("BrainDynamicsGPU: incomplete GPU capacity preflight readback")
		return false
	var demand := PackedInt32Array()
	demand.resize(_total)
	for target in _total:
		var source: int = selected[target]
		if source >= 0 and source < _total:
			demand[source] += 1
	var required: int = _stride
	for source in _total:
		# Public neurons are energy-ranked, so map back to spatial indices.
		var spatial: int = _rank_to_spatial[source]
		var neuron: Dictionary = _brain["neurons"][source]
		if not bool(neuron["input"]) and not bool(neuron["output"]):
			demand[spatial] += 1
		required = maxi(required, count_values[spatial] + demand[spatial])
	return _grow_to(required)


func advance(delta_seconds: float, capture_snapshot: bool = true) -> int:
	if rd == null:
		return 0
	var started_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	_iteration += 1
	_run_time += delta_seconds
	var pending_bytes: PackedByteArray = _pending.to_byte_array()
	rd.buffer_update(_buffers.pending, 0, pending_bytes.size(), pending_bytes)
	_pending.fill(0.0)
	# The prepare shader resets changes and rescue_flags on GPU, preserving
	# the same stage ordering and removing two host buffer updates per tick.
	var uploaded_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	var b := _buffers
	var plane: int = _size.x * _size.y
	var steps: Array = [
		["prepare", [b.phenotype,b.roles,b.states,b.pending,b.changes,b.rescue_flags], _total],
		["modulation", [b.states,b.phenotype,b.modulation], _total],
		["clear_incoming", [b.incoming], _total]
	]
	for k in _size.z:
		steps.append(["edges", [b.phenotype,b.roles,b.states,b.modulation,b.edges[k],b.counts,b.incoming_degrees,b.live_degrees,b.transmissions[k],b.changes,b.rescue_flags], plane, k])
	# Every source slice can now transmit to every target slice.
	# Dispatch a destination-complete gather for each source slice. Barriers
	# in _dispatch_batch preserve the previous deterministic summation order.
	for k in _size.z:
		steps.append(["gather", [b.heads[k],b.links[k],b.transmissions[k],b.incoming], _total, k])
	steps.append(["state", [b.phenotype,b.roles,b.states,b.modulation,b.incoming], _total])
	steps.append(["homeostasis_select", [b.roles,b.states,b.incoming_degrees,b.rescue_flags,b.rescue_sources], _total])
	if not _dispatch_batch(steps):
		return -1
	var changes: PackedInt32Array = rd.buffer_get_data(b.changes).to_int32_array()
	if changes.size() < 5:
		push_error("BrainDynamicsGPU: incomplete GPU changes readback")
		return -1
	if changes[0] > 0 and not _compact_dead_connections():
		return -1
	if not _ensure_capacity_for_next_step():
		return -1
	# The two plasticity kernels are the only stages that create synapses.
	# An upstream capacity preflight guarantees their reservations fit.
	b = _buffers
	var plasticity: Array = []
	var homeostasis_radius_k: int = int(_brain["initial_search_radius_k"])
	for k in _size.z:
		var low: int = maxi(0, k - homeostasis_radius_k)
		var high: int = mini(_size.z - 1, k + homeostasis_radius_k)
		plasticity.append(["homeostasis", [b.roles,b.rescue_sources,b.edges[k],b.counts,b.incoming_degrees,b.live_degrees,b.heads[k],b.links[k],b.changes], (high - low + 1) * plane, k])
	for k in _size.z:
		plasticity.append(["structural", [b.roles,b.states,b.edges[k],b.counts,b.incoming_degrees,b.live_degrees,b.heads[k],b.links[k],b.changes], plane, k])
	if capture_snapshot:
		for k in _size.z:
			plasticity.append(["statistics", [b.edges[k],b.counts,b.live_degrees,b.statistics], plane, k])
	if not _dispatch_batch(plasticity):
		return -1
	if capture_snapshot:
		_statistics_iteration = _iteration
	changes = rd.buffer_get_data(b.changes).to_int32_array()
	if changes.size() < 5 or changes[3] != 0 or changes[4] != 0:
		push_error("BrainDynamicsGPU: connection integrity/capacity failure")
		return -1
	_lost_edges += changes[0]
	_reconnected += changes[1]
	_structural_created += changes[2]
	_warnings_capacity += changes[4]
	var changes_read_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	_new_edges_since_snapshot += changes[1] + changes[2]
	var pulled := false
	if capture_snapshot:
		if not capture_visual_state():
			return -1
		pulled = true
	if PROFILE_DYNAMICS:
		_last_advance_profile_us = {
			"upload": uploaded_us - started_us,
			"gpu_record": _last_batch_profile_us.get("gpu_record", 0),
			"gpu_submit_sync": _last_batch_profile_us.get("gpu_submit_sync", 0),
			"gpu_cleanup": _last_batch_profile_us.get("gpu_cleanup", 0),
			"changes_readback": changes_read_us - uploaded_us,
			"state_readback": _last_pull_profile_us.get("state_readback", 0) if pulled else 0,
			"state_unpack": _last_pull_profile_us.get("state_unpack", 0) if pulled else 0,
			"total": Time.get_ticks_usec() - started_us
		}
	return changes[0]

func add_input_signal(input_slot: int, value: float) -> void:
	if input_slot >= 0 and input_slot < _input_indices.size():
		_pending[_rank_to_spatial[_input_indices[input_slot]]] = value

func _pull_state() -> bool:
	# Expensive snapshot, NOT part of each simulation step. Keep numerical
	# accumulation identical to the original code; avoid Dictionary indexing,
	# gene parameter lookups and exp() inside the neuron loop.
	var started_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	var raw: PackedFloat32Array = rd.buffer_get_data(_buffers.states).to_float32_array()
	var weights: PackedFloat32Array = rd.buffer_get_data(_buffers.statistics).to_float32_array() if _iteration > 0 else PackedFloat32Array()
	var incoming: PackedInt32Array = rd.buffer_get_data(_buffers.incoming_degrees).to_int32_array()
	var living: PackedInt32Array = rd.buffer_get_data(_buffers.live_degrees).to_int32_array()
	var read_us: int = Time.get_ticks_usec() if PROFILE_DYNAMICS else 0
	if raw.size() < 12 * _total or incoming.size() < _total or living.size() < _total or (_iteration > 0 and weights.size() < 4 * _total):
		push_error("BrainDynamicsGPU: incomplete state readback; GPU simulation stopped")
		return false
	var normalized := PackedFloat32Array()
	var states := PackedFloat32Array()
	var next_incoming := PackedFloat32Array()
	var fatigues := PackedFloat32Array()
	var fatigues_used := PackedFloat32Array()
	var traces := PackedFloat32Array()
	var thresholds := PackedFloat32Array()
	var releases := PackedFloat32Array()
	var fields := PackedFloat32Array()
	var modulations := PackedFloat32Array()
	var firing := PackedFloat32Array()
	var incoming_counts := PackedFloat32Array()
	var outgoing_counts := PackedFloat32Array()
	normalized.resize(_total)
	states.resize(_total)
	next_incoming.resize(_total)
	fatigues.resize(_total)
	fatigues_used.resize(_total)
	traces.resize(_total)
	thresholds.resize(_total)
	releases.resize(_total)
	fields.resize(_total)
	modulations.resize(_total)
	firing.resize(_total)
	incoming_counts.resize(_total)
	outgoing_counts.resize(_total)
	var output_signals := PackedFloat64Array()
	output_signals.resize(_output_indices.size())
	var input_signals_used := PackedFloat32Array()
	input_signals_used.resize(_input_indices.size())
	var signal_sum := 0.0
	var release_sum := 0.0
	var field_sum := 0.0
	var modulation_sum := 0.0
	var weight_sum := 0.0
	var trace_sum := 0.0
	var live_edges := 0
	var internal_out_sum := 0
	var outgoing_n := 0
	var min_degree := 2147483647
	var max_degree := 0
	for rank in _total:
		var spatial: int = _rank_to_spatial[rank]
		var offset: int = 12 * spatial
		var neuron_signal: float = raw[offset + 3]
		var release: float = raw[offset + 9]
		var field: float = raw[offset + 8]
		var modulation: float = raw[offset + 10]
		var degree: int = living[spatial]
		normalized[rank] = neuron_signal
		states[rank] = raw[offset + 4]
		next_incoming[rank] = raw[offset + 7]
		fatigues[rank] = raw[offset + 1]
		fatigues_used[rank] = raw[offset + 5]
		traces[rank] = raw[offset + 2]
		thresholds[rank] = raw[offset + 6]
		releases[rank] = release
		fields[rank] = field
		modulations[rank] = modulation
		firing[rank] = raw[offset + 11]
		incoming_counts[rank] = float(incoming[spatial])
		outgoing_counts[rank] = float(degree)
		signal_sum += neuron_signal
		release_sum += release
		field_sum += field
		modulation_sum += modulation
		live_edges += degree
		if _rank_is_output[rank] == 0:
			outgoing_n += 1
			internal_out_sum += degree
			min_degree = mini(min_degree, degree)
			max_degree = maxi(max_degree, degree)
		if _iteration > 0:
			weight_sum += weights[4 * spatial]
			trace_sum += weights[4 * spatial + 1]
	for i in _output_indices.size():
		var r: int = _output_indices[i]
		output_signals[i] = normalized[r] if firing[r] > 0.5 else 0.0
	for i in _input_indices.size():
		input_signals_used[i] = states[_input_indices[i]]
	_statistics = {
		"normalized_signals": normalized,
		"neuron_states_used": states,
		"incoming_signals": next_incoming,
		"fatigues": fatigues,
		"fatigues_used": fatigues_used,
		"activity_traces": traces,
		"effective_thresholds_used": thresholds,
		"decay_rates": _decay_rates,
		"modulatory_releases": releases,
		"modulatory_fields": fields,
		"effective_modulations": modulations,
		"firing_states": firing,
		"incoming_connection_counts": incoming_counts,
		"outgoing_connection_counts": outgoing_counts,
		"output_signals": output_signals,
		"input_signals_used": input_signals_used,
		"input_neuron_indices": _input_indices,
		"output_slot_by_neuron_index": _output_slots,
		"iteration_count": _iteration,
		"elapsed_time_seconds": _run_time,
		"total_connections": live_edges,
		"total_connections_destroyed": _lost_edges,
		"total_homeostatic_connections_formed": _reconnected,
		"total_structural_plasticity_connections_formed": _structural_created,
		"added_connections_last_iteration": 0,
		"input_gain": float(_brain.input_gain),
		"input_min_weight": 1.0 / (1.0 + float(_brain.input_gain)),
		"mean_signal": signal_sum / maxi(1, _total),
		"mean_modulatory_release": release_sum / maxi(1, _total),
		"mean_modulatory_field": field_sum / maxi(1, _total),
		"mean_effective_modulation": modulation_sum / maxi(1, _total),
		"mean_weight": weight_sum / maxi(1, live_edges) if _iteration > 0 else float(_brain.get("connection_weight_sum", 0.0)) / maxi(1, int(_brain.get("connection_total", 0))),
		"mean_eligibility_trace": trace_sum / maxi(1, live_edges),
		"mean_outgoing_connections": int(float(internal_out_sum) / maxi(1, outgoing_n)),
		"min_out_degree": 0 if outgoing_n == 0 else min_degree,
		"max_out_degree": max_degree
	}
	if PROFILE_DYNAMICS:
		_last_pull_profile_us = {
			"state_readback": read_us - started_us,
			"state_unpack": Time.get_ticks_usec() - read_us
		}
	return true

# Allows a precise paused view without advancing the neuronal dynamics.
func capture_visual_state() -> bool:
	if rd == null:
		return false
	# A pause may occur after a non-visual iteration. Refresh weight/trace
	# aggregates once without advancing any of the neuronal dynamics.
	if _iteration > 0 and _statistics_iteration != _iteration:
		var b := _buffers
		var plane: int = _size.x * _size.y
		for k in _size.z:
			if not _dispatch("statistics", [b.edges[k],b.counts,b.live_degrees,b.statistics], plane, 0, k):
				return false
		_statistics_iteration = _iteration
	if not _pull_state():
		return false
	_statistics["added_connections_last_iteration"] = _new_edges_since_snapshot
	_new_edges_since_snapshot = 0
	return true

func get_viewer_state() -> Dictionary:
	return _statistics

func get_last_advance_profile() -> Dictionary:
	return _last_advance_profile_us

func export_lines(selected_spatial: int = -1, capacity: int = MAX_LINE_SEGMENTS) -> Dictionary:
	if rd == null:
		return {}
	var count_buffer: RID = _zero_buffer(4)
	var lines_buffer: RID = rd.storage_buffer_create(capacity * 16)
	var plane: int = _size.x * _size.y
	for k in _size.z:
		if not _dispatch("export", [_buffers.edges[k],_buffers.counts,count_buffer,lines_buffer],plane,capacity,k,selected_spatial):
			rd.free_rid(count_buffer)
			rd.free_rid(lines_buffer)
			return {}
	var count_bytes: PackedInt32Array = rd.buffer_get_data(count_buffer).to_int32_array()
	var available: int = mini(int(count_bytes[0]), capacity)
	var values: PackedFloat32Array = rd.buffer_get_data(lines_buffer,0,available*16).to_float32_array() if available>0 else PackedFloat32Array()
	rd.free_rid(count_buffer)
	rd.free_rid(lines_buffer)
	return {"data":values,"reported":count_bytes[0],"count":available}


func read_synapses() -> Array:
	# The exhaustive readback is for validation only, never for Viewer stepping.
	var result: Array = []
	if rd == null:
		return result
	var plane: int = _size.x * _size.y
	var counts: PackedInt32Array = rd.buffer_get_data(_buffers.counts).to_int32_array()
	for k in _size.z:
		var raw: PackedByteArray = rd.buffer_get_data(_buffers.edges[k])
		for local in plane:
			var source: int = k * plane + local
			for slot in counts[source]:
				var off: int = (local * _stride + slot) * 16
				var target: int = raw.decode_s32(off)
				if target >= 0:
					result.append([source, target, raw.decode_float(off + 4), raw.decode_float(off + 8)])
	result.sort_custom(func(a, b):
		if int(a[0]) != int(b[0]):
			return int(a[0]) < int(b[0])
		return int(a[1]) < int(b[1]))
	return result


func count_reciprocal_edges() -> int:
	if rd == null:
		return 0
	# Diagnostics only: the old shader checked just five neighbouring
	# slices. Checking all K*K slice pairs on the GPU would be extremely
	# expensive; scan the live edges once and include remote pairs.
	var counts: PackedInt32Array = rd.buffer_get_data(_buffers.counts).to_int32_array()
	if counts.size() < _total:
		return 0
	var present: Dictionary = {}
	var plane: int = _size.x * _size.y
	for k in _size.z:
		var raw: PackedByteArray = rd.buffer_get_data(_buffers.edges[k])
		for local in plane:
			var source: int = k * plane + local
			for slot in counts[source]:
				var target: int = raw.decode_s32((local * _stride + slot) * 16)
				if target >= 0 and target < _total:
					present[Vector2i(source, target)] = true
	var result: int = 0
	for pair in present:
		if present.has(Vector2i(pair.y, pair.x)):
			result += 1
	return result


func get_initial_degrees() -> PackedInt32Array:
	return rd.buffer_get_data(_buffers.incoming_degrees).to_int32_array()


func clear() -> void:
	if rd == null:
		_reset_session_state()
		return
	if _owns_shader_cache:
		for key in _shader_cache:
			rd.free_rid(_shader_cache[key].pipeline)
			rd.free_rid(_shader_cache[key].shader)
	_shader_cache.clear()
	for key in _buffers:
		if key == "phenotype" or key == "roles":
			continue
		var resource: Variant = _buffers[key]
		if resource is Array:
			for rid in resource:
				if rid.is_valid():
					rd.free_rid(rid)
		elif resource is RID and resource.is_valid():
			rd.free_rid(resource)
	_buffers.clear()
	if _brain.has("gpu_state"):
		var remaining: Dictionary = _brain["gpu_state"]
		for key in ["neuron_buffer", "roles_buffer", "outgoing_buffer", "incoming_buffer"]:
			if remaining.has(key) and remaining[key].is_valid():
				rd.free_rid(remaining[key])
		for slot in remaining.get("slot_buffers", []):
			if slot.is_valid():
				rd.free_rid(slot)
		_brain.erase("gpu_state")
	if _owns_device:
		rd.free()
	rd = null
	_reset_session_state()

func reset(defer_synapse_restore: bool = false) -> void:
	# The ordinary Viewer does not retain the genotype's immutable GPU edges.
	# Simulation instances do, allowing identical resets between target tests.
	if not _retain_initial or rd == null:
		push_warning("BrainDynamicsGPU: reset requires a retained initial synapse snapshot")
		return
	_iteration = 0
	_run_time = 0.0
	_lost_edges = 0
	_reconnected = 0
	_structural_created = 0
	_warnings_capacity = 0
	_new_edges_since_snapshot = 0
	_statistics_iteration = -1
	_statistics = {}
	_pending.fill(0.0)
	for key in ["states", "modulation", "incoming", "pending", "changes", "rescue_flags", "rescue_sources", "statistics"]:
		var previous_rid: RID = _buffers[key]
		var byte_size: int = 20 if key == "changes" else 0
		# These byte sizes are fixed regardless of changes to synaptic stride.
		match key:
			"states": byte_size = _total * 48
			"modulation": byte_size = _total * 8
			"incoming", "pending", "rescue_flags", "rescue_sources": byte_size = _total * 4
			"statistics": byte_size = _total * 16
		var empty_bytes := PackedByteArray()
		empty_bytes.resize(byte_size)
		rd.buffer_update(previous_rid, 0, byte_size, empty_bytes)
	# Exact initial incoming counts (the copy shader restores outgoing counts).
	rd.buffer_update(_buffers.incoming_degrees, 0, _total * 4, _brain["incoming_counts"].to_byte_array())
	_reset_heads()
	if defer_synapse_restore:
		# SimulationsGPUBackend restores all one hundred brains in one GPU batch.
		return
	for k in _size.z:
		if not _dispatch("copy", [_buffers.initial_slots[k], _buffers.initial_counts, _buffers.edges[k], _buffers.counts, _buffers.live_degrees, _buffers.heads[k], _buffers.links[k], _buffers.changes], _size.x * _size.y, _initial_stride, k):
			push_error("BrainDynamicsGPU: could not restore initial edges")
			return
