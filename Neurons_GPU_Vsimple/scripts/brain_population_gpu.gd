class_name BrainPopulationGPU
extends RefCounted

# Population-wide Vsimple GPU neural engine.
# Once built, weights and topology are immutable. Three GPU stages run at
# every step: source emission, CSR matrix-vector product and state replacement.
# The fixed-slot ELLPACK weights and immutable receiver-indexed CSR are created
# once from the GPU builder's authoritative initial graph.
const WORKGROUP: int = 64
const EDGE_TILE: int = 4
# Compute-plan switches retained for selecting V5 or V7, independently of profiling.
# The legacy implementation remains available for numerical cross-checking.
const USE_FUSED_CSR_EDGES: bool = false
# V7 A/B switch: false restores the original V5 compute plan (same scenes and
# mathematical model). Switch only before starting a population.
const USE_V7_PACKED_SOURCE_AND_FUSION: bool = false
const FUSED_WORKGROUP: int = 64  # supported: 64 or 128
const PARAM_BYTES: int = 112
const STATUS_WORDS: int = 6
const CHANGE_WORDS: int = 6  # The six production graph-event counters are unchanged.
# Godot's storage buffer size parameter is a 32-bit quantity. This is a
# per-buffer representation boundary, NOT a 2 GiB limit on combined GPU memory.
# The actual device can impose a lower storage-buffer range.
const MAX_STORAGE_BYTES: int = 4294967295
const PREFIX: String = "res://shaders/brain_packed_"
const KERNELS := ["initial_upload", "simple_prepare", "spmv", "simple_state",
    "simple_reset", "simple_state_reset", "restore", "csr_init", "csr_reserve", "csr_fill", "status",
    "inspect", "visual_lines"]

var rd: RenderingDevice
var dynamics: Array = []
var population_size: int = 0
var input_count: int = 0
var output_count: int = 0
var input_counts := PackedInt32Array()
var output_counts := PackedInt32Array()
var input_offsets := PackedInt32Array()
var output_offsets := PackedInt32Array()
var output_signals := PackedFloat32Array()
var enable_inspection: bool = false
var inspection_changes := PackedInt32Array([0, 0, 0])
var last_error: String = ""

var _brains: Array = []
var _buffers: Dictionary = {}
var _kernels: Dictionary = {}
var _sets: Dictionary = {}
var _strides := PackedInt32Array()
var _initial_strides := PackedInt32Array()
var _edge_offsets := PackedInt32Array()
var _initial_offsets := PackedInt32Array()
var _params_bytes := PackedByteArray()
var _total: int = 0
var _max_stride: int = 0
var _dimension := Vector3i.ZERO
var _iteration: int = 0
var _stages_first: Array = []
var _stages_second: Array = []
var _stages_all: Array = []

func _alloc(name: String, size_bytes: int, initial: PackedByteArray = PackedByteArray()) -> bool:
	if size_bytes <= 0 or size_bytes > MAX_STORAGE_BYTES:
		last_error = "GPU storage buffer %s requires %d bytes; single-buffer storage range exceeded." % [name, size_bytes]
		return false
	var rid: RID = GPUResourceAudit.create(rd, size_bytes, initial, name)
	if not rid.is_valid():
		last_error = "GPU allocation failed for %s (%d bytes)." % [name, size_bytes]
		return false
	_buffers[name] = rid
	return true

func _load_kernels() -> bool:
	for name in KERNELS:
		if not enable_inspection and name in ["inspect", "visual_lines"]:
			continue
		var path: String = PREFIX + name + ".glsl"
		var resource: RDShaderFile = load(path)
		if resource == null:
			last_error = "Missing packed compute kernel: " + path
			return false
		var spirv: RDShaderSPIRV = resource.get_spirv()
		if spirv == null:
			last_error = "SPIR-V missing for " + path
			return false
		var compile_error: String = spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
		if not compile_error.is_empty():
			last_error = path + ": " + compile_error
			return false
		var shader: RID = rd.shader_create_from_spirv(spirv)
		if not shader.is_valid():
			last_error = "Shader creation failed: " + path
			return false
		var pipeline: RID = rd.compute_pipeline_create(shader)
		if not pipeline.is_valid():
			GPUResourceAudit.release(rd, shader)
			last_error = "Compute pipeline creation failed: " + path
			return false
		_kernels[name] = {"shader": shader, "pipeline": pipeline}
	return true

func _uniform_set(name: String, names: Array, temporary: bool = false) -> RID:
	if not temporary and _sets.has(name):
		return _sets[name]
	var uniforms: Array[RDUniform] = []
	for slot in names.size():
		var value: Variant = names[slot]
		var rid: RID = value if value is RID else _buffers.get(String(value), RID())
		if not rid.is_valid():
			last_error = "Missing GPU binding %s in packed kernel %s." % [str(value), name]
			return RID()
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = slot
		uniform.add_id(rid)
		uniforms.append(uniform)
	var result: RID = rd.uniform_set_create(uniforms, _kernels[name]["shader"], 0)
	if not result.is_valid():
		last_error = "Descriptor creation failed: " + name
	elif not temporary:
		_sets[name] = result
	return result

func _release_sets() -> void:
	for rid in _sets.values():
		GPUResourceAudit.release(rd, rid)
	_sets.clear()


# All population stages now have zero push constants. The CSR reset force
# is derived from the existing iteration counter, so switching between
# compute pipelines never leaves an incompatible 4-byte push constant active.
func _stage(name: String, keys: Array, only_one: bool = false) -> Dictionary:
	return {"name": name, "keys": keys, "one": only_one}

# Each phase is dispatched once for the entire population,
# including the synaptic slot-vectorized kernel.
func _submit(stages: Array) -> bool:
	if stages.is_empty():
		return true
	var commands: int = rd.compute_list_begin()
	for i in stages.size():
		if i > 0:
			rd.compute_list_add_barrier(commands)
		var stage: Dictionary = stages[i]
		var label: String = stage["name"]
		var uniform: RID = _uniform_set(label, stage["keys"])
		if not uniform.is_valid():
			rd.compute_list_end()
			return false
		rd.compute_list_bind_compute_pipeline(commands, _kernels[label]["pipeline"])
		rd.compute_list_bind_uniform_set(commands, uniform, 0)
		var groups: int = 1 if bool(stage["one"]) else ceili(float(_total) / float(128 if label == "fused128" else WORKGROUP))
		if label == "edges":
			# X=neurons, Y=groups of four outgoing slots, Z=agents.
			# Covers synapse capacity without materializing dense N*N weights.
			rd.compute_list_dispatch(commands, groups, ceili(float(_max_stride) / float(EDGE_TILE)), population_size)
		else:
			rd.compute_list_dispatch(commands, groups, population_size, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	return true

# Single temporary dispatch, used for construction or inspection.
func _temporary_dispatch(name: String, keys: Array, groups_x: int, groups_y: int, pc: PackedByteArray = PackedByteArray()) -> bool:
	var uniform: RID = _uniform_set(name, keys, true)
	if not uniform.is_valid():
		return false
	var commands: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, _kernels[name]["pipeline"])
	rd.compute_list_bind_uniform_set(commands, uniform, 0)
	if not pc.is_empty():
		rd.compute_list_set_push_constant(commands, pc, pc.size())
	rd.compute_list_dispatch(commands, groups_x, groups_y, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	GPUResourceAudit.release(rd, uniform)
	return true

func _compute_offsets(strides: PackedInt32Array) -> PackedInt32Array:
	var offsets := PackedInt32Array()
	var total_slots: int = 0
	for value in strides:
		offsets.append(total_slots)
		total_slots += _total * int(value)
	return offsets

func _slot_count(strides: PackedInt32Array) -> int:
	var count: int = 0
	for stride in strides:
		count += _total * stride
	return count

func _pack_parameters() -> void:
	_params_bytes = PackedByteArray()
	for index in population_size:
		var meta: BrainDynamicsGPU = dynamics[index]
		meta._stride = _strides[index]
		meta._iteration = _iteration
		var bytes: PackedByteArray = meta._params()
		bytes.encode_u32(72, _initial_strides[index])
		bytes.encode_u32(76, index)
		bytes.resize(PARAM_BYTES)
		bytes.encode_u32(80, index * _total)
		bytes.encode_u32(84, _edge_offsets[index])
		bytes.encode_u32(88, _initial_offsets[index])
		var brain: Dictionary = _brains[index]
		bytes.encode_u32(92, int(brain["initial_search_radius_i"]))
		bytes.encode_u32(96, int(brain["initial_search_radius_j"]))
		bytes.encode_u32(100, int(brain["initial_search_radius_k"]))
		_params_bytes.append_array(bytes)

func _upload_parameters(delta: float = 0.0) -> bool:
	for index in population_size:
		var base: int = index * PARAM_BYTES
		_params_bytes.encode_u32(base + 16, _strides[index])
		_params_bytes.encode_u32(base + 20, _iteration)
		_params_bytes.encode_float(base + 24, delta)
		_params_bytes.encode_u32(base + 84, _edge_offsets[index])
	if rd.buffer_update(_buffers["parameters"], 0, _params_bytes.size(), _params_bytes) != OK:
		last_error = "Could not upload GPU neural parameters."
		return false
	return true

func _copy_initial(builder_data: Dictionary, index: int) -> bool:
	var offset: int = index * _total
	var copies: Array = [
		["neuron_buffer", "phenotype", 32],
		["roles_buffer", "roles", 4],
		["outgoing_buffer", "initial_counts", 4],
		["incoming_buffer", "initial_incoming", 4]
	]
	for entry in copies:
		var size_bytes: int = _total * int(entry[2])
		if rd.buffer_copy(builder_data[entry[0]], _buffers[entry[1]], 0, offset * int(entry[2]), size_bytes) != OK:
			last_error = "Failed to copy builder phenotype for brain %d." % (index + 1)
			return false
	return true

func initialize(brains: Array, with_inspection: bool = false) -> bool:
	if rd != null:
		last_error = "Already initialized; clear before reinitializing."
		return false
	last_error = ""
	_brains = brains.duplicate()
	population_size = brains.size()
	enable_inspection = with_inspection
	inspection_changes = PackedInt32Array([0, 0, 0])
	if population_size == 0 or not (brains[0] is Dictionary) or not brains[0].has("gpu_state"):
		last_error = "No GPU-resident brains provided."
		return false
	rd = brains[0]["gpu_state"]["rd"]
	_dimension = brains[0]["size"]
	_total = _dimension.x * _dimension.y * _dimension.z
	if _total <= 0:
		last_error = "Invalid neural dimensions."
		return false
	var input_maps: Array = []
	var output_maps: Array = []
	for i in population_size:
		var brain: Dictionary = brains[i]
		if not brain.has("gpu_state") or brain["gpu_state"]["rd"] != rd or brain["size"] != _dimension:
			last_error = "All brain buffers must belong to one RenderingDevice and share dimensions."
			return false
		var initial: int = int(brain["gpu_state"]["slot_stride"])
		_initial_strides.append(initial)
		if initial != int(brain["max_connections"]):
			last_error = "Brain %d: initial slots (%d) differ from N_max (%d)." % [i + 1, initial, int(brain["max_connections"])]
			return false
		_strides.append(initial)
		var meta := BrainDynamicsGPU.new()
		meta._brain = brain
		meta._size = _dimension
		meta._total = _total
		meta._build_rank_maps()
		var ni: int = meta._input_indices.size()
		var no: int = meta._output_indices.size()
		if ni <= 0 or no <= 0:
			last_error = "Brain %d lacks inputs or outputs." % (i + 1)
			return false
		input_offsets.append(input_count)
		output_offsets.append(output_count)
		input_counts.append(ni)
		output_counts.append(no)
		var input_reverse := PackedInt32Array()
		var output_reverse := PackedInt32Array()
		input_reverse.resize(_total)
		output_reverse.resize(_total)
		input_reverse.fill(-1)
		output_reverse.fill(-1)
		for channel in ni:
			input_reverse[meta._rank_to_spatial[meta._input_indices[channel]]] = input_count + channel
		for channel in no:
			output_reverse[meta._rank_to_spatial[meta._output_indices[channel]]] = output_count + channel
		input_count += ni
		output_count += no
		input_maps.append(input_reverse.to_byte_array())
		output_maps.append(output_reverse.to_byte_array())
		dynamics.append(meta)
	_max_stride = 0
	for s in _strides:
		_max_stride = maxi(_max_stride, s)
	_edge_offsets = _compute_offsets(_strides)
	_initial_offsets = _compute_offsets(_initial_strides)
	_pack_parameters()
	var neural_slots: int = _total * population_size
	var synapse_slots: int = _slot_count(_strides)
	var initial_slots: int = _slot_count(_initial_strides)
	var layout: Dictionary = {
		"parameters": population_size * PARAM_BYTES,
		"inputs": input_count * 4, "outputs": output_count * 4,
		"status": population_size * STATUS_WORDS * 4,
		"phenotype": neural_slots * 32, "roles": neural_slots * 4,
		"states": neural_slots * 48, "incoming": neural_slots * 4,
		"edges": synapse_slots * 16, "initial_edges": initial_slots * 16,
		"links": synapse_slots * 4, "emission": neural_slots * 4,
		"csr_offsets": neural_slots * 4, "csr_lengths": neural_slots * 4,
		"csr_indices": synapse_slots * 4, "csr_cursor": population_size * 4,
		"csr_dirty": population_size * 4,
		"counts": neural_slots * 4, "initial_counts": neural_slots * 4,
		"live_degrees": neural_slots * 4, "incoming_degrees": neural_slots * 4,
		"initial_incoming": neural_slots * 4, "heads": neural_slots * 4,
		"changes": population_size * CHANGE_WORDS * 4,
		"input_reverse": neural_slots * 4, "output_reverse": neural_slots * 4
	}
	if enable_inspection:
		layout["inspect_stats"] = neural_slots * 8
	for key in layout:
		if not _alloc(key, int(layout[key])):
			return false
	if not _load_kernels() or not _upload_parameters():
		return false
	for i in population_size:
		if not _copy_initial(brains[i]["gpu_state"], i):
			return false
		var offset: int = i * _total * 4
		var inputs_ok: bool = rd.buffer_update(_buffers["input_reverse"], offset, _total * 4, input_maps[i]) == OK
		var outputs_ok: bool = rd.buffer_update(_buffers["output_reverse"], offset, _total * 4, output_maps[i]) == OK
		if not inputs_ok or not outputs_ok:
			last_error = "Could not upload I/O reverse mapping."
			return false
	# Builder slices are retained in GPU storage and uploaded in batches.
	# One submission/synchronization per batch, NOT one per brain or slice.
	for batch in range(0, population_size, 8):
		if not _upload_initial_batch(brains, batch, mini(batch + 8, population_size)):
			return false
		var builder := BrainBuilderGPU.new()
		for i in range(batch, mini(batch + 8, population_size)):
			builder.release_gpu_state(brains[i])
	_build_plans()
	if not _initialize_static_graph():
		return false
	reset(true)
	return last_error.is_empty()

func _upload_initial_batch(brains: Array, first: int, end: int) -> bool:
	var commands: int = rd.compute_list_begin()
	var temporary_sets: Array[RID] = []
	rd.compute_list_bind_compute_pipeline(commands, _kernels["initial_upload"]["pipeline"])
	for agent in range(first, end):
		var gpu: Dictionary = brains[agent]["gpu_state"]
		for slice_index in _dimension.z:
			var uniform: RID = _uniform_set("initial_upload", ["parameters", gpu["slot_buffers"][slice_index],
				"initial_counts", "initial_edges"], true)
			if not uniform.is_valid():
				rd.compute_list_end()
				for rid in temporary_sets:
					GPUResourceAudit.release(rd, rid)
				return false
			temporary_sets.append(uniform)
			rd.compute_list_bind_uniform_set(commands, uniform, 0)
			var pc := PackedByteArray()
			pc.resize(16)
			pc.encode_u32(0, agent)
			pc.encode_u32(4, slice_index)
			pc.encode_u32(8, _initial_strides[agent])
			rd.compute_list_set_push_constant(commands, pc, pc.size())
			rd.compute_list_dispatch(commands, ceili(float(_dimension.x * _dimension.y) / float(WORKGROUP)), 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	for rid in temporary_sets:
		GPUResourceAudit.release(rd, rid)
	return true

func _build_plans() -> void:
	# All weights/topology were finalized at construction. No lifetime
	# synaptic writes, plasticity, modulation or CSR rebuild passes.
	_stages_first = [
		_stage("simple_prepare", ["parameters", "phenotype", "roles", "states", "inputs", "input_reverse", "emission"]),
		_stage("spmv", ["parameters", "csr_offsets", "csr_lengths", "csr_indices", "edges", "emission", "incoming"]),
		_stage("simple_state", ["parameters", "roles", "states", "incoming", "output_reverse", "outputs"])
	]
	_stages_second = []
	_stages_all = _stages_first

func advance(delta: float, inputs: PackedFloat32Array) -> bool:
	if rd == null or _buffers.is_empty():
		last_error = "The GPU neural population is not initialized."
		return false
	if inputs.size() != input_count:
		last_error = "Expected %d input channels; got %d." % [input_count, inputs.size()]
		return false
	_iteration += 1
	if not _upload_parameters(delta):
		return false
	var input_bytes: PackedByteArray = inputs.to_byte_array()
	if rd.buffer_update(_buffers["inputs"], 0, input_bytes.size(), input_bytes) != OK:
		last_error = "Input upload failed."
		return false
	# Exactly one command list and synchronization per neural iteration.
	if not _submit(_stages_all):
		return false
	output_signals = rd.buffer_get_data(_buffers["outputs"]).to_float32_array()
	if output_signals.size() != output_count:
		last_error = "GPU output readback incomplete."
		return false
	# No graph event can occur in Vsimple: weights and topology are immutable.
	if enable_inspection:
		inspection_changes.fill(0)
	return true

# Build the receiver-indexed sparse matrix once; topology never changes afterward.
func _initialize_static_graph() -> bool:
	if not _submit([
		_stage("simple_reset", ["parameters", "states", "incoming", "changes", "heads", "counts", "live_degrees", "incoming_degrees", "initial_incoming"]),
		_stage("restore", ["parameters", "initial_edges", "initial_counts", "edges", "counts", "live_degrees", "heads", "links", "changes"]),
		_stage("csr_init", ["parameters", "changes", "csr_cursor", "csr_dirty"], true),
		_stage("csr_reserve", ["parameters", "incoming_degrees", "csr_cursor", "csr_dirty", "csr_offsets", "changes"]),
		_stage("csr_fill", ["parameters", "heads", "links", "edges", "incoming_degrees", "csr_offsets", "csr_lengths", "csr_indices", "csr_dirty", "changes"]),
		_stage("status", ["parameters", "changes", "status"], true)
	]):
		return false
	var status: PackedInt32Array = rd.buffer_get_data(_buffers["status"], 0, population_size * STATUS_WORDS * 4).to_int32_array()
	if status.size() != population_size * STATUS_WORDS:
		last_error = "Incomplete initial graph integrity readback."
		return false
	for index in population_size:
		if status[STATUS_WORDS * index + 3] != 0:
			last_error = "Invalid initial synapse index in brain %d." % (index + 1)
			return false
	# The initial snapshot, linked lists, and CSR construction scratch are
	# discarded. The ELLPACK weights and CSR graph are the sole lifetime graph.
	_release_sets()
	for key in ["initial_edges", "initial_counts", "initial_incoming", "heads", "links", "changes", "status", "csr_cursor", "csr_dirty"]:
		GPUResourceAudit.release(rd, _buffers[key])
		_buffers.erase(key)
	return true

func reset(reset_neural: bool = true) -> void:
	if rd == null or _buffers.is_empty():
		return
	last_error = ""
	if reset_neural:
		_iteration = 0
		inspection_changes.fill(0)
		if not _upload_parameters():
			return
		if not _submit([_stage("simple_state_reset", ["parameters", "states", "incoming"])]):
			return
	output_signals.resize(output_count)
	output_signals.fill(0.0)

func get_brain_outputs(brain_index: int) -> PackedFloat32Array:
	if brain_index < 0 or brain_index >= population_size or output_signals.size() != output_count:
		return PackedFloat32Array()
	return output_signals.slice(output_offsets[brain_index], output_offsets[brain_index] + output_counts[brain_index])

func _read_neurons(key: String, index: int, bytes_per_neuron: int) -> PackedByteArray:
	return rd.buffer_get_data(_buffers[key], index * _total * bytes_per_neuron, _total * bytes_per_neuron)

func read_visual_signals(brain_index: int) -> PackedFloat32Array:
	if rd == null or brain_index < 0 or brain_index >= population_size:
		return PackedFloat32Array()
	var data: PackedFloat32Array = _read_neurons("states", brain_index, 48).to_float32_array()
	if data.size() != 12 * _total:
		return PackedFloat32Array()
	var result := PackedFloat32Array()
	result.resize(_total)
	for spatial in _total:
		result[spatial] = data[12 * spatial + 3]
	return result

func selected_statistics(agent_index: int) -> Dictionary:
	if rd == null or agent_index < 0 or agent_index >= dynamics.size():
		return {}
	var meta: BrainDynamicsGPU = dynamics[agent_index]
	var data: PackedFloat32Array = _read_neurons("states", agent_index, 48).to_float32_array()
	if data.size() != _total * 12:
		return {}
	var states := PackedFloat32Array()
	var signals := PackedFloat32Array()
	states.resize(_total)
	signals.resize(_total)
	for rank in _total:
		var spatial: int = meta._rank_to_spatial[rank]
		states[rank] = data[12 * spatial + 4]
		signals[rank] = data[12 * spatial + 3]
	var inputs := PackedFloat32Array()
	inputs.resize(meta._input_indices.size())
	for i in inputs.size():
		inputs[i] = states[meta._input_indices[i]]
	var outputs := PackedFloat32Array()
	outputs.resize(meta._output_indices.size())
	for i in outputs.size():
		var public_id: int = meta._output_indices[i]
		var spatial: int = meta._rank_to_spatial[public_id]
		outputs[i] = output_signals[output_offsets[agent_index] + i] if output_signals.size() == output_count else (signals[public_id] if data[12 * spatial + 11] > 0.5 else 0.0)
	return {"neuron_states_used": states, "normalized_signals": signals,
		"input_neuron_indices": meta._input_indices, "input_signals_used": inputs, "output_signals": outputs}

func inspection_snapshot(brain_index: int = 0) -> Dictionary:
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size:
		return {}
	var pc := PackedByteArray()
	pc.resize(4)
	pc.encode_u32(0, brain_index)
	if not _temporary_dispatch("inspect", ["parameters", "edges", "counts", "inspect_stats"],
		ceili(float(_total) / float(WORKGROUP)), 1, pc):
		return {}
	return {"states": _read_neurons("states", brain_index, 48).to_float32_array(),
		"incoming_degrees": _read_neurons("incoming_degrees", brain_index, 4).to_int32_array(),
		"live_degrees": _read_neurons("live_degrees", brain_index, 4).to_int32_array(),
		"synapse_stats": _read_neurons("inspect_stats", brain_index, 8).to_float32_array()}

func inspection_lines(brain_index: int, selected_spatial: int, capacity: int) -> Dictionary:
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size or capacity <= 0 or capacity > MAX_STORAGE_BYTES / 16:
		return {}
	var count_rid: RID = GPUResourceAudit.create(rd, 4)
	var lines_rid: RID = GPUResourceAudit.create(rd, capacity * 16)
	if not count_rid.is_valid() or not lines_rid.is_valid():
		GPUResourceAudit.release(rd, count_rid)
		GPUResourceAudit.release(rd, lines_rid)
		return {}
	var zero := PackedByteArray()
	zero.resize(4)
	rd.buffer_update(count_rid, 0, 4, zero)
	var pc := PackedByteArray()
	pc.resize(16)
	pc.encode_u32(0, brain_index)
	pc.encode_u32(4, selected_spatial if selected_spatial >= 0 else _total)
	pc.encode_u32(8, capacity)
	var ok: bool = _temporary_dispatch("visual_lines", ["parameters", "edges", "counts", count_rid, lines_rid],
		ceili(float(_total) / float(WORKGROUP)), 1, pc)
	var answer: Dictionary = {}
	if ok:
		var count_data: PackedInt32Array = rd.buffer_get_data(count_rid).to_int32_array()
		if not count_data.is_empty():
			var count: int = mini(count_data[0], capacity)
			answer = {"data": rd.buffer_get_data(lines_rid, 0, count * 16).to_float32_array() if count > 0 else PackedFloat32Array(),
				"reported": count_data[0], "count": count}
	GPUResourceAudit.release(rd, count_rid)
	GPUResourceAudit.release(rd, lines_rid)
	return answer

func inspection_synapses(brain_index: int = 0) -> Array:
	var result: Array = []
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size:
		return result
	var counts: PackedInt32Array = _read_neurons("counts", brain_index, 4).to_int32_array()
	var byte_offset: int = _edge_offsets[brain_index] * 16
	var edges: PackedByteArray = rd.buffer_get_data(_buffers["edges"], byte_offset, _total * _strides[brain_index] * 16)
	for source in _total:
		for slot in counts[source]:
			var at: int = (slot * _total + source) * 16
			var target: int = edges.decode_s32(at)
			if target >= 0:
				result.append([source, target, edges.decode_float(at + 4), edges.decode_float(at + 8)])
	result.sort_custom(func(a, b):
		if int(a[0]) != int(b[0]):
			return int(a[0]) < int(b[0])
		return int(a[1]) < int(b[1]))
	return result

func inspection_incoming_degrees(brain_index: int = 0) -> PackedInt32Array:
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size:
		return PackedInt32Array()
	return _read_neurons("incoming_degrees", brain_index, 4).to_int32_array()

func clear() -> void:
	if rd == null:
		return
	_release_sets()
	for rid in _buffers.values():
		GPUResourceAudit.release(rd, rid)
	_buffers.clear()
	for entry in _kernels.values():
		GPUResourceAudit.release(rd, entry["pipeline"])
		GPUResourceAudit.release(rd, entry["shader"])
	_kernels.clear()
	var builder := BrainBuilderGPU.new()
	for brain in _brains:
		if brain is Dictionary and brain.has("gpu_state"):
			builder.release_gpu_state(brain)
	dynamics.clear()
	rd.free()
	rd = null
	_brains.clear()
	_stages_first.clear()
	_stages_second.clear()
	_stages_all.clear()
	_strides = PackedInt32Array()
	_initial_strides = PackedInt32Array()
	_edge_offsets = PackedInt32Array()
	_initial_offsets = PackedInt32Array()
	_params_bytes = PackedByteArray()
	input_counts = PackedInt32Array()
	output_counts = PackedInt32Array()
	input_offsets = PackedInt32Array()
	output_offsets = PackedInt32Array()
	output_signals = PackedFloat32Array()
	population_size = 0
	input_count = 0
	output_count = 0
	_total = 0
	_iteration = 0
