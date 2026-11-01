class_name BrainPopulationGPU
extends RefCounted

# Generic batched neural dynamics. The environment supplies a packed input vector
# and receives a packed output vector for all brains. This class has no knowledge
# of positions, targets, collisions, physics, simulation numbers or fitness.
# Storage buffers for neural data and synapses are physically independent per brain.
# All GPU resources are accessed on the thread that created the RenderingDevice.
const LOCAL_SIZE: int = 64
const MAX_BUFFER_BYTES: int = 2147483647
# Six counters are required by neural integrity checks and Viewer inspection:
# deaths, rescued, structural, invalid, capacity failures, reclaimed.
const STATUS_WORDS: int = 6
const SHADER_PREFIX: String = "res://shaders/brain_population_"
const INSPECTION_SHADERS := ["inspect", "visual_lines"]
const SHADERS := ["initial_upload", "reset", "restore", "prepare", "modulation",
	"edges", "gather", "state", "select", "capacity", "homeostasis", "structural",
	"clear_heads", "resize"]

# Optional visualization support. Off by default: agent simulations allocate no
# inspection buffers, compile no inspection shaders and perform no extra readbacks.
var enable_inspection: bool = false
var inspection_changes: PackedInt32Array = PackedInt32Array([0, 0, 0])

var rd: RenderingDevice
var dynamics: Array = []
var _brains: Array = []
var _brain_buffers: Array[Dictionary] = []
var _control: Dictionary = {}
var _pipelines: Dictionary = {}
var _uniform_cache: Dictionary = {}
var _strides: PackedInt32Array = PackedInt32Array()
var _initial_strides: PackedInt32Array = PackedInt32Array()
var _total: int = 0
var _dimension: Vector3i = Vector3i.ZERO
var _iteration: int = 0
var population_size: int = 0
var input_count: int = 0
var output_count: int = 0
var input_counts: PackedInt32Array = PackedInt32Array()
var output_counts: PackedInt32Array = PackedInt32Array()
var input_offsets: PackedInt32Array = PackedInt32Array()
var output_offsets: PackedInt32Array = PackedInt32Array()
var output_signals: PackedFloat32Array = PackedFloat32Array()
var last_error: String = ""
# Immutable dispatch plans. Resources are resolved at submission time so a
# per-brain synaptic resize never leaves stale RIDs in these plans.
var _advance_first_stages: Array = []
var _advance_second_stages: Array = []
# Cached 80-byte per-brain parameters. Only iteration/dt (and stride on growth) change.
var _packed_parameters: PackedByteArray = PackedByteArray()

func _allocate(key: String, bytes: int, brain_index: int = -1, initial: PackedByteArray = PackedByteArray()) -> bool:
	if bytes <= 0 or bytes > MAX_BUFFER_BYTES:
		last_error = "Brain %d: buffer %s requires %d bytes; limit %d per buffer." % [brain_index + 1, key, bytes, MAX_BUFFER_BYTES]
		return false
	var rid: RID = GPUResourceAudit.create(rd, bytes, initial, key)
	if not rid.is_valid():
		last_error = "Could not allocate %s (%d MiB) for brain %d." % [key, ceili(float(bytes) / 1048576.0), brain_index + 1]
		return false
	if brain_index < 0:
		_control[key] = rid
	else:
		_brain_buffers[brain_index][key] = rid
	return true

func _load_pipelines() -> bool:
	for shader_name in (SHADERS + INSPECTION_SHADERS if enable_inspection else SHADERS):
		var path: String = SHADER_PREFIX + shader_name + ".glsl"
		var resource: RDShaderFile = load(path)
		if resource == null:
			last_error = "Neural population shader missing: " + path
			return false
		var spirv: RDShaderSPIRV = resource.get_spirv()
		if spirv == null:
			last_error = "No SPIR-V for " + path
			return false
		var error_text: String = spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
		if not error_text.is_empty():
			last_error = path + ": " + error_text
			return false
		var shader: RID = rd.shader_create_from_spirv(spirv)
		if not shader.is_valid():
			last_error = "Could not create shader: " + path
			return false
		var pipeline: RID = rd.compute_pipeline_create(shader)
		if not pipeline.is_valid():
			GPUResourceAudit.release(rd, shader)
			last_error = "Could not create pipeline: " + path
			return false
		_pipelines[shader_name] = {"shader": shader, "pipeline": pipeline}
	return true

func _resolve_buffer(agent_id: int, key: Variant) -> RID:
	if key is RID:
		return key
	var label: String = String(key)
	if label.begins_with("shared:"):
		return _control.get(label.substr(7), RID())
	return _brain_buffers[agent_id].get(label, RID())

func _bindings(shader_name: String, agent_id: int, keys: Array) -> RID:
	var cache_key: String = str(agent_id) + "|" + shader_name
	# All ordinary-phase bindings remain unchanged until a growth operation,
	# which invalidates only this brain's descriptors. Reuse the cached set
	# without allocating arrays/stringifying a dozen RIDs on every step.
	var temporary_bindings: bool = shader_name == "initial_upload" or shader_name == "resize" or (enable_inspection and shader_name == "visual_lines")
	if not temporary_bindings and _uniform_cache.has(cache_key):
		return _uniform_cache[cache_key]
	var resources: Array[RID] = []
	for key in keys:
		var rid: RID = _resolve_buffer(agent_id, key)
		if not rid.is_valid():
			last_error = "Invalid binding %s in %s, brain %d." % [str(key), shader_name, agent_id + 1]
			return RID()
		resources.append(rid)
		if temporary_bindings:
			cache_key += ":" + str(rid.get_id())
	if _uniform_cache.has(cache_key):
		return _uniform_cache[cache_key]
	var uniforms: Array[RDUniform] = []
	for slot in resources.size():
		var uniform: RDUniform = RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = slot
		uniform.add_id(resources[slot])
		uniforms.append(uniform)
	var result: RID = rd.uniform_set_create(uniforms, _pipelines[shader_name]["shader"], 0)
	if result.is_valid():
		_uniform_cache[cache_key] = result
	else:
		last_error = "Could not bind %s for brain %d." % [shader_name, agent_id + 1]
	return result

func _invalidate_uniforms(agent_id: int = -1) -> void:
	# A stride expansion must not invalidate descriptor sets for other
	# brains. Cache keys begin with the physical buffer owner ID.
	if agent_id < 0:
		for rid in _uniform_cache.values():
			if rid.is_valid():
				GPUResourceAudit.release(rd, rid)
		_uniform_cache.clear()
	else:
		var prefix: String = str(agent_id) + "|"
		for key in _uniform_cache.keys():
			if String(key).begins_with(prefix):
				var rid: RID = _uniform_cache[key]
				if rid.is_valid():
					GPUResourceAudit.release(rd, rid)
				_uniform_cache.erase(key)
	# The persistent task tuples own no resources: clearing their borrowed RID
	# references prevents reuse after a brain's synaptic stride is enlarged.
	for stages in [_advance_first_stages, _advance_second_stages]:
		for stage in stages:
			for task in stage:
				if agent_id < 0 or int(task[1]) == agent_id:
					task[5] = RID()

# Command tuple: name, agent, bound resources, count, PC, cached uniform RID,
# cached workgroup count. Persistent tuples are invalidated on stride growth.
func _task(shader_name: String, agent_id: int, resources: Array, count: int, pc: PackedByteArray = PackedByteArray()) -> Array:
	if pc.is_empty():
		pc = _id_pc(agent_id)
	return [shader_name, agent_id, resources, count, pc, RID(), ceili(float(count) / float(LOCAL_SIZE))]

func _id_pc(agent_id: int) -> PackedByteArray:
	var bytes: PackedByteArray = PackedByteArray()
	bytes.resize(4)
	bytes.encode_u32(0, agent_id)
	return bytes

# Homeostatic candidate selection uses exactly the first-pass V_init radii.
# Embed them in this stage's push constants instead of changing the common
# 80-byte parameter layout (which is shared by every other shader).
func _select_pc(agent_id: int) -> PackedByteArray:
	var brain: Dictionary = _brains[agent_id]
	var bytes: PackedByteArray = PackedByteArray()
	bytes.resize(16)
	bytes.encode_u32(0, agent_id)
	bytes.encode_u32(4, int(brain["initial_search_radius_i"]))
	bytes.encode_u32(8, int(brain["initial_search_radius_j"]))
	bytes.encode_u32(12, int(brain["initial_search_radius_k"]))
	return bytes

func _capacity_pc(stage: int, agent_id: int) -> PackedByteArray:
	var bytes: PackedByteArray = PackedByteArray()
	bytes.resize(12)
	bytes.encode_u32(0, stage)
	bytes.encode_u32(4, agent_id)
	# One preflight error flag per brain follows the packed status records.
	bytes.encode_u32(8, population_size * STATUS_WORDS)
	return bytes

func _resize_pc(stride: int, agent_id: int) -> PackedByteArray:
	var bytes: PackedByteArray = PackedByteArray()
	bytes.resize(8)
	bytes.encode_u32(0, stride)
	bytes.encode_u32(4, agent_id)
	return bytes

func _population_stage(shader_name: String, keys: Array, count: int, mode: int = -1) -> Array:
	var tasks: Array = []
	for index in population_size:
		var pc: PackedByteArray = _capacity_pc(mode, index) if mode >= 0 else (_select_pc(index) if shader_name == "select" else _id_pc(index))
		tasks.append(_task(shader_name, index, keys, count, pc))
	return tasks

func _submit(stages: Array) -> bool:
	var command_list: int = rd.compute_list_begin()
	var bound_pipeline: String = ""
	for stage_index in stages.size():
		if stage_index > 0:
			rd.compute_list_add_barrier(command_list)
		var stage: Array = stages[stage_index]
		if stage.is_empty():
			continue
		# Each stage has one shader for every brain; resolve/bind pipeline once.
		var label: String = stage[0][0]
		if bound_pipeline != label:
			rd.compute_list_bind_compute_pipeline(command_list, _pipelines[label]["pipeline"])
			bound_pipeline = label
		for task in stage:
			var uniform_set: RID = task[5]
			if not uniform_set.is_valid():
				uniform_set = _bindings(label, int(task[1]), task[2])
				if not uniform_set.is_valid():
					rd.compute_list_end()
					return false
				task[5] = uniform_set
			rd.compute_list_bind_uniform_set(command_list, uniform_set, 0)
			var pc: PackedByteArray = task[4]
			rd.compute_list_set_push_constant(command_list, pc, pc.size())
			rd.compute_list_dispatch(command_list, int(task[6]), 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	return true

func _init_packed_parameters() -> void:
	# Freeze genome-derived constants once per population. Repacking from the
	# genome on every neural iteration was ~0.7 ms of CPU-side work in v12.
	_packed_parameters = PackedByteArray()
	for index in population_size:
		var meta: BrainDynamicsGPU = dynamics[index]
		meta._iteration = 0
		meta._stride = _strides[index]
		var packed: PackedByteArray = meta._params()
		packed.encode_u32(72, _initial_strides[index])
		packed.encode_u32(76, index)
		_packed_parameters.append_array(packed)

func _params_upload(delta_seconds: float = 0.0) -> bool:
	# No genome access, allocation or logarithms in this hot loop.
	for index in population_size:
		var meta: BrainDynamicsGPU = dynamics[index]
		meta._iteration = _iteration
		meta._stride = _strides[index]
		var base: int = index * 80
		_packed_parameters.encode_u32(base + 16, _strides[index])
		_packed_parameters.encode_u32(base + 20, _iteration)
		_packed_parameters.encode_float(base + 24, delta_seconds)
	if rd.buffer_update(_control["parameters"], 0, _packed_parameters.size(), _packed_parameters) != OK:
		last_error = "Parameter buffer update failed."
		return false
	return true

func _copy_builder_data(index: int, builder_state: Dictionary) -> bool:
	var n4: int = _total * 4
	var jobs: Array = [
		[builder_state["neuron_buffer"], "phenotype", n4 * 8],
		[builder_state["roles_buffer"], "roles", n4],
		[builder_state["outgoing_buffer"], "initial_counts", n4],
		[builder_state["incoming_buffer"], "initial_incoming", n4]
	]
	for job in jobs:
		var source: RID = job[0]
		var destination: RID = _brain_buffers[index][job[1]]
		if rd.buffer_copy(source, destination, 0, 0, int(job[2])) != OK:
			last_error = "Builder buffer copy failed for brain %d (%s)." % [index + 1, String(job[1])]
			return false
	return true

func initialize(brains: Array, with_inspection: bool = false) -> bool:
	if rd != null:
		last_error = "BrainPopulationGPU is already initialized; call clear() before reuse."
		return false
	# Receives GPU-resident phenotypes generated by BrainBuilderGPU.
	# This operation consumes their builder-side buffers after copying them.
	last_error = ""
	enable_inspection = with_inspection
	inspection_changes = PackedInt32Array([0, 0, 0])
	population_size = brains.size()
	if population_size == 0 or not (brains[0] is Dictionary) or not brains[0].has("gpu_state"):
		last_error = "Incomplete GPU brain population."
		return false
	rd = brains[0]["gpu_state"]["rd"]
	_brains = brains.duplicate()
	_dimension = brains[0]["size"]
	_total = _dimension.x * _dimension.y * _dimension.z
	var plane: int = _dimension.x * _dimension.y
	var input_reverse_tables: Array = []
	var output_reverse_tables: Array = []
	var total_inputs: int = 0
	var total_outputs: int = 0
	for index in population_size:
		var brain: Dictionary = brains[index]
		if not brain.has("gpu_state") or brain["gpu_state"]["rd"] != rd or brain["size"] != _dimension:
			last_error = "All brains must have equal dimensions and share the RenderingDevice."
			return false
		var initial_stride: int = int(brain["gpu_state"]["slot_stride"])
		_initial_strides.append(initial_stride)
		_strides.append(initial_stride + maxi(64, ceili(float(initial_stride) * 0.5)))
		var meta: BrainDynamicsGPU = BrainDynamicsGPU.new()
		meta._brain = brain
		meta._size = _dimension
		meta._total = _total
		meta._build_rank_maps()
		var ni: int = meta._input_indices.size()
		var no: int = meta._output_indices.size()
		if ni <= 0 or no <= 0:
			last_error = "Brain %d has no input or output neurons." % (index + 1)
			return false
		input_counts.append(ni)
		output_counts.append(no)
		input_offsets.append(total_inputs)
		output_offsets.append(total_outputs)
		total_inputs += ni
		total_outputs += no
		# For each spatial neuron, store the global stream position or -1.
		# Reverse maps avoid a separate GPU scatter/gather dispatch per brain.
		var reverse_inputs := PackedInt32Array()
		var reverse_outputs := PackedInt32Array()
		reverse_inputs.resize(_total)
		reverse_outputs.resize(_total)
		reverse_inputs.fill(-1)
		reverse_outputs.fill(-1)
		for channel in ni:
			var spatial: int = meta._rank_to_spatial[meta._input_indices[channel]]
			reverse_inputs[spatial] = input_offsets[index] + channel
		for channel in no:
			var spatial: int = meta._rank_to_spatial[meta._output_indices[channel]]
			reverse_outputs[spatial] = output_offsets[index] + channel
		input_reverse_tables.append(reverse_inputs.to_byte_array())
		output_reverse_tables.append(reverse_outputs.to_byte_array())
		dynamics.append(meta)
		_brain_buffers.append({})
	input_count = total_inputs
	output_count = total_outputs
	for item in [["parameters", population_size * 80], ["inputs", total_inputs * 4],
			["outputs", total_outputs * 4], ["required", population_size * 4],
			# Packed integrity counters followed by one preflight flag per brain.
			["status", population_size * (STATUS_WORDS + 1) * 4]]:
		if not _allocate(String(item[0]), int(item[1])):
			return false
	for index in population_size:
		var slots: int = _total * _strides[index]
		var original_slots: int = _total * _initial_strides[index]
		var sizes: Dictionary = {
			"phenotype": _total * 32, "roles": _total * 4,
			"states": _total * 48, "modulation": _total * 8,
			"pending": _total * 4, "incoming": _total * 4,
			"edges": slots * 16, "initial_edges": original_slots * 16,
			"links": slots * 4, "transmissions": slots * 8,
			"counts": _total * 4, "initial_counts": _total * 4,
			"live_degrees": _total * 4, "incoming_degrees": _total * 4,
			"initial_incoming": _total * 4,
			"heads": _total * 4, "changes": STATUS_WORDS * 4,
			"free_heads": _total * 4, "free_counts": _total * 4,
			"free_next": slots * 4, "dirty": _total * 4,
			"rescue_flags": _total * 4, "rescue_sources": _total * 4,
			"demand": _total * 4,
			"input_reverse": _total * 4, "output_reverse": _total * 4
		}
		if enable_inspection:
			sizes["inspect_stats"] = _total * 8
		for key in sizes:
			var label: String = String(key)
			var initial: PackedByteArray = PackedByteArray()
			if label == "input_reverse":
				initial = input_reverse_tables[index]
			elif label == "output_reverse":
				initial = output_reverse_tables[index]
			if not _allocate(label, int(sizes[key]), index, initial):
				return false
	_init_packed_parameters()
	if not _load_pipelines() or not _params_upload():
		return false
	for index in population_size:
		if not _copy_builder_data(index, brains[index]["gpu_state"]):
			return false
	rd.submit()
	rd.sync()
	for batch_start in range(0, population_size, 10):
		var upload: Array = []
		var batch_end: int = mini(batch_start + 10, population_size)
		for index in range(batch_start, batch_end):
			var builder_state: Dictionary = brains[index]["gpu_state"]
			for slice_index in _dimension.z:
				var pc: PackedByteArray = PackedByteArray()
				pc.resize(16)
				pc.encode_u32(0, index)
				pc.encode_u32(4, slice_index)
				pc.encode_u32(8, _initial_strides[index])
				upload.append(_task("initial_upload", index,
					["shared:parameters", builder_state["slot_buffers"][slice_index], "initial_counts", "initial_edges"], plane, pc))
		if not _submit([upload]):
			last_error = "Initial synaptic upload failed: " + last_error
			return false
		var builder: BrainBuilderGPU = BrainBuilderGPU.new()
		for index in range(batch_start, batch_end):
			_invalidate_uniforms(index)
			builder.release_gpu_state(brains[index])
	reset(true)
	if not last_error.is_empty():
		return false
	_build_advance_plans()
	return true

func _build_advance_plans() -> void:
	# Task tuples contain only stable names, counts and push constants. The
	# actual uniform sets are resolved (and possibly regenerated) by _submit().
	var n: int = _total
	var capacity_keys: Array = ["shared:parameters", "counts", "rescue_sources", "roles", "demand", "shared:required", "changes", "shared:status", "free_counts"]
	_advance_first_stages = [
		_population_stage("prepare", ["shared:parameters", "phenotype", "roles", "states", "shared:inputs", "changes", "rescue_flags", "incoming", "input_reverse"], n),
		_population_stage("modulation", ["shared:parameters", "states", "phenotype", "modulation"], n),
		_population_stage("edges", ["shared:parameters", "phenotype", "roles", "states", "modulation", "edges", "counts", "incoming_degrees", "live_degrees", "transmissions", "changes", "rescue_flags", "dirty"], n),
		_population_stage("gather", ["shared:parameters", "heads", "links", "transmissions", "incoming", "edges", "free_heads", "free_next", "free_counts", "dirty", "changes"], n),
		_population_stage("state", ["shared:parameters", "phenotype", "roles", "states", "modulation", "incoming", "output_reverse", "shared:outputs"], n),
		_population_stage("select", ["shared:parameters", "roles", "states", "incoming_degrees", "rescue_flags", "rescue_sources", "demand", "shared:required"], n)
	]
	for mode in [1, 2]:
		_advance_first_stages.append(_population_stage("capacity", capacity_keys, n, mode))
	_advance_second_stages = [
		_population_stage("homeostasis", ["shared:parameters", "roles", "rescue_sources", "edges", "counts", "incoming_degrees", "live_degrees", "heads", "links", "changes", "free_heads", "free_next", "free_counts"], n),
		_population_stage("structural", ["shared:parameters", "roles", "states", "edges", "counts", "incoming_degrees", "live_degrees", "heads", "links", "changes", "free_heads", "free_next", "free_counts"], n),
		# Capacity mode 3 only snapshots changes[0..13] when n == 0.
		# Dispatch one workgroup per brain: no GPU neuron state is updated here.
		_population_stage("capacity", capacity_keys, 1, 3)
	]

func advance(delta: float, inputs: PackedFloat32Array) -> bool:
	if rd == null or _control.is_empty():
		last_error = "Uninitialized brain population."
		return false
	if inputs.size() != input_count:
		last_error = "Expected %d input floats, received %d." % [input_count, inputs.size()]
		return false
	_iteration += 1
	if not _params_upload(delta):
		return false
	var input_bytes: PackedByteArray = inputs.to_byte_array()
	if rd.buffer_update(_control["inputs"], 0, input_bytes.size(), input_bytes) != OK:
		last_error = "Input buffer update failed."
		return false
	# The first submission computes neural signals and required synaptic capacity.
	if not _submit(_advance_first_stages):
		return false
	var required: PackedInt32Array = rd.buffer_get_data(_control["required"]).to_int32_array()
	# The packed preflight flags are necessary to identify invalid synaptic lists
	# before resizing or executing the second neural phase.
	var preflight: PackedInt32Array = rd.buffer_get_data(_control["status"],
		population_size * STATUS_WORDS * 4, population_size * 4).to_int32_array()
	if required.size() != population_size or preflight.size() != population_size:
		last_error = "Incomplete capacity readback."
		return false
	var grow_agents: Array[int] = []
	for index in population_size:
		if preflight[index] != 0:
			last_error = "Invalid synaptic target/list in brain %d." % (index + 1)
			return false
		if required[index] > _strides[index]:
			grow_agents.append(index)
	# Prune already removed dead edges from inbound lists on the GPU.
	# No full compact pass or third submission is needed on ordinary steps.
	for index in grow_agents:
		if not _grow_one(index, required[index]):
			return false
	if not _submit(_advance_second_stages):
		return false
	output_signals = rd.buffer_get_data(_control["outputs"]).to_float32_array()
	var status: PackedInt32Array = rd.buffer_get_data(_control["status"],
		0, population_size * STATUS_WORDS * 4).to_int32_array()
	if output_signals.size() != output_count or status.size() != population_size * STATUS_WORDS:
		last_error = "Incomplete output/status readback."
		return false
	for index in population_size:
		var at: int = index * STATUS_WORDS
		if status[at] != status[at + 5]:
			last_error = "Synaptic reclamation mismatch in brain %d: %d deaths, %d unlinked slots." % [
				index + 1, status[at], status[at + 5]]
			return false
		if status[at + 3] != 0 or status[at + 4] != 0:
			last_error = "Synaptic integrity or capacity failure in brain %d." % (index + 1)
			return false
	if enable_inspection:
		# Viewer counts are already included in the required integrity readback.
		inspection_changes[0] = status[0]
		inspection_changes[1] = status[1]
		inspection_changes[2] = status[2]
	return true

func _grow_one(agent_id: int, required_stride: int) -> bool:
	var old_stride: int = _strides[agent_id]
	var next_stride: int = maxi(required_stride, old_stride * 2)
	var slots: int = _total * next_stride
	var replacements: Dictionary = {}
	for key in ["edges", "links", "transmissions", "free_next"]:
		var slot_bytes: int = 16 if key == "edges" else (8 if key == "transmissions" else 4)
		var byte_size: int = slots * slot_bytes
		if byte_size > MAX_BUFFER_BYTES or byte_size <= 0:
			for rid in replacements.values():
				GPUResourceAudit.release(rd, rid)
			last_error = "Brain %d cannot expand %s to %d bytes (2 GiB buffer limit)." % [agent_id + 1, key, byte_size]
			return false
		var rid: RID = GPUResourceAudit.create(rd, byte_size, PackedByteArray(), key + "-growth")
		if not rid.is_valid():
			for previous_rid in replacements.values():
				GPUResourceAudit.release(rd, previous_rid)
			last_error = "Synaptic buffer allocation failed for brain %d." % (agent_id + 1)
			return false
		replacements[key] = rid
	if not _submit([
		[_task("clear_heads", agent_id, ["shared:parameters", "heads"], _total)],
		[_task("resize", agent_id, ["shared:parameters", "edges", "counts", "heads", replacements["edges"], replacements["links"], "free_heads", "free_counts"], _total, _resize_pc(next_stride, agent_id))]]):
		_invalidate_uniforms(agent_id)
		for rid in replacements.values():
			GPUResourceAudit.release(rd, rid)
		return false
	# The new transmissions buffer needs no copy; edges overwrites all active
	# slots before gather in the next step. Old immutable initial_edges stays put.
	_invalidate_uniforms(agent_id)
	for key in replacements:
		GPUResourceAudit.release(rd, _brain_buffers[agent_id][key])
		_brain_buffers[agent_id][key] = replacements[key]
	_strides[agent_id] = next_stride
	if not _params_upload():
		return false
	return true

func reset(reset_neural: bool = true) -> void:
	if rd == null or _control.is_empty():
		return
	last_error = ""
	if reset_neural:
		if enable_inspection:
			inspection_changes.fill(0)
		_iteration = 0
		if not _params_upload():
			return
		var capacity_keys: Array = ["shared:parameters", "counts", "rescue_sources", "roles", "demand", "shared:required", "changes", "shared:status", "free_counts"]
		if not _submit([
			_population_stage("reset", ["shared:parameters", "states", "modulation", "incoming", "pending", "changes", "rescue_flags", "rescue_sources", "heads", "counts", "live_degrees", "incoming_degrees", "initial_incoming", "free_heads", "free_counts", "dirty"], _total),
			_population_stage("restore", ["shared:parameters", "initial_edges", "initial_counts", "edges", "counts", "live_degrees", "heads", "links", "changes"], _total),
			# The reset snapshot uses precisely the same n == 0-only kernel path.
			_population_stage("capacity", capacity_keys, 1, 3)]):
			last_error = "Synaptic reset failed: " + last_error
			return
		var status_check: PackedInt32Array = rd.buffer_get_data(_control["status"],
			0, population_size * STATUS_WORDS * 4).to_int32_array()
		for agent_id in population_size:
			if status_check.size() != population_size * STATUS_WORDS or status_check[agent_id * STATUS_WORDS + 3] != 0:
				last_error = "Invalid initial synaptic target in brain %d." % (agent_id + 1)
				return
	# Inputs and outputs are controlled by the calling environment. Their
	# per-step contents are replaced in advance(), not by the neural reset.
	output_signals = PackedFloat32Array()
	output_signals.resize(output_count)

func get_brain_outputs(brain_index: int) -> PackedFloat32Array:
	if brain_index < 0 or brain_index >= population_size or output_signals.size() != output_count:
		return PackedFloat32Array()
	var first: int = output_offsets[brain_index]
	return output_signals.slice(first, first + output_counts[brain_index])

# Optional lightweight signal readback for the simulation's selected-agent 3D
# preview. No shader, buffer, descriptor, readback or CPU work is added to the
# regular advance path; callers must opt in explicitly. Unlike full inspection,
# this works on an already initialized simulation population.
func read_visual_signals(brain_index: int) -> PackedFloat32Array:
	if rd == null or brain_index < 0 or brain_index >= population_size:
		return PackedFloat32Array()
	var data: PackedFloat32Array = rd.buffer_get_data(_brain_buffers[brain_index]["states"]).to_float32_array()
	if data.size() != _total * 12:
		return PackedFloat32Array()
	var signals := PackedFloat32Array()
	signals.resize(_total)
	for spatial in _total:
		# states[3 * spatial].w is the normalized signal used by the current
		# dynamics iteration (the same quantity visualized in Brain Viewer).
		signals[spatial] = data[spatial * 12 + 3]
	return signals


func selected_statistics(agent_index: int) -> Dictionary:
	if rd == null or agent_index < 0 or agent_index >= dynamics.size():
		return {}
	var meta: BrainDynamicsGPU = dynamics[agent_index]
	var n: int = _total
	var data: PackedFloat32Array = rd.buffer_get_data(_brain_buffers[agent_index]["states"]).to_float32_array()
	if data.size() != n * 12:
		return {}
	var states: PackedFloat32Array = PackedFloat32Array()
	var signals: PackedFloat32Array = PackedFloat32Array()
	states.resize(n)
	signals.resize(n)
	for rank in n:
		var spatial: int = meta._rank_to_spatial[rank]
		states[rank] = data[12 * spatial + 4]
		signals[rank] = data[12 * spatial + 3]
	var inputs: PackedFloat32Array = PackedFloat32Array()
	inputs.resize(meta._input_indices.size())
	for index in meta._input_indices.size():
		inputs[index] = states[meta._input_indices[index]]
	var outputs: PackedFloat32Array = PackedFloat32Array()
	outputs.resize(meta._output_indices.size())
	for index in meta._output_indices.size():
		var public_id: int = meta._output_indices[index]
		var spatial: int = meta._rank_to_spatial[public_id]
		outputs[index] = output_signals[output_offsets[agent_index] + index] if output_signals.size() == output_count else (signals[public_id] if data[12 * spatial + 11] > 0.5 else 0.0)
	return {"neuron_states_used": states, "normalized_signals": signals,
		"input_neuron_indices": meta._input_indices, "input_signals_used": inputs,
		"output_signals": outputs}


# The following API is intentionally inaccessible to the normal simulation.
# It is the only path allowed to read entire neural states or export topology.
func inspection_snapshot(brain_index: int = 0) -> Dictionary:
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size:
		return {}
	var buffers: Dictionary = _brain_buffers[brain_index]
	if not _submit([[_task("inspect", brain_index,
			["shared:parameters", "edges", "counts", "inspect_stats"], _total)]]):
		return {}
	var result: Dictionary = {
		"states": rd.buffer_get_data(buffers["states"]).to_float32_array(),
		"incoming_degrees": rd.buffer_get_data(buffers["incoming_degrees"]).to_int32_array(),
		"live_degrees": rd.buffer_get_data(buffers["live_degrees"]).to_int32_array(),
		"synapse_stats": rd.buffer_get_data(buffers["inspect_stats"]).to_float32_array()
	}
	return result

func inspection_lines(brain_index: int, selected_spatial: int, capacity: int) -> Dictionary:
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size or capacity <= 0:
		return {}
	if capacity > MAX_BUFFER_BYTES / 16:
		return {}
	var count_rid: RID = rd.storage_buffer_create(4)
	var lines_rid: RID = rd.storage_buffer_create(capacity * 16)
	if not count_rid.is_valid() or not lines_rid.is_valid():
		if count_rid.is_valid(): rd.free_rid(count_rid)
		if lines_rid.is_valid(): rd.free_rid(lines_rid)
		return {}
	var zero := PackedByteArray()
	zero.resize(4)
	rd.buffer_update(count_rid, 0, 4, zero)
	var pc := PackedByteArray()
	pc.resize(16)
	pc.encode_u32(0, brain_index)
	pc.encode_u32(4, selected_spatial if selected_spatial >= 0 else _total)
	pc.encode_u32(8, capacity)
	var ok: bool = _submit([[_task("visual_lines", brain_index,
		["shared:parameters", "edges", "counts", count_rid, lines_rid], _total, pc)]])
	# These two buffers are transient. Their descriptor set must not be cached
	# for the next export: otherwise it would reference released GPU memory.
	var visual_prefix: String = str(brain_index) + "|visual_lines:"
	for key in _uniform_cache.keys():
		if String(key).begins_with(visual_prefix):
			GPUResourceAudit.release(rd, _uniform_cache[key])
			_uniform_cache.erase(key)
	var answer: Dictionary = {}
	if ok:
		var count_values: PackedInt32Array = rd.buffer_get_data(count_rid).to_int32_array()
		if not count_values.is_empty():
			var count: int = mini(int(count_values[0]), capacity)
			var values: PackedFloat32Array = rd.buffer_get_data(lines_rid, 0, count * 16).to_float32_array() if count > 0 else PackedFloat32Array()
			answer = {"data": values, "reported": count_values[0], "count": count}
	rd.free_rid(count_rid)
	rd.free_rid(lines_rid)
	return answer

func inspection_synapses(brain_index: int = 0) -> Array:
	var result: Array = []
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size:
		return result
	var buffers: Dictionary = _brain_buffers[brain_index]
	var counts: PackedInt32Array = rd.buffer_get_data(buffers["counts"]).to_int32_array()
	var edges: PackedByteArray = rd.buffer_get_data(buffers["edges"])
	var stride: int = _strides[brain_index]
	for source in _total:
		for slot in counts[source]:
			var byte_offset: int = (source * stride + slot) * 16
			var target: int = edges.decode_s32(byte_offset)
			if target >= 0:
				result.append([source, target, edges.decode_float(byte_offset + 4), edges.decode_float(byte_offset + 8)])
	result.sort_custom(func(a, b):
		if int(a[0]) != int(b[0]):
			return int(a[0]) < int(b[0])
		return int(a[1]) < int(b[1]))
	return result

func inspection_incoming_degrees(brain_index: int = 0) -> PackedInt32Array:
	# Despite the Viewer's legacy get_initial_degrees() name, this must reflect
	# the live count after structural growth and synaptic death.
	if not enable_inspection or rd == null or brain_index < 0 or brain_index >= population_size:
		return PackedInt32Array()
	return rd.buffer_get_data(_brain_buffers[brain_index]["incoming_degrees"]).to_int32_array()

func clear() -> void:
	if rd == null:
		return
	_invalidate_uniforms()
	for buffers in _brain_buffers:
		for rid in buffers.values():
			if rid.is_valid():
				GPUResourceAudit.release(rd, rid)
	_brain_buffers.clear()
	_advance_first_stages.clear()
	_advance_second_stages.clear()
	for rid in _control.values():
		if rid.is_valid():
			GPUResourceAudit.release(rd, rid)
	_control.clear()
	for entry in _pipelines.values():
		if entry["pipeline"].is_valid():
			GPUResourceAudit.release(rd, entry["pipeline"])
		if entry["shader"].is_valid():
			GPUResourceAudit.release(rd, entry["shader"])
	_pipelines.clear()
	# On partial initialization, unconsumed builder-owned resources still exist.
	var builder: BrainBuilderGPU = BrainBuilderGPU.new()
	for brain in _brains:
		if brain is Dictionary and brain.has("gpu_state"):
			builder.release_gpu_state(brain)
	_brains.clear()
	dynamics.clear()
	rd.free()
	rd = null
	_brains.clear()
	_brain_buffers.clear()
	_strides = PackedInt32Array()
	_initial_strides = PackedInt32Array()
	_packed_parameters = PackedByteArray()
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
