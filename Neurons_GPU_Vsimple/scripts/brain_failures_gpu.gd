class_name BrainFailuresGPU
extends RefCounted

# Exact directional reciprocity without materializing edge dictionaries on CPU.
# An open-addressed GPU hash table stores (global source, global destination).
# After all slices have inserted their edges, a second pass counts reverse
# edges. The hash table uses at most ~50% occupancy and is freed per brain.
const INIT_SHADER := "res://shaders/brain_failures_hash_init.glsl"
const RECIPROCITY_SHADER := "res://shaders/brain_failures_reciprocity.glsl"


func calculate(brain: Dictionary) -> Dictionary:
	if not brain.has("gpu_state"):
		push_error("BrainFailuresGPU: GPU state was not retained")
		return {}
	var state: Dictionary = brain["gpu_state"]
	var rd: RenderingDevice = state["rd"]
	var dimensions: Vector3i = brain["size"]
	var plane: int = dimensions.x * dimensions.y
	var total_neurons: int = plane * dimensions.z
	var edge_count: int = int(brain["connection_total"])
	if edge_count == 0:
		return {"total": 0, "reciprocal": 0, "rate": 0.0}
	# Keep the load factor at most 1/2, with room for a sentinel.
	var capacity: int = 2
	while capacity <= 2 * edge_count:
		capacity *= 2
	if capacity * 8 >= 4294967296:
		push_error("BrainFailuresGPU: reciprocity hash buffer would exceed the 32-bit API limit")
		return {}
	var table_buffer: RID = rd.storage_buffer_create(capacity * 8)
	var zeros := PackedByteArray()
	zeros.resize(8)
	var results_buffer: RID = rd.storage_buffer_create(8, zeros)
	if not table_buffer.is_valid() or not results_buffer.is_valid():
		push_error("BrainFailuresGPU: hash allocation failed")
		_cleanup(rd, [table_buffer, results_buffer])
		return {}
	var runner := BrainBuilderGPU.new()
	var init_params := PackedByteArray()
	init_params.resize(16)
	init_params.encode_u32(0, capacity)
	if not runner._run_shader(rd, INIT_SHADER, [table_buffer], init_params, mini(4096, ceili(float(capacity) / 64.0))):
		push_error("BrainFailuresGPU: hash initialization shader failed")
		_cleanup(rd, [table_buffer, results_buffer])
		return {}
	var bindings: Array = []
	var constants: Array[PackedByteArray] = []
	for k in dimensions.z:
		bindings.append([state["slot_buffers"][k], state["outgoing_buffer"], table_buffer, results_buffer])
		var params := PackedByteArray()
		params.resize(32)
		params.encode_u32(0, plane)
		params.encode_u32(4, k)
		params.encode_u32(8, int(state["slot_stride"]))
		params.encode_u32(12, capacity)
		params.encode_u32(16, 0)
		params.encode_u32(20, total_neurons)
		constants.append(params)
	var groups: int = ceili(float(plane) / 64.0)
	if not runner._run_shader_slices(rd, RECIPROCITY_SHADER, bindings, constants, groups):
		push_error("BrainFailuresGPU: directed edge insertion failed")
		_cleanup(rd, [table_buffer, results_buffer])
		return {}
	# Probe all reverse edges only AFTER the completed insertion dispatch.
	for k in constants.size():
		constants[k].encode_u32(16, 1)
	if not runner._run_shader_slices(rd, RECIPROCITY_SHADER, bindings, constants, groups):
		push_error("BrainFailuresGPU: reciprocal-edge lookup failed")
		_cleanup(rd, [table_buffer, results_buffer])
		return {}
	var data: PackedByteArray = rd.buffer_get_data(results_buffer)
	_cleanup(rd, [table_buffer, results_buffer])
	if data.size() != 8:
		push_error("BrainFailuresGPU: invalid hash result readback")
		return {}
	var reciprocal_count: int = data.decode_u32(0)
	var errors: int = data.decode_u32(4)
	if errors != 0 or reciprocal_count > edge_count:
		push_error("BrainFailuresGPU: invalid edges or hash insertion failures: %d" % errors)
		return {}
	return {
		"total": edge_count,
		"reciprocal": reciprocal_count,
		"rate": 100.0 * float(reciprocal_count) / float(edge_count)
	}


func _cleanup(rd: RenderingDevice, buffers: Array) -> void:
	for rid in buffers:
		if rid.is_valid():
			rd.free_rid(rid)
