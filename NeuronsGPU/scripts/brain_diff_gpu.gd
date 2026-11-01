class_name BrainDiffGPU
extends RefCounted

# GPU-only intersection and weight-difference histogram. The two phenotypes
# must belong to the SAME RenderingDevice (created by the first build).
# Host readbacks: 4-byte shared count, 4-byte maximum, and 20*K histogram bins.
const SORT_SHADER := "res://shaders/brain_diff_sort.glsl"
const EDGE_SHADER := "res://shaders/brain_diff_edges.glsl"
const BINS := 20


func compare(first_brain: Dictionary, second_brain: Dictionary) -> Dictionary:
	if not first_brain.has("gpu_state") or not second_brain.has("gpu_state"):
		push_error("BrainDiffGPU: retained GPU buffers are required")
		return {}
	var first: Dictionary = first_brain["gpu_state"]
	var second: Dictionary = second_brain["gpu_state"]
	if first["rd"] != second["rd"] or first_brain["size"] != second_brain["size"]:
		push_error("BrainDiffGPU: both brains must share the same RenderingDevice and dimensions")
		return {}

	var rd: RenderingDevice = first["rd"]
	var size: Vector3i = first_brain["size"]
	var plane: int = size.x * size.y
	var stride_a: int = int(first["slot_stride"])
	var stride_b: int = int(second["slot_stride"])
	var total_a: int = int(first_brain["connection_total"])
	var total_b: int = int(second_brain["connection_total"])
	var zero_four := PackedByteArray()
	zero_four.resize(4)
	var zero_bins := PackedByteArray()
	zero_bins.resize(size.z * BINS * 4)
	var shared_buffer: RID = rd.storage_buffer_create(4, zero_four)
	var maximum_buffer: RID = rd.storage_buffer_create(4, zero_four)
	var histogram_buffer: RID = rd.storage_buffer_create(zero_bins.size(), zero_bins)
	var temporaries: Array[RID] = [shared_buffer, maximum_buffer, histogram_buffer]
	for rid in temporaries:
		if not rid.is_valid():
			push_error("BrainDiffGPU: cannot allocate comparison aggregates")
			_cleanup(rd, temporaries)
			return {}

	var sorting: Array = []
	var sorting_params: Array[PackedByteArray] = []
	var weight_bindings: Array = []
	var edge_params: Array[PackedByteArray] = []
	for k in size.z:
		var first_slots: RID = first["slot_buffers"][k]
		var second_slots: RID = second["slot_buffers"][k]
		var pair: Array = [first_slots, second_slots, first["outgoing_buffer"], second["outgoing_buffer"]]
		sorting.append(pair)
		weight_bindings.append(pair + [shared_buffer, maximum_buffer, histogram_buffer])
		var constants := PackedByteArray()
		constants.resize(32)
		constants.encode_u32(0, plane)
		constants.encode_u32(4, k)
		constants.encode_u32(8, stride_a)
		constants.encode_u32(12, stride_b)
		sorting_params.append(constants.slice(0, 16))
		edge_params.append(constants)

	var runner := BrainBuilderGPU.new()
	var groups: int = ceili(float(plane) / 64.0)
	if not runner._run_shader_slices(rd, SORT_SHADER, sorting, sorting_params, groups):
		push_error("BrainDiffGPU: GPU connection sorting failed")
		_cleanup(rd, temporaries)
		return {}
	if not runner._run_shader_slices(rd, EDGE_SHADER, weight_bindings, edge_params, groups):
		push_error("BrainDiffGPU: GPU intersection/max-difference pass failed")
		_cleanup(rd, temporaries)
		return {}
	for k in edge_params.size():
		edge_params[k].encode_u32(16, 1)
	if not runner._run_shader_slices(rd, EDGE_SHADER, weight_bindings, edge_params, groups):
		push_error("BrainDiffGPU: GPU weight histogram pass failed")
		_cleanup(rd, temporaries)
		return {}

	var shared_raw: PackedByteArray = rd.buffer_get_data(shared_buffer)
	var maximum_raw: PackedByteArray = rd.buffer_get_data(maximum_buffer)
	var bins_raw: PackedByteArray = rd.buffer_get_data(histogram_buffer)
	_cleanup(rd, temporaries)
	if shared_raw.size() != 4 or maximum_raw.size() != 4 or bins_raw.size() != size.z * BINS * 4:
		push_error("BrainDiffGPU: invalid GPU comparison readback")
		return {}
	var shared_count: int = shared_raw.decode_u32(0)
	var union_count: int = total_a + total_b - shared_count
	if shared_count > mini(total_a, total_b) or union_count < 0:
		push_error("BrainDiffGPU: invalid shared connection count")
		return {}
	var max_value: float = maximum_raw.decode_float(0)
	if is_nan(max_value) or is_inf(max_value) or max_value < 0.0:
		push_error("BrainDiffGPU: invalid maximum weight difference")
		return {}
	var slice_histograms: Array = []
	var aggregate := PackedInt32Array()
	aggregate.resize(BINS)
	var histogram_total: int = 0
	for k in size.z:
		var bins := PackedInt32Array()
		bins.resize(BINS)
		for b in BINS:
			var value: int = bins_raw.decode_u32((k * BINS + b) * 4)
			bins[b] = value
			aggregate[b] += value
			histogram_total += value
		slice_histograms.append(bins)
	if histogram_total != union_count:
		push_error("BrainDiffGPU: weight histogram does not cover the union of connections")
		return {}
	return {
		"shared": shared_count,
		"union": union_count,
		"shared_percent": 100.0 if union_count == 0 else 100.0 * float(shared_count) / float(union_count),
		"max_abs_weight_difference": max_value,
		"weight_slices": slice_histograms,
		"weight_all": aggregate
	}


func _cleanup(rd: RenderingDevice, buffers: Array[RID]) -> void:
	for rid in buffers:
		if rid.is_valid():
			rd.free_rid(rid)
