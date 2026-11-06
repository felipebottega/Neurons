class_name BrainBuilderGPU
extends RefCounted

# GPU phenotype construction. The authoritative connection list (including
# developmental rescue) lives in GPU slots and degrees throughout generation.
# CPU tasks: assign roles by prime-gap spatial indices after provisional connections, validate degrees, compute psi.
# Brain Explorer consumes GPU aggregates. A simulation backend may request
# preservation of the exact same finalized GPU buffers rather than reconstructing
# connections from the Explorer summary.

const NEURON_SHADER := "res://shaders/brain_neurons.glsl"
const CONNECTION_SHADER := "res://shaders/brain_connections.glsl"
const APPLY_ROLES_SHADER := "res://shaders/brain_apply_roles.glsl"
const FILTER_ROLES_SHADER := "res://shaders/brain_filter_roles.glsl"
const RESCUE_SHADER := "res://shaders/brain_rescue.glsl"
const SCATTER_RESCUE_SHADER := "res://shaders/brain_scatter_rescue.glsl"
const FINALIZE_SHADER := "res://shaders/brain_finalize_weights.glsl"
const REDUCE_SHADER := "res://shaders/brain_reduce_statistics.glsl"
const FIELD_GENES := ["polarity_factor", "connections"]
const CONNECTION_WEIGHT_BINS := 20
# Optional post-finalization aggregates. These never change the GPU phenotype.
# NONE avoids both the histogram and the weight-sum reduction/readback.
# SUM requests only per-slice weight sums. HISTOGRAM requests sums and bins.
const STATS_NONE := 0
const STATS_WEIGHT_SUM := 1
const STATS_WEIGHT_HISTOGRAM := 2

func build(genome: Dictionary, size: Vector3i, num_inputs: int, num_outputs: int, progress: Callable = Callable(), retain_gpu_resources: bool = false, shared_rd: RenderingDevice = null, connections_density: float = 1.0, statistics_level: int = STATS_NONE) -> Dictionary:
	var I: int = size.x
	var J: int = size.y
	var K: int = size.z
	if I < 2 or J < 2 or K < 2 or num_inputs < 0 or num_outputs < 0:
		push_error("BrainBuilderGPU: invalid dimensions or input/output counts")
		return {}
	if (is_nan(connections_density) or is_inf(connections_density)) or connections_density <= 0.0:
		push_error("BrainBuilderGPU: connections density (eta) must be a finite positive number")
		return {}
	if statistics_level < STATS_NONE or statistics_level > STATS_WEIGHT_HISTOGRAM:
		push_error("BrainBuilderGPU: invalid statistics level")
		return {}
	var total: int = I * J * K
	if total <= 0 or num_inputs + num_outputs > total:
		push_error("BrainBuilderGPU: invalid neuron count")
		return {}

	var math := Functions.new()
	var beta: float = math.Beta(genome["beta"], I, J)
	var max_connections: int = maxi(1, floori(connections_density * math.F(beta, genome["max_connections"])))
	var connection_gene: Array = genome["connections"]
	var region_gene: Array = genome["initial_search_region"]
	var search_extent: float = math.S(region_gene)
	var radius_i: int = mini(I - 1, ceili(search_extent * float(I)))
	var radius_j: int = mini(J - 1, ceili(search_extent * float(J)))
	var radius_k: int = mini(K - 1, ceili(search_extent * float(K)))
	var brain: Dictionary = {
		"size": size,
		"beta": beta,
		"connections_density": connections_density,
		"max_connections": max_connections,
		"initial_search_extent": search_extent,
		"initial_search_radius_i": radius_i,
		"initial_search_radius_j": radius_j,
		"initial_search_radius_k": radius_k,
		"exponential_factor": math.F(0.4, genome["exponential_factor"]),
		"neurons": []
	}
	_progress(progress, 0)

	var owns_rd: bool = shared_rd == null
	var rd: RenderingDevice = RenderingServer.create_local_rendering_device() if owns_rd else shared_rd
	if rd == null:
		push_error("BrainBuilderGPU: could not create local RenderingDevice")
		math.free()
		return {}

	var gene_bytes := _pack_field_genes(genome)
	var gene_buffer: RID = GPUResourceAudit.create(rd, gene_bytes.size(), gene_bytes, "gene_buffer")
	var neuron_buffer: RID = GPUResourceAudit.create(rd, total * 32, PackedByteArray(), "neuron_buffer")
	var params := _dimensions_params(I, J, K, total)
	if not _run_shader(rd, NEURON_SHADER, [gene_buffer, neuron_buffer], params, ceili(float(total) / 64.0)):
		GPUResourceAudit.release(rd, gene_buffer)
		GPUResourceAudit.release(rd, neuron_buffer)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var neuron_bytes: PackedByteArray = rd.buffer_get_data(neuron_buffer)
	GPUResourceAudit.release(rd, gene_buffer)
	if neuron_bytes.size() != total * 32:
		push_error("BrainBuilderGPU: invalid neuron GPU readback")
		GPUResourceAudit.release(rd, neuron_buffer)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	_progress(progress, 20)

	# No genetic role field: role placement depends on tensor dimensions and
	# provisional connectivity. The connection graph is built before assignment.
	var neurons_spatial: Array = _unpack_neurons(neuron_bytes, size)
	_progress(progress, 30)

	# All neurons are provisionally internal during the two H selection passes.
	# A final filter will impose the role restrictions without recomputing H.
	var roles := PackedInt32Array()
	roles.resize(total)

	# N_max is a construction-time outgoing-degree ceiling, including developmental rescue.
	# Allocate exactly N_max slots per source at birth; no growth is permitted.
	# There can be fewer distinct reachable targets, but the genetic ceiling is
	# not reduced by that fact (and the second search retains its test budget).
	var reachable_limit: int = total - 1
	var initial_limit: int = maxi(1, mini(max_connections, reachable_limit))
	var slots_per_neuron: int = max_connections
	var slice_slot_bytes: int = I * J * slots_per_neuron * 16
	# The Godot buffer-size interface is uint32; individual slice allocations
	# must stay inside that bound, independently of total VRAM availability.
	if slice_slot_bytes <= 0 or slice_slot_bytes >= 4294967296:
		push_error("BrainBuilderGPU: slice slot buffer exceeds the 32-bit API limit")
		GPUResourceAudit.release(rd, neuron_buffer)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var roles_bytes := roles.to_byte_array()
	var roles_buffer: RID = GPUResourceAudit.create(rd, roles_bytes.size(), roles_bytes, "roles_buffer")
	var slot_buffers: Array[RID] = []
	for k in K:
		var slice_buffer: RID = GPUResourceAudit.create(rd, slice_slot_bytes, PackedByteArray(), "slot_slice")
		if not slice_buffer.is_valid():
			push_error("BrainBuilderGPU: failed to allocate connection slots for slice %d" % k)
			_free_slot_buffers(rd, slot_buffers)
			GPUResourceAudit.release(rd, roles_buffer)
			GPUResourceAudit.release(rd, neuron_buffer)
			if owns_rd:
				rd.free()
			math.free()
			return {}
		slot_buffers.append(slice_buffer)
	var count_buffer: RID = GPUResourceAudit.create(rd, total * 4, PackedByteArray(), "count_buffer")
	# The incoming array MUST start at zero for atomicAdd in brain_connections.
	var zero_incoming := PackedByteArray()
	zero_incoming.resize(total * 4)
	var incoming_buffer: RID = GPUResourceAudit.create(rd, total * 4, zero_incoming, "incoming_buffer")
	var h_params := _dimensions_params(I, J, K, slots_per_neuron)
	var coefficients: Vector4 = math.H_coefficients(connection_gene)
	# Connection uses a 60-byte push constant: density-adjusted Nmax at offset 48,
	# slice index at 52 and radius_k at 56. The original fields remain
	# in place; rescue uses its own 48-byte layout.
	h_params.resize(48)
	h_params.encode_float(16, coefficients.x)
	h_params.encode_float(20, coefficients.y)
	h_params.encode_float(24, coefficients.z)
	h_params.encode_float(28, coefficients.w)
	h_params.encode_u32(32, initial_limit)
	h_params.encode_u32(36, radius_i)
	h_params.encode_u32(40, radius_j)
	# The supplementary search is allowed the effective Nmax individual signed tests,
	# even when fewer than Nmax distinct destinations can be stored.
	# Share only its reachable prefix (ivec4 per tuple) across neurons.
	var candidate_offsets: PackedInt32Array = _initial_offset_table(region_gene, I, J, K, radius_i, radius_j, radius_k, max_connections)
	var tuple_count: int = candidate_offsets.size() / 4
	if radius_i >= I - 1 and radius_j >= J - 1 and radius_k >= K - 1:
		tuple_count = 0
	h_params.encode_u32(44, tuple_count)
	h_params.resize(60) # Genetic attempt budget, source slice, and third search radius.
	h_params.encode_u32(48, max_connections)
	h_params.encode_u32(56, radius_k)
	var offsets_bytes: PackedByteArray = candidate_offsets.to_byte_array()
	var offsets_buffer: RID = GPUResourceAudit.create(rd, offsets_bytes.size(), offsets_bytes, "offsets_buffer")
	# The initial pass counts all provisional local edges. This aggregate is
	# cleared and recomputed by the subsequent role-filter pass.
	var zero_local_count := PackedByteArray()
	zero_local_count.resize(4)
	var local_count_buffer: RID = GPUResourceAudit.create(rd, 4, zero_local_count, "local_count_buffer")

	var connection_params: Array[PackedByteArray] = []
	var connection_bindings: Array = []
	for k in K:
		var slice_params: PackedByteArray = h_params.duplicate()
		slice_params.encode_u32(52, k)
		connection_params.append(slice_params)
		connection_bindings.append([neuron_buffer, roles_buffer, slot_buffers[k], count_buffer, incoming_buffer, offsets_buffer, local_count_buffer])
	var ok := _run_shader_slices(rd, CONNECTION_SHADER, connection_bindings, connection_params, ceili(float(I * J) / 64.0))
	# The offset table is used only by the completed initial connection pass.
	GPUResourceAudit.release(rd, offsets_buffer)
	if not ok:
		GPUResourceAudit.release(rd, local_count_buffer)
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}

	# Read the two degree arrays of the provisional graph; the actual slots
	# remain on the GPU. The degrees are sufficient to select eligible roles.
	var outgoing_bytes: PackedByteArray = rd.buffer_get_data(count_buffer)
	var incoming_bytes: PackedByteArray = rd.buffer_get_data(incoming_buffer)
	if outgoing_bytes.size() != total * 4 or incoming_bytes.size() != total * 4:
		push_error("BrainBuilderGPU: invalid provisional GPU degree readback")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, local_count_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var outgoing: PackedInt32Array = outgoing_bytes.to_int32_array()
	var incoming: PackedInt32Array = incoming_bytes.to_int32_array()
	var selection: Dictionary = _rank_and_assign_roles(neurons_spatial, size, num_inputs, num_outputs, outgoing, incoming)
	var neurons: Array = selection["neurons"]
	brain["role_selection_gap"] = selection["gap"]
	brain["neurons"] = neurons
	# A pathological lack of eligible candidates makes the brain invalid,
	# even if the existing developmental rescue subsequently adds edges.
	brain["role_selection_valid"] = bool(selection["valid"])
	for neuron in neurons:
		var spatial: int = _spatial_index(neuron["position"], I, J)
		roles[spatial] = 1 if neuron["input"] else (2 if neuron["output"] else 0)
	if rd.buffer_update(roles_buffer, 0, total * 4, roles.to_byte_array()) != OK:
		push_error("BrainBuilderGPU: failed to upload selected roles")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, local_count_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	# Input phenotype overrides are applied only AFTER role selection.
	if not _run_shader(rd, APPLY_ROLES_SHADER, [neuron_buffer, roles_buffer], _dimensions_params(I, J, K, 0), ceili(float(total) / 64.0)):
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, local_count_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	# Filtering changes the receiving degrees, so recompute them from zero.
	# The filter reads existing slots: no second invocation of H is involved.
	if (rd.buffer_update(incoming_buffer, 0, total * 4, zero_incoming) != OK
		or rd.buffer_update(local_count_buffer, 0, 4, zero_local_count) != OK):
		push_error("BrainBuilderGPU: failed to reset filtered degree aggregates")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, local_count_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var filter_bindings: Array = []
	var filter_params: Array[PackedByteArray] = []
	for k in K:
		filter_bindings.append([slot_buffers[k], roles_buffer, count_buffer, incoming_buffer, local_count_buffer])
		var slice_params: PackedByteArray = _dimensions_params(I, J, K, slots_per_neuron)
		slice_params.resize(32)
		slice_params.encode_u32(16, k)
		slice_params.encode_u32(20, radius_i)
		slice_params.encode_u32(24, radius_j)
		slice_params.encode_u32(28, radius_k)
		filter_params.append(slice_params)
	if not _run_shader_slices(rd, FILTER_ROLES_SHADER, filter_bindings, filter_params, ceili(float(I * J) / 64.0)):
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, local_count_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	# Read final ordinary degrees and the local-pass aggregate AFTER filtering.
	outgoing_bytes = rd.buffer_get_data(count_buffer)
	incoming_bytes = rd.buffer_get_data(incoming_buffer)
	var local_count_bytes: PackedByteArray = rd.buffer_get_data(local_count_buffer)
	GPUResourceAudit.release(rd, local_count_buffer)
	if outgoing_bytes.size() != total * 4 or incoming_bytes.size() != total * 4 or local_count_bytes.size() != 4:
		push_error("BrainBuilderGPU: invalid filtered GPU degree readback")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	outgoing = outgoing_bytes.to_int32_array()
	incoming = incoming_bytes.to_int32_array()
	_progress(progress, 65)
	var before_rescue_count: int = 0
	for source_id in total:
		if outgoing[source_id] < 0 or outgoing[source_id] > slots_per_neuron or incoming[source_id] < 0:
			push_error("BrainBuilderGPU: invalid GPU degree values")
			for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
				GPUResourceAudit.release(rd, rid)
			_free_slot_buffers(rd, slot_buffers)
			if owns_rd:
				rd.free()
			math.free()
			return {}
		before_rescue_count += outgoing[source_id]
	var before_rescue_in: int = 0
	for degree in incoming:
		before_rescue_in += degree
	if before_rescue_in != before_rescue_count:
		push_error("BrainBuilderGPU: GPU incoming/outgoing degree sums disagree")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var local_edges: int = local_count_bytes.decode_u32(0)
	if local_edges > before_rescue_count:
		push_error("BrainBuilderGPU: GPU local-pass count exceeds initial edge count")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	brain["initial_local_connections"] = local_edges
	brain["initial_additional_connections"] = before_rescue_count - local_edges
	brain["incoming_counts"] = incoming
	brain["outgoing_counts"] = outgoing
	brain["connection_total"] = before_rescue_count
	# Compact rescue worklists preserve the established phenotype/rank order.
	# The shader, not GDScript, evaluates H^2 and writes the actual GPU slots.
	var receiver_rescue_ids := PackedInt32Array()
	var input_rescue_ids := PackedInt32Array()
	var isolated_before := {"input": 0, "internal": 0, "output": 0}
	for neuron in neurons:
		var spatial: int = _spatial_index(neuron["position"], I, J)
		if neuron["input"]:
			if outgoing[spatial] == 0:
				input_rescue_ids.append(spatial)
				isolated_before["input"] += 1
		elif incoming[spatial] == 0:
			receiver_rescue_ids.append(spatial)
			if neuron["output"]:
				isolated_before["output"] += 1
			else:
				isolated_before["internal"] += 1
	brain["isolated_before_rescue"] = isolated_before
	# Cleaning may delete all initially eligible edges of an interface neuron.
	# Such a role-selection failure invalidates the brain, even if diagnostic
	# developmental rescue later succeeds in adding a replacement edge.
	brain["role_cleanup_valid"] = isolated_before["input"] == 0 and isolated_before["output"] == 0
	_progress(progress, 75)

	# Rescue stays globally ordered on the GPU. Each accepted edge is appended
	# to a compact rescue log, later scattered into its source-slice slots.
	var stats_bytes := PackedByteArray()
	stats_bytes.resize(8 * 4) # H evaluations, failures by cause, rescued edges.
	var stats_buffer: RID = GPUResourceAudit.create(rd, stats_bytes.size(), stats_bytes, "stats_buffer")
	var rescue_log_capacity: int = maxi(1, receiver_rescue_ids.size() + input_rescue_ids.size())
	if rescue_log_capacity * 16 >= 4294967296:
		push_error("BrainBuilderGPU: rescue log exceeds buffer API size")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, stats_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var rescue_log_buffer: RID = GPUResourceAudit.create(rd, rescue_log_capacity * 16, PackedByteArray(), "rescue_log_buffer")
	if not rescue_log_buffer.is_valid():
		push_error("BrainBuilderGPU: failed to allocate compact rescue log")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, stats_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var receiver_ids_bytes: PackedByteArray = receiver_rescue_ids.to_byte_array()
	var input_ids_bytes: PackedByteArray = input_rescue_ids.to_byte_array()
	# Godot storage buffers cannot be zero bytes; an empty list has a dummy ID.
	if receiver_ids_bytes.is_empty():
		receiver_ids_bytes.resize(4)
	if input_ids_bytes.is_empty():
		input_ids_bytes.resize(4)
	var receiver_ids_buffer: RID = GPUResourceAudit.create(rd, receiver_ids_bytes.size(), receiver_ids_bytes, "receiver_ids_buffer")
	var input_ids_buffer: RID = GPUResourceAudit.create(rd, input_ids_bytes.size(), input_ids_bytes, "input_ids_buffer")
	var rescue_params: PackedByteArray = h_params.duplicate()
	rescue_params.resize(48) # Match the rescue shader push-constant layout.
	rescue_params.encode_u32(32, 0) # receiver pass
	rescue_params.encode_u32(36, receiver_rescue_ids.size())
	ok = _run_shader(rd, RESCUE_SHADER, [neuron_buffer, roles_buffer, rescue_log_buffer, count_buffer, incoming_buffer, stats_buffer, receiver_ids_buffer], rescue_params, 1)
	if not ok:
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, stats_buffer, receiver_ids_buffer, input_ids_buffer, rescue_log_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}

	rescue_params.encode_u32(32, 1) # input pass; sees completed receiver passes
	rescue_params.encode_u32(36, input_rescue_ids.size())
	ok = _run_shader(rd, RESCUE_SHADER, [neuron_buffer, roles_buffer, rescue_log_buffer, count_buffer, incoming_buffer, stats_buffer, input_ids_buffer], rescue_params, 1)
	if not ok:
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, stats_buffer, receiver_ids_buffer, input_ids_buffer, rescue_log_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}

	var result_stats: PackedByteArray = rd.buffer_get_data(stats_buffer)
	outgoing_bytes = rd.buffer_get_data(count_buffer)
	incoming_bytes = rd.buffer_get_data(incoming_buffer)
	for rid in [stats_buffer, receiver_ids_buffer, input_ids_buffer]:
		GPUResourceAudit.release(rd, rid)
	if result_stats.size() != 32 or outgoing_bytes.size() != total * 4 or incoming_bytes.size() != total * 4:
		push_error("BrainBuilderGPU: invalid GPU rescue statistics/degree readback")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, rescue_log_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var gpu_rescue_stats: PackedInt32Array = result_stats.to_int32_array()
	outgoing = outgoing_bytes.to_int32_array()
	incoming = incoming_bytes.to_int32_array()
	brain["rescue_candidates_receiver"] = gpu_rescue_stats[0]
	brain["rescue_candidates_input"] = gpu_rescue_stats[1]
	brain["rescue_candidates_total"] = gpu_rescue_stats[0] + gpu_rescue_stats[1]
	brain["rescue_edge_count"] = gpu_rescue_stats[6] + gpu_rescue_stats[7]
	brain["rescue_failure_causes"] = {
		"no_structural_candidate": gpu_rescue_stats[2],
		"no_positive_h_candidate": gpu_rescue_stats[3],
		"positive_h_candidates_saturated": gpu_rescue_stats[4],
		"input_source_saturated": gpu_rescue_stats[5]
	}
	brain["incoming_counts"] = incoming
	brain["outgoing_counts"] = outgoing
	var after: Dictionary = _isolated(neurons, incoming, outgoing, size)
	brain["isolated_after_rescue"] = after
	brain["valid"] = (brain["role_selection_valid"] and brain["role_cleanup_valid"]
		and after["input"] == 0 and after["internal"] == 0 and after["output"] == 0
		and gpu_rescue_stats[2] + gpu_rescue_stats[3] + gpu_rescue_stats[4] + gpu_rescue_stats[5] == 0)
	var final_in: int = 0
	var final_out: int = 0
	for spatial in total:
		if outgoing[spatial] < 0 or outgoing[spatial] > max_connections or incoming[spatial] < 0:
			brain["valid"] = false
		final_in += incoming[spatial]
		final_out += outgoing[spatial]
	if final_in != final_out or final_out != before_rescue_count + int(brain["rescue_edge_count"]):
		push_error("BrainBuilderGPU: rescue degree/slot accounting mismatch")
		brain["valid"] = false
	# Final connectivity includes the rescue edges, even if no optional
	# statistics were requested by the caller.
	brain["connection_total"] = final_out
	# Scatter rescue slots only after BOTH ordered rescue phases have finished.
	# No CPU edge list/readback is needed: only the small rescue count is read.
	var rescue_records: int = int(brain["rescue_edge_count"])
	var scatter_ok: bool = rescue_records <= rescue_log_capacity and rescue_records >= 0
	if scatter_ok and rescue_records > 0:
		var scatter_bindings: Array = []
		var scatter_params: Array[PackedByteArray] = []
		for k in K:
			scatter_bindings.append([rescue_log_buffer, slot_buffers[k]])
			var scatter_slice_params: PackedByteArray = _dimensions_params(I, J, K, slots_per_neuron)
			scatter_slice_params.resize(32)
			scatter_slice_params.encode_u32(16, k)
			scatter_slice_params.encode_u32(20, rescue_records)
			scatter_params.append(scatter_slice_params)
		scatter_ok = _run_shader_slices(rd, SCATTER_RESCUE_SHADER, scatter_bindings, scatter_params, ceili(float(rescue_records) / 64.0))
	GPUResourceAudit.release(rd, rescue_log_buffer)
	if not scatter_ok:
		push_error("BrainBuilderGPU: failed to distribute rescued connections to source slices")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}

	# psi depends on the FINAL rescued incoming degrees. Role IDs are spatial.
	var internal_count: int = 0
	var internal_incoming_sum: int = 0
	for spatial in total:
		if roles[spatial] == 0:
			internal_count += 1
			internal_incoming_sum += incoming[spatial]
	var mean_incoming: float = 0.0 if internal_count == 0 else float(internal_incoming_sum) / float(internal_count)
	brain["input_gain"] = (1.0 + sin(math.G(true, true, genome["input_influence"]))) * mean_incoming
	_progress(progress, 83)

	# Apply attenuation and the input floor after psi is known. Statistics
	# are optional: the default build does not scan edges for sums/bins, run a
	# reduction shader or read aggregate buffers back to the CPU.
	# The two unused bindings remain valid four-byte placeholders because the
	# finalization shader has a fixed storage-buffer interface.
	var want_sum: bool = statistics_level >= STATS_WEIGHT_SUM
	var want_bins: bool = statistics_level >= STATS_WEIGHT_HISTOGRAM
	var source_sum_buffer: RID = GPUResourceAudit.create(rd, total * 4 if want_sum else 4, PackedByteArray(), "source_sum_buffer")
	var source_bin_buffer: RID = GPUResourceAudit.create(rd, total * CONNECTION_WEIGHT_BINS * 4 if want_bins else 4, PackedByteArray(), "source_bin_buffer")
	if not source_sum_buffer.is_valid() or not source_bin_buffer.is_valid():
		push_error("BrainBuilderGPU: failed to allocate optional statistic buffers")
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, source_sum_buffer, source_bin_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	var final_params := _dimensions_params(I, J, K, slots_per_neuron)
	final_params.resize(32) # FP32 input floor and optional statistics level.
	final_params.encode_float(16, 1.0 / (1.0 + float(brain["input_gain"])))
	final_params.encode_u32(24, statistics_level)
	var finalize_bindings: Array = []
	var finalize_params: Array[PackedByteArray] = []
	for k in K:
		finalize_bindings.append([neuron_buffer, roles_buffer, slot_buffers[k], count_buffer, source_sum_buffer, source_bin_buffer])
		var finalize_slice_params: PackedByteArray = final_params.duplicate()
		finalize_slice_params.encode_u32(20, k)
		finalize_params.append(finalize_slice_params)
	var ok_finalize: bool = _run_shader_slices(rd, FINALIZE_SHADER, finalize_bindings, finalize_params, ceili(float(I * J) / 64.0))
	if not ok_finalize:
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, source_sum_buffer, source_bin_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
		math.free()
		return {}
	_progress(progress, 91)

	var weight_bins_raw := PackedByteArray()
	var weight_sums_raw := PackedByteArray()
	if want_sum:
		var slice_sum_buffer: RID = GPUResourceAudit.create(rd, K * 4, PackedByteArray(), "slice_sum_buffer")
		var slice_bin_buffer: RID = GPUResourceAudit.create(rd, K * CONNECTION_WEIGHT_BINS * 4 if want_bins else 4, PackedByteArray(), "slice_bin_buffer")
		if not slice_sum_buffer.is_valid() or not slice_bin_buffer.is_valid():
			push_error("BrainBuilderGPU: failed to allocate optional aggregate buffers")
			for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, source_sum_buffer, source_bin_buffer, slice_sum_buffer, slice_bin_buffer]:
				GPUResourceAudit.release(rd, rid)
			_free_slot_buffers(rd, slot_buffers)
			if owns_rd:
				rd.free()
			math.free()
			return {}
		var reduce_params := _dimensions_params(I, J, K, statistics_level)
		var reduce_count: int = K * CONNECTION_WEIGHT_BINS if want_bins else K
		var reduce_ok: bool = _run_shader(rd, REDUCE_SHADER, [source_sum_buffer, source_bin_buffer, slice_sum_buffer, slice_bin_buffer], reduce_params, ceili(float(reduce_count) / 64.0))
		if reduce_ok:
			weight_sums_raw = rd.buffer_get_data(slice_sum_buffer)
			if want_bins:
				weight_bins_raw = rd.buffer_get_data(slice_bin_buffer)
		for rid in [slice_sum_buffer, slice_bin_buffer]:
			GPUResourceAudit.release(rd, rid)
		if not reduce_ok:
			for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer, source_sum_buffer, source_bin_buffer]:
				GPUResourceAudit.release(rd, rid)
			_free_slot_buffers(rd, slot_buffers)
			if owns_rd:
				rd.free()
			math.free()
			return {}
	# Aggregate scratch buffers are temporary; the final slots/degrees may
	# remain GPU-resident for dynamics or for comparison/validation consumers.
	for rid in [source_sum_buffer, source_bin_buffer]:
		GPUResourceAudit.release(rd, rid)
	if retain_gpu_resources:
		brain["gpu_state"] = {
			"rd": rd,
			"owns_rd": owns_rd,
			"neuron_buffer": neuron_buffer,
			"roles_buffer": roles_buffer,
			"slot_buffers": slot_buffers,
			"slice_neuron_count": I * J,
			"slot_layout": "source_slice",
			"outgoing_buffer": count_buffer,
			"incoming_buffer": incoming_buffer,
			"slot_stride": slots_per_neuron,
			"initial_limit": initial_limit
		}
	else:
		for rid in [neuron_buffer, roles_buffer, count_buffer, incoming_buffer]:
			GPUResourceAudit.release(rd, rid)
		_free_slot_buffers(rd, slot_buffers)
		if owns_rd:
			rd.free()
	if want_sum:
		if weight_sums_raw.size() != K * 4 or (want_bins and weight_bins_raw.size() != K * CONNECTION_WEIGHT_BINS * 4):
			push_error("BrainBuilderGPU: invalid aggregate GPU readback")
			if retain_gpu_resources:
				release_gpu_state(brain)
			math.free()
			return {}
		var gpu_sums: PackedFloat32Array = weight_sums_raw.to_float32_array()
		var weight_sum: float = 0.0
		for slice_sum in gpu_sums:
			weight_sum += slice_sum
		brain["connection_weight_sum"] = weight_sum
		if want_bins:
			var gpu_bins: PackedInt32Array = weight_bins_raw.to_int32_array()
			var weight_slices: Array = []
			var histogram_total: int = 0
			for k in K:
				var bins := PackedInt32Array()
				bins.resize(CONNECTION_WEIGHT_BINS)
				for bin in CONNECTION_WEIGHT_BINS:
					var count: int = gpu_bins[k * CONNECTION_WEIGHT_BINS + bin]
					bins[bin] = count
					histogram_total += count
				weight_slices.append(bins)
			brain["connection_weight_bins"] = weight_slices
			if histogram_total != final_out:
				push_error("BrainBuilderGPU: GPU weight histogram/degree mismatch")
				brain["valid"] = false
	_progress(progress, 100)
	math.free()
	return brain


func _progress(callback: Callable, percent: int) -> void:
	if callback.is_valid():
		callback.call(percent)


func _dimensions_params(I: int, J: int, K: int, last: int) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(16)
	bytes.encode_u32(0, I)
	bytes.encode_u32(4, J)
	bytes.encode_u32(8, K)
	bytes.encode_u32(12, last)
	return bytes


# The 5985 unordered quadruples form 285 distinct 21-rotation orbits.
# Orbit representatives all have first index 0, and can be enumerated
# lexicographically by their remaining three positions. Rotate each quadruple
# and sort its four indices before forming base-10 (not base-4) numbers.
# The first orbit is deliberately (0,5,10,15), independently of this order.
# Build only ceil(Nmax/8) tuples: more cannot be reached by the GPU's
# Nmax individual signed-candidate tests. These tuples are shared by neurons.
func _initial_offset_table(gene: Array, I: int, J: int, K: int, radius_i: int, radius_j: int, radius_k: int, limit: int) -> PackedInt32Array:
	var offsets := PackedInt32Array()
	var capacity_i: int = I - radius_i - 1
	var capacity_j: int = J - radius_j - 1
	var capacity_k: int = K - radius_k - 1
	if capacity_i <= 0 and capacity_j <= 0 and capacity_k <= 0:
		return PackedInt32Array([0, 0, 0, 0]) # A nonempty SSBO is required.
	var tuple_limit: int = mini(5985, ceili(float(limit) / 8.0))
	offsets = _append_initial_orbit(offsets, gene, 0, 5, 10, 15, I, J, K, radius_i, radius_j, radius_k, capacity_i, capacity_j, capacity_k, tuple_limit)
	if offsets.size() / 4 >= tuple_limit:
		return offsets
	# For every orbit, exactly four rotated members start at position 0.
	# Among those, the lexicographically smallest one is its representative.
	# Their encoded indices are compared directly to avoid generating all
	# 5985 quadruples and avoid maintaining a 'visited' set.
	for b in range(1, 19):
		for c in range(b + 1, 20):
			for d in range(c + 1, 21):
				if b == 5 and c == 10 and d == 15:
					continue # The preferred first orbit was already emitted.
				var key: int = b * 441 + c * 21 + d
				if (key > (c - b) * 441 + (d - b) * 21 + (21 - b)
					or key > (d - c) * 441 + (21 - c) * 21 + (21 + b - c)
					or key > (21 - d) * 441 + (21 + b - d) * 21 + (21 + c - d)):
					continue
				offsets = _append_initial_orbit(offsets, gene, 0, b, c, d, I, J, K, radius_i, radius_j, radius_k, capacity_i, capacity_j, capacity_k, tuple_limit)
				if offsets.size() / 4 >= tuple_limit:
					return offsets
	return offsets


func _append_initial_orbit(offsets: PackedInt32Array, gene: Array, q0: int, q1: int, q2: int, q3: int, I: int, J: int, K: int, radius_i: int, radius_j: int, radius_k: int, capacity_i: int, capacity_j: int, capacity_k: int, tuple_limit: int) -> PackedInt32Array:
	for rotation in 21:
		var indices: Array[int] = [(q0 + rotation) % 21, (q1 + rotation) % 21, (q2 + rotation) % 21, (q3 + rotation) % 21]
		indices.sort()
		var digit_i: int = 0
		var digit_j: int = 0
		var digit_k: int = 0
		for idx in indices:
			digit_i = digit_i * 10 + int(gene[idx])
			digit_j = digit_j * 10 + int(gene[21 + idx])
			digit_k = digit_k * 10 + int(gene[42 + idx])
		var a: int = digit_i % I
		var b: int = digit_j % J
		var c: int = digit_k % K
		# Cycle the forced-outside axis i -> j -> k, with cyclic fallback
		# whenever that axis has no room beyond its local search radius.
		var tuple_index: int = offsets.size() / 4
		var axis: int = -1
		for shift in 3:
			var candidate_axis: int = (tuple_index + shift) % 3
			if ((candidate_axis == 0 and capacity_i > 0)
				or (candidate_axis == 1 and capacity_j > 0)
				or (candidate_axis == 2 and capacity_k > 0)):
				axis = candidate_axis
				break
		if axis == 0:
			a = radius_i + 1 + a % capacity_i
		elif axis == 1:
			b = radius_j + 1 + b % capacity_j
		else:
			c = radius_k + 1 + c % capacity_k
		offsets.append(a)
		offsets.append(b)
		offsets.append(c)
		offsets.append(0) # ivec4 for std430 alignment.
		if offsets.size() / 4 >= tuple_limit:
			return offsets
	return offsets


func _pack_field_genes(genome: Dictionary) -> PackedByteArray:
	var values := PackedInt32Array()
	for key in FIELD_GENES:
		for digit in genome[key]:
			values.append(int(digit))
	return values.to_byte_array()


# One shader execution, with explicit synchronization only at the end of each
# construction stage. This is not an animation/frame loop.
func _run_shader(rd: RenderingDevice, path: String, buffers: Array, constants: PackedByteArray, groups_x: int) -> bool:
	var resource: RDShaderFile = load(path)
	if resource == null:
		push_error("BrainBuilderGPU: failed to load " + path)
		return false
	var spirv: RDShaderSPIRV = resource.get_spirv()
	if spirv == null:
		push_error("BrainBuilderGPU: no SPIR-V for " + path)
		return false
	var compile_error: String = spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not compile_error.is_empty():
		push_error("BrainBuilderGPU: shader compilation failed: " + compile_error)
		return false
	var shader: RID = rd.shader_create_from_spirv(spirv)
	if not shader.is_valid():
		push_error("BrainBuilderGPU: could not create shader: " + path)
		return false
	var pipeline: RID = rd.compute_pipeline_create(shader)
	if not pipeline.is_valid():
		GPUResourceAudit.release(rd, shader)
		return false

	var uniforms: Array[RDUniform] = []
	for binding_index in buffers.size():
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding_index
		uniform.add_id(buffers[binding_index])
		uniforms.append(uniform)
	var uniform_set: RID = rd.uniform_set_create(uniforms, shader, 0)
	if not uniform_set.is_valid():
		GPUResourceAudit.release(rd, pipeline)
		GPUResourceAudit.release(rd, shader)
		return false

	var commands: int = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, pipeline)
	rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
	rd.compute_list_set_push_constant(commands, constants, constants.size())
	rd.compute_list_dispatch(commands, groups_x, 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	GPUResourceAudit.release(rd, uniform_set)
	GPUResourceAudit.release(rd, pipeline)
	GPUResourceAudit.release(rd, shader)
	return true


func _spatial_index(pos: Vector3i, I: int, J: int) -> int:
	return pos.x + I * (pos.y + J * pos.z)


func _position(index: int, I: int, J: int) -> Vector3i:
	var k: int = index / (I * J)
	var j: int = (index / I) % J
	return Vector3i(index % I, j, k)


func _unpack_neurons(raw: PackedByteArray, size: Vector3i) -> Array:
	var result: Array = []
	var total: int = size.x * size.y * size.z
	result.resize(total)
	for spatial in total:
		var b: int = spatial * 32
		result[spatial] = {
			"position": _position(spatial, size.x, size.y),
			"input": false,
			"output": false,
			"polarity_factor": int(raw.decode_float(b + 12)),
			"connections_value": raw.decode_float(b + 28),
		}
	return result


## Largest prime strictly below value; returns 0 if none exists.
static func _prime_below(value: int) -> int:
	for candidate in range(value - 1, 1, -1):
		var divisor: int = 2
		var is_prime: bool = true
		while divisor * divisor <= candidate:
			if candidate % divisor == 0:
				is_prime = false
				break
			divisor += 1
		if is_prime:
			return candidate
	return 0


## Each channel r prefers linear index r*g-1, with circular search from there.
## Inputs require >=2 provisional outgoing edges and outputs >=2 incoming edges.
## A role cannot be assigned twice and an unsuccessful search invalidates the brain.
func _rank_and_assign_roles(neurons_spatial: Array, size: Vector3i, num_inputs: int, num_outputs: int, outgoing: PackedInt32Array, incoming: PackedInt32Array) -> Dictionary:
	var total: int = size.x * size.y * size.z
	var channel_count: int = num_inputs + num_outputs
	var gap: int = 0
	if channel_count > 0:
		gap = _prime_below(floori(float(total) / float(channel_count)))
		if gap == 0:
			return {"neurons": neurons_spatial, "valid": false, "gap": 0}
	var occupied := PackedByteArray()
	occupied.resize(total)
	var selected_inputs: Array[int] = []
	var selected_outputs: Array[int] = []
	var valid_selection: bool = true
	for role_index in channel_count:
		var is_input: bool = role_index < num_inputs
		var starting_index: int = ((role_index + 1) * gap - 1) % total
		var chosen_index: int = -1
		for advance in total:
			var candidate: int = (starting_index + advance) % total
			if occupied[candidate] != 0:
				continue
			if is_input and outgoing[candidate] < 2:
				continue
			if not is_input and incoming[candidate] < 2:
				continue
			chosen_index = candidate
			break
		if chosen_index == -1:
			valid_selection = false
			break
		occupied[chosen_index] = 1
		if is_input:
			neurons_spatial[chosen_index]["input"] = true
			selected_inputs.append(chosen_index)
		else:
			neurons_spatial[chosen_index]["output"] = true
			selected_outputs.append(chosen_index)
	# Channel order must agree with the GPU model: inputs first, then
	# internal neurons, then outputs, retaining the output channel order.
	var ordered: Array = []
	ordered.resize(total)
	var destination: int = 0
	for spatial in selected_inputs:
		ordered[destination] = neurons_spatial[spatial]
		destination += 1
	for spatial in total:
		if occupied[spatial] == 0:
			ordered[destination] = neurons_spatial[spatial]
			destination += 1
	for spatial in selected_outputs:
		ordered[destination] = neurons_spatial[spatial]
		destination += 1
	for neuron in ordered:
		if bool(neuron["input"]):
			neuron["polarity_factor"] = 1
	return {"neurons": ordered, "valid": valid_selection, "gap": gap}


# The active connection slots are kept in GPU buffers throughout construction.
# Optional per-slice weight aggregates are generated only when requested.
# No CPU-side per-edge targets, weights, or offset arrays are produced.


func _isolated(neurons: Array, incoming: PackedInt32Array, out_degrees: PackedInt32Array, size: Vector3i) -> Dictionary:
	var counts := {"input": 0, "internal": 0, "output": 0}
	for neuron in neurons:
		var spatial: int = _spatial_index(neuron["position"], size.x, size.y)
		if neuron["input"]:
			if out_degrees[spatial] == 0:
				counts["input"] += 1
		elif incoming[spatial] == 0:
			if neuron["output"]:
				counts["output"] += 1
			else:
				counts["internal"] += 1
	return counts


# Free a retained GPU phenotype when the simulation has finished with it.
# The Explorer uses retain_gpu_resources=false and frees immediately in build().
func release_gpu_state(brain: Dictionary) -> void:
	if not brain.has("gpu_state"):
		return
	var state: Dictionary = brain["gpu_state"]
	var rd: RenderingDevice = state["rd"]
	for key in ["neuron_buffer", "roles_buffer", "outgoing_buffer", "incoming_buffer"]:
		GPUResourceAudit.release(rd, state[key])
	_free_slot_buffers(rd, state["slot_buffers"])
	if state.get("owns_rd", true):
		rd.free()
	brain.erase("gpu_state")


func _free_slot_buffers(rd: RenderingDevice, buffers: Array) -> void:
	for rid in buffers:
		if rid.is_valid():
			GPUResourceAudit.release(rd, rid)


func _run_shader_slices(rd: RenderingDevice, path: String, bindings: Array, constants: Array[PackedByteArray], groups_x: int) -> bool:
	if bindings.is_empty() or bindings.size() != constants.size():
		return false
	var resource: RDShaderFile = load(path)
	if resource == null:
		push_error("BrainBuilderGPU: failed to load " + path)
		return false
	var spirv: RDShaderSPIRV = resource.get_spirv()
	if spirv == null:
		push_error("BrainBuilderGPU: shader has no SPIR-V: " + path)
		return false
	var compile_error: String = spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not compile_error.is_empty():
		push_error("BrainBuilderGPU: invalid compute shader: " + path + "\n" + compile_error)
		return false
	var shader: RID = rd.shader_create_from_spirv(spirv)
	if not shader.is_valid():
		return false
	var pipeline: RID = rd.compute_pipeline_create(shader)
	if not pipeline.is_valid():
		GPUResourceAudit.release(rd, shader)
		return false
	var sets: Array[RID] = []
	for buffer_list in bindings:
		var uniforms: Array[RDUniform] = []
		for binding_index in buffer_list.size():
			var uniform := RDUniform.new()
			uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
			uniform.binding = binding_index
			uniform.add_id(buffer_list[binding_index])
			uniforms.append(uniform)
		var uniform_set: RID = rd.uniform_set_create(uniforms, shader, 0)
		if not uniform_set.is_valid():
			for existing in sets:
				GPUResourceAudit.release(rd, existing)
			GPUResourceAudit.release(rd, pipeline)
			GPUResourceAudit.release(rd, shader)
			return false
		sets.append(uniform_set)
	for k in sets.size():
		var commands: int = rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(commands, pipeline)
		rd.compute_list_bind_uniform_set(commands, sets[k], 0)
		rd.compute_list_set_push_constant(commands, constants[k], constants[k].size())
		rd.compute_list_dispatch(commands, groups_x, 1, 1)
		rd.compute_list_end()
	rd.submit()
	rd.sync()
	for uniform_set in sets:
		GPUResourceAudit.release(rd, uniform_set)
	GPUResourceAudit.release(rd, pipeline)
	GPUResourceAudit.release(rd, shader)
	return true
