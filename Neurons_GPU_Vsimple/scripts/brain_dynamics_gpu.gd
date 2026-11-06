class_name BrainDynamicsGPU
extends RefCounted

# Vsimple per-brain metadata; all neural iterations run in BrainPopulationGPU.
var _brain: Dictionary = {}
var _size := Vector3i.ZERO
var _total: int = 0
var _stride: int = 0
var _iteration: int = 0
var _input_indices := PackedInt32Array()
var _output_indices := PackedInt32Array()
var _rank_to_spatial := PackedInt32Array()
var _rank_is_output := PackedByteArray()
var _output_slots := PackedInt32Array()

func _build_rank_maps() -> void:
	_rank_to_spatial.resize(_total)
	_rank_is_output.resize(_total)
	_output_slots.resize(_total)
	_output_slots.fill(-1)
	_input_indices = PackedInt32Array()
	_output_indices = PackedInt32Array()
	for rank in _total:
		var neuron: Dictionary = _brain["neurons"][rank]
		var pos: Vector3i = neuron["position"]
		_rank_to_spatial[rank] = pos.x + _size.x * (pos.y + _size.y * pos.z)
		_rank_is_output[rank] = 1 if neuron["output"] else 0
		if neuron["input"]:
			_input_indices.append(rank)
		elif neuron["output"]:
			_output_slots[rank] = _output_indices.size()
			_output_indices.append(rank)

# Preserve the 112-byte UBO ABI used by the immutable graph's GPU kernels.
# Deleted genetic parameters always occupy zero, never influence dynamics.
func _params() -> PackedByteArray:
	var data := PackedByteArray()
	data.resize(80)
	data.encode_u32(0, _size.x)
	data.encode_u32(4, _size.y)
	data.encode_u32(8, _size.z)
	data.encode_u32(12, _total)
	data.encode_u32(16, _stride)
	data.encode_u32(20, _iteration)
	data.encode_float(28, float(_brain["exponential_factor"]))
	data.encode_float(60, float(_brain["input_gain"]))
	data.encode_float(64, 1.0 / (1.0 + float(_brain["input_gain"])))
	return data
