class_name BrainDynamicsNative
extends RefCounted

var runtime: BrainDynamics = BrainDynamics.new()

func set_brain(brain: Dictionary) -> void:
    runtime.set_brain(brain.duplicate(true))

func reset() -> void:
    runtime.reset()

func clear() -> void:
    runtime.clear()

func advance(delta: float) -> int:
    return runtime.advance(delta)

func add_input_signal(slot: int, value: float) -> void:
    runtime.add_input_signal(slot, value)

func get_output_signal(index: int) -> float:
    return runtime.output_signals[index] if index >= 0 and index < runtime.output_signals.size() else 0.0

func set_profiling_enabled(enabled: bool) -> void:
    runtime.profiling_enabled = enabled

func get_profile_total_us() -> int:
    return runtime.profile_total_us
func get_profile_prepare_and_firing_us() -> int:
    return runtime.profile_prepare_and_firing_us
func get_profile_modulatory_field_us() -> int:
    return runtime.profile_modulatory_field_us
func get_profile_connections_us() -> int:
    return runtime.profile_connections_us
func get_profile_homeostatic_us() -> int:
    return runtime.profile_homeostatic_us
func get_profile_propagation_statistics_us() -> int:
    return runtime.profile_propagation_statistics_us
func get_profile_state_update_us() -> int:
    return runtime.profile_state_update_us
func get_profile_structural_plasticity_us() -> int:
    return runtime.profile_structural_plasticity_us

func get_brain_snapshot() -> Dictionary:
    return runtime.brain.duplicate(true)

func get_viewer_state() -> Dictionary:
    var r: BrainDynamics = runtime
    return {
        "neuron_states": r.neuron_states,
        "neuron_states_used": r.neuron_states_used,
        "normalized_signals": r.normalized_signals,
        "firing_states": r.firing_states,
        "activity_traces": r.activity_traces,
        "incoming_signals": r.incoming_signals,
        "incoming_connection_counts": r.incoming_connection_counts,
        "fatigues": r.fatigues,
        "fatigues_used": r.fatigues_used,
        "effective_thresholds_used": r.effective_thresholds_used,
        "decay_rates": r.decay_rates,
        "modulatory_releases": r.modulatory_releases,
        "modulatory_fields": r.modulatory_fields,
        "effective_modulations": r.effective_modulations,
        "input_neuron_indices": r.input_neuron_indices,
        "output_slot_by_neuron_index": r.output_slot_by_neuron_index,
        "input_signals_used": r.input_signals_used,
        "output_signals": r.output_signals,
        "input_gain": r.input_gain,
        "input_min_weight": r.input_min_weight,
        "total_connections": r.total_connections,
        "total_connections_formed": r.total_connections_formed,
        "total_structural_plasticity_connections_formed": r.total_structural_plasticity_connections_formed,
        "total_homeostatic_connections_formed": r.total_homeostatic_connections_formed,
        "total_connections_destroyed": r.total_connections_destroyed,
        "added_connections_last_iteration": r.added_connections_last_iteration,
        "mean_outgoing_connections": r.mean_outgoing_connections,
        "min_out_degree": r.min_out_degree,
        "max_out_degree": r.max_out_degree,
        "mean_signal": r.mean_signal,
        "mean_modulatory_release": r.mean_modulatory_release,
        "mean_modulatory_field": r.mean_modulatory_field,
        "mean_effective_modulation": r.mean_effective_modulation,
        "mean_eligibility_trace": r.mean_eligibility_trace,
        "mean_weight": r.mean_weight,
        "iteration_count": r.iteration_count,
        "elapsed_time_seconds": r.elapsed_time_seconds
    }

func get_visual_signals() -> PackedFloat32Array:
    var size: Vector3i = runtime.brain.get("size", Vector3i.ZERO)
    var values := PackedFloat32Array()
    values.resize(size.x * size.y * size.z)
    if runtime.brain.is_empty():
        return values
    var neurons: Array = runtime.brain["neurons"]
    var plane := size.x * size.y
    for index in neurons.size():
        var p: Vector3i = neurons[index]["position"]
        values[p.x + size.x * p.y + plane * p.z] = runtime.normalized_signals[index]
    return values

func get_neuron_statistics(parameter: String) -> Dictionary:
    var values: Array[float] = []
    var source: PackedFloat64Array
    match parameter:
        "state": source = runtime.neuron_states
        "signal": source = runtime.normalized_signals
        "fatigue": source = runtime.fatigues
        "effective_threshold": source = runtime.effective_thresholds_used
        "activity_trace": source = runtime.activity_traces
        "incoming_signal": source = runtime.incoming_signals
        "modulatory_release": source = runtime.modulatory_releases
        "modulatory_field": source = runtime.modulatory_fields
        "effective_modulation": source = runtime.effective_modulations
        _: source = runtime.neuron_states
    for idx in source.size():
        if (parameter == "state" or parameter == "signal") and runtime.brain["neurons"][idx]["input"]:
            continue
        values.append(source[idx])
    values.sort()
    var n := values.size()
    if n == 0:
        return {"min":0.0,"max":0.0,"mean":0.0,"std":0.0,"25%":0.0,"50%":0.0,"75%":0.0,"count":0}
    var total := 0.0
    for v in values: total += v
    var mean := total / n
    var variance := 0.0
    for v in values: variance += (v - mean) * (v - mean)
    return {"min":values[0], "max":values[-1], "mean":mean, "std":sqrt(variance / n), "25%":_percentile(values,0.25), "50%":_percentile(values,0.5), "75%":_percentile(values,0.75), "count":n}

func _percentile(values: Array[float], q: float) -> float:
    var position: float = float(values.size() - 1) * q
    var left: int = int(floor(position))
    var right: int = mini(left + 1, values.size() - 1)
    return lerpf(values[left], values[right], position - left)

func read_synapses() -> Array:
    var rows: Array = []
    if runtime.brain.is_empty():
        return rows
    var neurons: Array = runtime.brain["neurons"]
    var size: Vector3i = runtime.brain["size"]
    var plane: int = size.x * size.y
    for neuron in neurons:
        var p: Vector3i = neuron["position"]
        var source: int = p.x + size.x * p.y + plane * p.z
        for connection in neuron["connections"]:
            var q: Vector3i = connection["target"]
            rows.append([source, q.x + size.x * q.y + plane * q.z, connection["weight"], connection["eligibility_trace"]])
    rows.sort_custom(func(a: Array, b: Array) -> bool: return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]))
    return rows
