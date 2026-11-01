class_name BrainBuilderNative
extends RefCounted

# Adapter maintains the public interface used by GPU-aligned scenes, but every
# model calculation is performed by scripts/brain_builder.gd on the CPU.
var builder := BrainBuilder.new()

func build(genome: GenomeNative, size: Vector3i, num_inputs: int, num_outputs: int, progress_callback: Callable = Callable(), connections_density: float = 1.0) -> Dictionary:
    if genome == null:
        return {}
    return builder.build(genome, size, num_inputs, num_outputs, progress_callback, connections_density)
