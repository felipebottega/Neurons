class_name GPUResourceAudit
extends RefCounted

# Compatibility wrapper for GPU resource ownership. No memory accounting,
# resource registry, diagnostic readbacks or console reporting.
static func create(device: RenderingDevice, byte_size: int,
	initial: PackedByteArray = PackedByteArray(), _label: String = "storage") -> RID:
	return device.storage_buffer_create(byte_size, initial)

static func release(device: RenderingDevice, rid: RID) -> void:
	if rid.is_valid():
		device.free_rid(rid)
