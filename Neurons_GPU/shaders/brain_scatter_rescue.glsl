#[compute]
#version 450

// Scatter the ordered rescue log into the slots of the SOURCE slice.
// Targets may belong to any slice. Slot indexes were reserved in the ordered
// rescue shader, so distinct records have distinct (source,slot) addresses.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct RescueRecord {
    uint source;
    uint slot;
    int target;
    float h;
};
layout(set = 0, binding = 0, std430) readonly buffer RescueRecords {
    RescueRecord data[];
} records;
struct Slot {
    int target;
    float h;
    float trace;
    float pad;
};
layout(set = 0, binding = 1, std430) buffer Slots {
    Slot data[];
} slots;
layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint slot_stride;
    uint slice_k;
    uint record_count;
    uint reserved0;
    uint reserved1;
} p;
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= p.record_count) return;
    RescueRecord entry = records.data[id];
    uint plane = p.I * p.J;
    if (entry.source / plane != p.slice_k) return;
    uint local_source = entry.source % plane;
    // The ordered rescue pass has already checked slot capacity.
    slots.data[local_source * p.slot_stride + entry.slot] = Slot(entry.target, entry.h, 0.0, 1.0);
}
