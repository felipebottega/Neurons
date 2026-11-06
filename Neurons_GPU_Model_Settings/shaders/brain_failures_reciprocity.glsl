#[compute]
#version 450
// Hash directed edges (source,target), then count those with a reverse edge.
// Each invocation handles all outgoing edges of one neuron in source_slice.
// Different slices reuse the same hash table; source and target IDs are GLOBAL.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Slot { int target; float weight; float trace; float pad; };
struct Entry { uint source; uint target; };
layout(set = 0, binding = 0, std430) readonly buffer Slots { Slot data[]; } slots;
layout(set = 0, binding = 1, std430) readonly buffer Degrees { int data[]; } degrees;
layout(set = 0, binding = 2, std430) buffer HashTable { Entry data[]; } table_entries;
layout(set = 0, binding = 3, std430) buffer Results { uint data[]; } results;
layout(push_constant, std430) uniform Params {
    uint plane; uint source_slice; uint stride; uint capacity;
    uint mode; uint total; uint pad0; uint pad1;
} params;
uint hash_pair(uint src, uint dst) {
    uint value = src * 0x9e3779b1u ^ dst * 0x85ebca77u;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    return value & (params.capacity - 1u);
}
void insert_edge(uint src, uint dst) {
    uint slot = hash_pair(src, dst);
    for (uint probe = 0u; probe < params.capacity; ++probe) {
        uint old_source = atomicCompSwap(table_entries.data[slot].source, 0xffffffffu, src);
        if (old_source == 0xffffffffu) {
            // The lookup dispatch is separated by a submit/sync from insertion.
            table_entries.data[slot].target = dst;
            return;
        }
        slot = (slot + 1u) & (params.capacity - 1u);
    }
    atomicAdd(results.data[1], 1u);
}
bool edge_exists(uint src, uint dst) {
    uint slot = hash_pair(src, dst);
    for (uint probe = 0u; probe < params.capacity; ++probe) {
        uint stored_source = table_entries.data[slot].source;
        if (stored_source == 0xffffffffu) return false;
        if (stored_source == src && table_entries.data[slot].target == dst) return true;
        slot = (slot + 1u) & (params.capacity - 1u);
    }
    return false;
}
void main() {
    uint local_id = gl_GlobalInvocationID.x;
    if (local_id >= params.plane) return;
    uint source = params.source_slice * params.plane + local_id;
    int count = degrees.data[source];
    uint base = local_id * params.stride;
    uint reciprocal = 0u;
    for (int c = 0; c < count; ++c) {
        int target = slots.data[base + uint(c)].target;
        if (target < 0 || uint(target) >= params.total) {
            atomicAdd(results.data[1], 1u);
            continue;
        }
        if (params.mode == 0u) insert_edge(source, uint(target));
        else if (edge_exists(uint(target), source)) reciprocal++;
    }
    if (params.mode == 1u && reciprocal != 0u)
        atomicAdd(results.data[0], reciprocal);
}
