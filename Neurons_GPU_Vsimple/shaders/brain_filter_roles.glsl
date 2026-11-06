#[compute]
#version 450

// Remove forbidden connections AFTER the pre-role graph and the degree-based
// selection of inputs/outputs. Reuses the already evaluated H values in slots:
// this pass does not evaluate H, build candidates, or change N_max.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Slot {
    int target;
    float h;
    float trace;
    float pad;
};
layout(set = 0, binding = 0, std430) buffer Slots {
    Slot data[];
} slots;
layout(set = 0, binding = 1, std430) readonly buffer Roles {
    int data[];
} roles;
layout(set = 0, binding = 2, std430) buffer Counts {
    int data[];
} counts;
layout(set = 0, binding = 3, std430) buffer Incoming {
    int data[];
} incoming;
layout(set = 0, binding = 4, std430) buffer LocalPassCount {
    uint edges;
} local_pass_count;

layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint slot_stride;
    uint slice_k;
    uint radius_i;
    uint radius_j;
    uint radius_k;
} p;

void main() {
    uint plane = p.I * p.J;
    uint local_source = gl_GlobalInvocationID.x;
    if (local_source >= plane) return;
    uint source = p.slice_k * plane + local_source;
    uint base = local_source * p.slot_stride;
    uint previous_count = uint(counts.data[source]);
    int source_role = roles.data[source];

    uint source_i = source % p.I;
    uint source_j = (source / p.I) % p.J;
    uint source_k = source / plane;
    uint kept = 0u;
    uint local_kept = 0u;
    for (uint c = 0u; c < previous_count; ++c) {
        Slot edge = slots.data[base + c];
        uint target = uint(edge.target);
        int target_role = roles.data[target];
        // Only internal neurons may be receivers of input connections;
        // inputs never receive, outputs never send, and outputs only receive
        // from internal neurons. This also excludes all interface-to-interface.
        bool allowed = source_role != 2 && target_role != 1 &&
                       (source_role != 1 || target_role == 0) &&
                       (target_role != 2 || source_role == 0);
        if (!allowed) continue;
        slots.data[base + kept] = edge;
        kept++;
        atomicAdd(incoming.data[target], 1);
        // In the constructor's second pass, every target is strictly outside
        // the local neighborhood; spatial bounds reproduce the local/extra
        // classification exactly, without adding a per-slot flag.
        uint ti = target % p.I;
        uint tj = (target / p.I) % p.J;
        uint tk = target / plane;
        if (abs(int(ti) - int(source_i)) <= int(p.radius_i) &&
            abs(int(tj) - int(source_j)) <= int(p.radius_j) &&
            abs(int(tk) - int(source_k)) <= int(p.radius_k)) {
            local_kept++;
        }
    }
    for (uint c = kept; c < previous_count; ++c)
        slots.data[base + c] = Slot(-1, -1.0, -1.0, -1.0);
    counts.data[source] = int(kept);
    if (local_kept != 0u) atomicAdd(local_pass_count.edges, local_kept);
}
