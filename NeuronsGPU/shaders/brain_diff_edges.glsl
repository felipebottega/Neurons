#[compute]
#version 450
// Two modes: 0 finds shared targets and the maximum |weight difference|;
// 1 computes 20-bin weight-difference histograms for the UNION of edges.
// Missing connections have weight 0, matching the original C++ Diff.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Slot { int target; float weight; float trace; float pad; };
layout(set = 0, binding = 0, std430) readonly buffer FirstSlots { Slot data[]; } a;
layout(set = 0, binding = 1, std430) readonly buffer SecondSlots { Slot data[]; } b;
layout(set = 0, binding = 2, std430) readonly buffer FirstDegrees { int data[]; } degree_a;
layout(set = 0, binding = 3, std430) readonly buffer SecondDegrees { int data[]; } degree_b;
layout(set = 0, binding = 4, std430) buffer Shared { uint data[]; } shared_counter;
layout(set = 0, binding = 5, std430) buffer Maximum { uint data[]; } maximum;
layout(set = 0, binding = 6, std430) buffer Histograms { uint data[]; } bins;
layout(push_constant, std430) uniform Params {
    uint plane; uint slice_id; uint stride_a; uint stride_b;
    uint mode; uint padding0; uint padding1; uint padding2;
} p;

int find_b(uint base, int count, int target) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int middle = lo + (hi - lo) / 2;
        if (b.data[base + uint(middle)].target < target) lo = middle + 1;
        else hi = middle;
    }
    if (lo < count && b.data[base + uint(lo)].target == target) return lo;
    return -1;
}
int find_a(uint base, int count, int target) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int middle = lo + (hi - lo) / 2;
        if (a.data[base + uint(middle)].target < target) lo = middle + 1;
        else hi = middle;
    }
    if (lo < count && a.data[base + uint(lo)].target == target) return lo;
    return -1;
}
uint weight_bin(float difference, float max_abs) {
    float limit = max_abs > 0.0 ? max_abs : 1.0;
    float scaled = (difference + limit) / (2.0 * limit);
    return uint(clamp(int(floor(scaled * 20.0)), 0, 19));
}
void main() {
    uint local_id = gl_GlobalInvocationID.x;
    if (local_id >= p.plane) return;
    uint source = p.slice_id * p.plane + local_id;
    int na = degree_a.data[source], nb = degree_b.data[source];
    uint base_a = local_id * p.stride_a;
    uint base_b = local_id * p.stride_b;
    uint shared_connections = 0u;
    uint local_bins[20];
    for (uint i = 0u; i < 20u; ++i) local_bins[i] = 0u;
    float local_max = 0.0;
    float global_max = uintBitsToFloat(maximum.data[0]);
    for (int c = 0; c < na; ++c) {
        Slot entry = a.data[base_a + uint(c)];
        int match_id = find_b(base_b, nb, entry.target);
        float difference = entry.weight - (match_id >= 0 ? b.data[base_b + uint(match_id)].weight : 0.0);
        if (p.mode == 0u) {
            if (match_id >= 0) shared_connections++;
            local_max = max(local_max, abs(difference));
        } else {
            local_bins[weight_bin(difference, global_max)]++;
        }
    }
    for (int c = 0; c < nb; ++c) {
        Slot entry = b.data[base_b + uint(c)];
        if (find_a(base_a, na, entry.target) >= 0) continue;
        float difference = -entry.weight;
        if (p.mode == 0u) local_max = max(local_max, abs(difference));
        else local_bins[weight_bin(difference, global_max)]++;
    }
    if (p.mode == 1u) {
        for (uint bin = 0u; bin < 20u; ++bin)
            if (local_bins[bin] != 0u) atomicAdd(bins.data[p.slice_id * 20u + bin], local_bins[bin]);
    }
    if (p.mode == 0u) {
        if (shared_connections > 0u) atomicAdd(shared_counter.data[0], shared_connections);
        if (local_max > 0.0) atomicMax(maximum.data[0], floatBitsToUint(local_max));
    }
}
