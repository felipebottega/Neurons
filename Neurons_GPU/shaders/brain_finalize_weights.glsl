#[compute]
#version 450

// Apply the source-dependent attenuation after the developmental rescue
// determines psi. Keep every live slot resident in GPU memory.
// One invocation per source neuron; updates only its own connection slots.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer Neurons {
    vec4 data[];
} neurons;
layout(set = 0, binding = 1, std430) readonly buffer Roles {
    int data[];
} roles;
struct Slot {
    int target;
    float h;
    float trace;
    float pad;
};
layout(set = 0, binding = 2, std430) buffer Slots {
    Slot data[];
} slots;
layout(set = 0, binding = 3, std430) readonly buffer Counts {
    int data[];
} counts;
layout(set = 0, binding = 4, std430) writeonly buffer SourceSums {
    float data[];
} source_sums;
layout(set = 0, binding = 5, std430) writeonly buffer SourceBins {
    int data[];
} source_bins;

layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint slot_count;
    float input_floor;
    uint slice_k;
    uint statistics_level;
    float padding2;
} p;

const uint BIN_COUNT = 20u;
void main() {
    uint plane = p.I * p.J;
    uint local_source = gl_GlobalInvocationID.x;
    if (local_source >= plane) return;
    uint source = p.slice_k * plane + local_source;

    bool want_sum = p.statistics_level >= 1u;
    bool want_bins = p.statistics_level >= 2u;
    int bins[20];
    if (want_bins) {
        for (uint b = 0u; b < BIN_COUNT; ++b) bins[b] = 0;
    }
    float a = neurons.data[2u * source + 1u].w;
    float attenuation = 1.0 - a * a;
    bool input_source = roles.data[source] == 1;
    float sum_weights = 0.0;
    uint base = local_source * p.slot_count;
    uint count = uint(counts.data[source]);
    for (uint c = 0u; c < count; ++c) {
        Slot entry = slots.data[base + c];
        float weight = max(entry.h * attenuation, 0.1);
        if (input_source) weight = max(weight, p.input_floor);
        entry.h = weight;
        // Construction tags are no longer needed. Runtime reuses this scalar
        // as decay rounding debt, including for developmental rescue edges.
        entry.pad = 0.0;
        slots.data[base + c] = entry;
        if (want_sum) sum_weights += weight;
        if (want_bins) {
            uint bin = uint(clamp(int(weight * float(BIN_COUNT)), 0, int(BIN_COUNT) - 1));
            bins[bin] += 1;
        }
    }
    if (want_sum) source_sums.data[source] = sum_weights;
    if (want_bins) {
        for (uint b = 0u; b < BIN_COUNT; ++b)
            source_bins.data[source * BIN_COUNT + b] = bins[b];
    }
}
