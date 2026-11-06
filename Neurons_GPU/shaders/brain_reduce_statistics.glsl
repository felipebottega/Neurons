#[compute]
#version 450

// Optional reduction of source sums and/or 20-bin histograms.
// Sum-only mode avoids all histogram work; NONE skips this shader entirely.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer SourceSums {
    float data[];
} source_sums;
layout(set = 0, binding = 1, std430) readonly buffer SourceBins {
    int data[];
} source_bins;
layout(set = 0, binding = 2, std430) writeonly buffer SliceSums {
    float data[];
} slice_sums;
layout(set = 0, binding = 3, std430) writeonly buffer SliceBins {
    int data[];
} slice_bins;
layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint statistics_level;
} p;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (p.statistics_level == 1u) {
        // Sum-only path: one invocation per slice, no bin reads or writes.
        if (id >= p.K) return;
        uint base = id * p.I * p.J;
        float sum_weights = 0.0;
        for (uint n = 0u; n < p.I * p.J; ++n)
            sum_weights += source_sums.data[base + n];
        slice_sums.data[id] = sum_weights;
        return;
    }
    if (id >= p.K * 20u) return;
    uint slice = id / 20u;
    uint bin = id % 20u;
    uint base = slice * p.I * p.J;
    int sum_bins = 0;
    float sum_weights = 0.0;
    for (uint n = 0u; n < p.I * p.J; ++n) {
        uint source = base + n;
        sum_bins += source_bins.data[source * 20u + bin];
        if (bin == 0u) sum_weights += source_sums.data[source];
    }
    slice_bins.data[id] = sum_bins;
    if (bin == 0u) slice_sums.data[slice] = sum_weights;
}
