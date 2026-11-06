#[compute]
#version 450

// GPU-resident developmental rescue. Edges are appended to a compact
// global rescue log; after BOTH ordered rescue passes, a scatter pass writes
// each record to the slot buffer belonging to its source slice.
// Ordered phases: receivers with zero incoming, then inputs with zero outgoing.
// The compact worklists are already ranked by the CPU neuron-ordering step;
// one GPU invocation processes each list in order so capacity contention
// cannot introduce nondeterministic genotype-to-phenotype differences.
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer Neurons {
    vec4 data[];
} neurons;
layout(set = 0, binding = 1, std430) readonly buffer Roles {
    int data[];
} roles;

struct RescueRecord {
    uint source;
    uint slot;
    int target;
    float h;
};
layout(set = 0, binding = 2, std430) buffer RescueRecords {
    RescueRecord data[];
} records;
layout(set = 0, binding = 3, std430) buffer Counts {
    int data[];
} counts;
layout(set = 0, binding = 4, std430) buffer Incoming {
    int data[];
} incoming;
layout(set = 0, binding = 5, std430) buffer Statistics {
    int data[];
} statistics;
layout(set = 0, binding = 6, std430) readonly buffer Worklist {
    int ids[];
} worklist;

layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint slot_stride;
    vec4 coefficients;
    uint phase;
    uint worklist_count;
    uint padding1;
    uint padding2;
} p;

const float PI = 3.14159265358979323846;

float h_value(float source_a, float target_a) {
    return sin(3.0 * PI / 8.0
        + p.coefficients.x * (sin(source_a)       + cos(target_a))
        + p.coefficients.y * (sin(2.0 * source_a) + cos(2.0 * target_a))
        + p.coefficients.z * (sin(3.0 * source_a) + cos(3.0 * target_a))
        + p.coefficients.w * (sin(4.0 * source_a) + cos(4.0 * target_a)));
}

// Reserve a canonical slot. A single ordered invocation processes each pass;
// CAS also guards capacity if the work scheduling is later parallelized.
int reserve_slot(uint source) {
    while (true) {
        int used = counts.data[source];
        if (used >= int(p.slot_stride)) return -1;
        int previous = atomicCompSwap(counts.data[source], used, used + 1);
        if (previous == used) return used;
    }
}

void rescue_one(uint neuron) {

    bool rescue_receivers = (p.phase == 0u);
    if (rescue_receivers) {
        if (roles.data[neuron] == 1 || incoming.data[neuron] != 0) return;
    } else {
        if (roles.data[neuron] != 1 || counts.data[neuron] != 0) return;
    }

    uint plane = p.I * p.J;
    uint k = neuron / plane;
    uint j = (neuron / p.I) % p.J;
    uint i = neuron % p.I;
    // Increasing spatial-index order, the same tie resolution as the CPU rule.
    int neighbor_offsets[6] = int[6](-int(plane), -int(p.I), -1, 1, int(p.I), int(plane));
    bool valid_neighbor[6] = bool[6](k > 0u, j > 0u, i > 0u,
                                     i + 1u < p.I, j + 1u < p.J, k + 1u < p.K);

    uint candidate_ids[6];
    float candidate_scores[6];
    int candidate_count = 0;
    int structural_count = 0;
    int positive_count = 0;

    for (int n = 0; n < 6; ++n) {
        if (!valid_neighbor[n]) continue;
        uint neighbor = uint(int(neuron) + neighbor_offsets[n]);
        uint source = rescue_receivers ? neighbor : neuron;
        uint target = rescue_receivers ? neuron : neighbor;
        int source_role = roles.data[source];
        int target_role = roles.data[target];
        if (rescue_receivers) {
            if (source_role == 2 || (target_role == 2 && source_role == 1)) continue;
        } else {
            if (target_role != 0) continue;
        }
        structural_count++;
        float source_a = neurons.data[2u * source + 1u].w;
        float target_a = neurons.data[2u * target + 1u].w;
        float h = h_value(source_a, target_a);
        atomicAdd(statistics.data[rescue_receivers ? 0 : 1], 1);
        float h_squared = h * h;
        if (!(h_squared > 0.0)) continue;
        positive_count++;
        candidate_ids[candidate_count] = neighbor;
        candidate_scores[candidate_count] = h_squared;
        candidate_count++;
    }

    // Try candidates by descending H^2. If a concurrent rescue has exhausted
    // the best source, attempt the next-best source instead of exceeding limits.
    bool rescued = false;
    for (int rank = 0; rank < candidate_count && !rescued; ++rank) {
        int best = -1;
        float best_score = 0.0;
        for (int c = 0; c < candidate_count; ++c) {
            if (candidate_scores[c] > best_score) {
                best_score = candidate_scores[c];
                best = c;
            }
        }
        if (best < 0) break;
        uint neighbor = candidate_ids[best];
        candidate_scores[best] = 0.0;
        uint source = rescue_receivers ? neighbor : neuron;
        uint target = rescue_receivers ? neuron : neighbor;
        int slot = reserve_slot(source);
        if (slot < 0) continue;
        // Both worklists are processed serially, in separate synchronized
        // dispatches. The two counters therefore define a dense record log.
        uint record_index = uint(statistics.data[6] + statistics.data[7]);
        records.data[record_index] = RescueRecord(source, uint(slot), int(target), best_score);
        atomicAdd(incoming.data[target], 1);
        atomicAdd(statistics.data[rescue_receivers ? 6 : 7], 1);
        rescued = true;
    }

    if (!rescued) {
        if (structural_count == 0) atomicAdd(statistics.data[2], 1);
        else if (positive_count == 0) atomicAdd(statistics.data[3], 1);
        else if (rescue_receivers) atomicAdd(statistics.data[4], 1);
        else atomicAdd(statistics.data[5], 1);
    }
}

void main() {
    // Each brain has at most a sparse set of isolated neurons. Sequential
    // processing of just this compact set preserves exact capacity ordering.
    for (uint n = 0u; n < p.worklist_count; ++n) {
        rescue_one(uint(worklist.ids[n]));
    }
}
