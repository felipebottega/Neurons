#[compute]
#version 450

// Two-pass initial connectivity for each source neuron:
// (1) top-Nmax H-positive targets in the genetically bounded local region;
// (2) if slots remain, up to Nmax deterministic signed candidate tests using
//     offsets shared across neurons (cyclic-combination order).
// Input/output filtering intentionally happens AFTER both selection passes.
// Every invocation owns its heap, candidate tracking, and output slots.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer Neurons {
    vec4 data[];
} neurons;
layout(set = 0, binding = 1, std430) readonly buffer Roles {
    int data[]; // 0 = internal, 1 = input, 2 = output
} roles;

struct Slot {
    int target;    // spatial index, -1 = unused
    float h;       // UNATTENUATED H, until filtering/rescue have finished
    float trace;   // initially zero
    float pad;
};
layout(set = 0, binding = 2, std430) buffer Slots {
    Slot data[];
} slots;
layout(set = 0, binding = 3, std430) writeonly buffer Counts {
    int data[]; // Outgoing degree AFTER role filtering
} counts;
layout(set = 0, binding = 4, std430) buffer Incoming {
    int data[]; // Must be zero initialized before dispatch
} incoming;
layout(set = 0, binding = 5, std430) readonly buffer CandidateOffsets {
    ivec4 data[]; // (di, dj, dk, unused): precomputed once per brain
} candidate_offsets;
// Single zero-initialized aggregate. Count only local edges that survive
// the common role filter; the complementary total is derived on the CPU.
layout(set = 0, binding = 6, std430) buffer LocalPassCount {
    uint edges;
} local_pass_count;

layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint slot_stride;
    vec4 coefficients;
    uint initial_limit;
    uint radius_i;
    uint radius_j;
    uint tuple_count;
    uint attempt_budget; // Genetic Nmax; independent of physical initial_limit.
    uint slice_k;
    uint radius_k; // Appended to preserve the preceding push-constant offsets.
} p;

const float PI = 3.14159265358979323846;

float h_value(float source_a, float target_a) {
    // Identical coefficient grouping to FunctionsNative::h_coefficients_value.
    return sin(3.0 * PI / 8.0
        + p.coefficients.x * (sin(source_a)       + cos(target_a))
        + p.coefficients.y * (sin(2.0 * source_a) + cos(2.0 * target_a))
        + p.coefficients.z * (sin(3.0 * source_a) + cos(3.0 * target_a))
        + p.coefficients.w * (sin(4.0 * source_a) + cos(4.0 * target_a)));
}

// A min-heap keeps precisely the greatest slot_count H-positive candidates.
// Tie-breaking by target index is deterministic; H ties are not ordered in the article.
bool less_than(Slot a, Slot b) {
    return a.h < b.h || (a.h == b.h && a.target > b.target);
}

// Open-addressed duplicate detection for the additional pass. The otherwise
// unused Slot.pad fields act as a temporary, per-source hash table; this avoids
// an O(Nmax) connection-list scan for every candidate. Each source invocation
// owns its entire slot range, so no atomics are required. The pad fields of
// all retained normal connections are reset to 0 before role filtering; rescue
// uses pad=1 until weight finalization clears it for runtime decay compensation.
// A zero bit pattern means an empty hash bucket. Actual entries encode
// target+1 by bit-cast, preserving all integral IDs representable by the GPU.
bool previously_tested_or_mark(uint base, uint target) {
    uint bucket = (target * 2654435761u) % p.slot_stride;
    while (true) {
        uint stored = floatBitsToUint(slots.data[base + bucket].pad);
        if (stored == target + 1u) return true;
        if (stored == 0u) {
            slots.data[base + bucket].pad = uintBitsToFloat(target + 1u);
            return false;
        }
        bucket = bucket + 1u == p.slot_stride ? 0u : bucket + 1u;
    }
}

void main() {
    uint plane = p.I * p.J;
    uint local_source = gl_GlobalInvocationID.x;
    if (local_source >= plane) return;
    uint source = p.slice_k * plane + local_source;

    uint base = local_source * p.slot_stride;
    for (uint index = 0u; index < p.slot_stride; ++index) {
        slots.data[base + index] = Slot(-1, -1.0, -1.0, 0.0);
    }
    if (roles.data[source] == 2) {
        counts.data[source] = 0;
        return;
    }

    uint source_k = source / (p.I * p.J);
    uint source_j = (source / p.I) % p.J;
    uint source_i = source % p.I;
    uint min_k = source_k > p.radius_k ? source_k - p.radius_k : 0u;
    uint max_k = min(p.K - 1u, source_k + p.radius_k);
    uint min_i = source_i > p.radius_i ? source_i - p.radius_i : 0u;
    uint max_i = min(p.I - 1u, source_i + p.radius_i);
    uint min_j = source_j > p.radius_j ? source_j - p.radius_j : 0u;
    uint max_j = min(p.J - 1u, source_j + p.radius_j);
    float source_a = neurons.data[2u * source + 1u].w;
    uint count = 0u;

    // Local pass: retain only the greatest H values in a min-heap.
    for (uint w = min_k; w <= max_k; ++w) {
        for (uint v = min_j; v <= max_j; ++v) {
            for (uint u = min_i; u <= max_i; ++u) {
                uint target = u + p.I * (v + p.J * w);
                if (target == source) continue;
                float h = h_value(source_a, neurons.data[2u * target + 1u].w);
                if (h <= 0.0) continue;

                Slot candidate = Slot(int(target), h, 0.0, 0.0);
                if (count < p.initial_limit) {
                    // Insert in the heap.
                    uint child = count;
                    count++;
                    while (child > 0u) {
                        uint parent = (child - 1u) / 2u;
                        Slot parent_value = slots.data[base + parent];
                        if (!less_than(candidate, parent_value)) break;
                        slots.data[base + child] = parent_value;
                        child = parent;
                    }
                    slots.data[base + child] = candidate;
                } else if (less_than(slots.data[base], candidate)) {
                    // Replace minimum and restore min-heap.
                    uint parent = 0u;
                    while (true) {
                        uint left = 2u * parent + 1u;
                        if (left >= count) break;
                        uint right = left + 1u;
                        uint child = left;
                        if (right < count && less_than(slots.data[base + right], slots.data[base + left]))
                            child = right;
                        Slot child_value = slots.data[base + child];
                        if (!less_than(child_value, candidate)) break;
                        slots.data[base + parent] = child_value;
                        parent = child;
                    }
                    slots.data[base + parent] = candidate;
                }
            }
        }
    }
    // The local heap occupies [0, local_end); accepted second-pass candidates
    // are appended after it. Preserve this boundary through role filtering.
    uint local_end = count;
    // Additional pass: the genetic Nmax is the attempt budget, regardless
    // of how many distinct reachable targets fit the physical slot cap.
    // Invalid, repeated and H-nonpositive candidates still consume attempts.
    uint tested = 0u;
    for (uint t = 0u; t < p.tuple_count && tested < p.attempt_budget && count < p.initial_limit; ++t) {
        ivec3 delta = candidate_offsets.data[t].xyz;
        for (int signs = 0; signs < 8 && tested < p.attempt_budget && count < p.initial_limit; ++signs) {
            tested++;
            // (+++), (++-), (+-+), (+--), (-++), (-+-), (--+), (---).
            int dx = (signs & 4) == 0 ? delta.x : -delta.x;
            int dy = (signs & 2) == 0 ? delta.y : -delta.y;
            int dz = (signs & 1) == 0 ? delta.z : -delta.z;
            ivec3 destination = ivec3(source_i, source_j, source_k) + ivec3(dx, dy, dz);
            if (any(lessThan(destination, ivec3(0))) ||
                any(greaterThanEqual(destination, ivec3(p.I, p.J, p.K)))) continue;
            uint target = uint(destination.x) + p.I * (uint(destination.y) + p.J * uint(destination.z));
            if (target == source || previously_tested_or_mark(base, target)) continue;
            float h = h_value(source_a, neurons.data[2u * target + 1u].w);
            if (!(h > 0.0)) continue;
            // Preserve hash buckets that happen to occupy a newly active slot.
            float hash_bucket = slots.data[base + count].pad;
            slots.data[base + count] = Slot(int(target), h, 0.0, hash_bucket);
            count++;
        }
    }
    // Slot.pad is a scratch hash table only during this pass. The finalized
    // normal edges must carry pad=0; developmental rescue later uses pad=1.
    for (uint c = 0u; c < count; ++c) {
        slots.data[base + c].pad = 0.0;
    }

    // Preserve the combined ordinary selection before role filtering.
    // Each invocation owns its slots and outgoing count; incoming degrees
    // require atomic increments because multiple sources share a target.
    uint kept = 0u;
    uint kept_local = 0u;
    for (uint c = 0u; c < count; ++c) {
        Slot candidate = slots.data[base + c];
        int target_role = roles.data[uint(candidate.target)];
        bool allowed = (target_role != 1) && !(roles.data[source] == 1 && target_role == 2);
        if (allowed) {
            if (c < local_end) kept_local++;
            slots.data[base + kept] = candidate;
            kept++;
            atomicAdd(incoming.data[uint(candidate.target)], 1);
        }
    }
    for (uint c = kept; c < count; ++c) {
        slots.data[base + c] = Slot(-1, -1.0, -1.0, -1.0);
    }
    counts.data[source] = int(kept);
    if (kept_local > 0u) atomicAdd(local_pass_count.edges, kept_local);
}
