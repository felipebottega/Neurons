#[compute]
#version 450
// In-place heapsort by target ID. Each invocation owns the slots of one source
// neuron in the selected slice; all targets remain GLOBAL spatial indices.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Slot { int target; float weight; float trace; float pad; };
layout(set = 0, binding = 0, std430) buffer FirstSlots { Slot data[]; } a;
layout(set = 0, binding = 1, std430) buffer SecondSlots { Slot data[]; } b;
layout(set = 0, binding = 2, std430) readonly buffer FirstDegrees { int data[]; } degree_a;
layout(set = 0, binding = 3, std430) readonly buffer SecondDegrees { int data[]; } degree_b;
layout(push_constant, std430) uniform Params {
    uint plane; uint slice_id; uint stride_a; uint stride_b;
} p;

void sift_a(uint base, int root, int end) {
    while (2 * root + 1 <= end) {
        int child = 2 * root + 1;
        if (child + 1 <= end && a.data[base + uint(child)].target < a.data[base + uint(child + 1)].target) child++;
        if (a.data[base + uint(root)].target >= a.data[base + uint(child)].target) break;
        Slot tmp = a.data[base + uint(root)];
        a.data[base + uint(root)] = a.data[base + uint(child)];
        a.data[base + uint(child)] = tmp;
        root = child;
    }
}
void sift_b(uint base, int root, int end) {
    while (2 * root + 1 <= end) {
        int child = 2 * root + 1;
        if (child + 1 <= end && b.data[base + uint(child)].target < b.data[base + uint(child + 1)].target) child++;
        if (b.data[base + uint(root)].target >= b.data[base + uint(child)].target) break;
        Slot tmp = b.data[base + uint(root)];
        b.data[base + uint(root)] = b.data[base + uint(child)];
        b.data[base + uint(child)] = tmp;
        root = child;
    }
}
void sort_a(uint base, int count) {
    if (count < 2) return;
    for (int start = (count - 2) / 2; start >= 0; --start) sift_a(base, start, count - 1);
    for (int end = count - 1; end > 0; --end) {
        Slot tmp = a.data[base];
        a.data[base] = a.data[base + uint(end)];
        a.data[base + uint(end)] = tmp;
        sift_a(base, 0, end - 1);
    }
}
void sort_b(uint base, int count) {
    if (count < 2) return;
    for (int start = (count - 2) / 2; start >= 0; --start) sift_b(base, start, count - 1);
    for (int end = count - 1; end > 0; --end) {
        Slot tmp = b.data[base];
        b.data[base] = b.data[base + uint(end)];
        b.data[base + uint(end)] = tmp;
        sift_b(base, 0, end - 1);
    }
}
void main() {
    uint local_id = gl_GlobalInvocationID.x;
    if (local_id >= p.plane) return;
    uint global_id = p.slice_id * p.plane + local_id;
    sort_a(local_id * p.stride_a, degree_a.data[global_id]);
    sort_b(local_id * p.stride_b, degree_b.data[global_id]);
}
