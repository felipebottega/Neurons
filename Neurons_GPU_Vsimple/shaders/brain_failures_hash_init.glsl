#[compute]
#version 450
// GPU hash table initialization: source = 0xffffffff marks an empty slot.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Entry { uint source; uint target; };
layout(set = 0, binding = 0, std430) buffer HashTable { Entry data[]; } table_entries;
layout(push_constant, std430) uniform Params {
    uint capacity; uint pad0; uint pad1; uint pad2;
} params;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint step = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (; idx < params.capacity; idx += step) {
        table_entries.data[idx].source = 0xffffffffu;
        table_entries.data[idx].target = 0u;
    }
}
